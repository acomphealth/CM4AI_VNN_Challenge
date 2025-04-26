import os
from itertools import product
import mlflow
import optuna

import pandas as pd
import numpy as np

from cellmaps_vnn.annotate import VNNAnnotate
from cellmaps_vnn.predict_mlflow import VNNPredict
from cellmaps_vnn.train_mlflow import VNNTrain

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

base_config = {
    "outdir": "output",
    "inputdir": "data",
    "training_data": "data/training_data.txt",
    "predict_data": "data/test_data.txt",
    "gene2id": "data/gene2ind.txt",
    "cell2id": "data/cell2ind.txt",
    "mutations": "data/cell2mutation.txt",
    "cn_deletions": "data/cell2cndeletion.txt",
    "cn_amplifications": "data/cell2cnamplification.txt",
    "batchsize": 64,
    "cuda": 0,
    "zscore_method": "auc",
    "std": "std.txt",
    "gene_attribute_name": "CD_MemberList",
    "optimize": 1,
    "cpu_count": 12,
    "parent_network": None,
    "disease": None,
    "hierarchy": None,
    "ndexserver": None,
    "ndexuser": None,
    "ndexpassword": None,
    "drug_count": 0,
    "genotype_hiddens": 4,
    "patience": 30,
    "delta": 0.001,
    "min_dropout_layer": 2,
    "dropout_fraction": 0.3
}


experiment_name = "vnn-challenge-hyperparameter-optimization"
mlflow.set_experiment(experiment_name)

def calculate_error(y_true, y_pred):

    mse = np.mean((y_true - y_pred) ** 2)
    return mse

optuna_mlflow_mapping = {}

def train(params):
    config = base_config.copy()
    config.update(params)

    base_outdir = config["outdir"]
    train_outdir = os.path.join(config["outdir"], "train")
    predict_outdir = os.path.join(config["outdir"], "predict")
    config["model_predictions"] = [predict_outdir]

    annotate_outdir = os.path.join(config["outdir"], "annotate")
    os.makedirs(train_outdir, exist_ok=True)
    os.makedirs(predict_outdir, exist_ok=True)
    os.makedirs(annotate_outdir, exist_ok=True)

    with mlflow.start_run() as run:
        mlflow.log_params(config)

        config["outdir"] = train_outdir
        train_config = AttrDict(config)
        train_cmd = VNNTrain(train_config)
        train_cmd.run()

        mlflow.log_artifact(os.path.join(train_outdir, "hierarchy.cx2"))
        mlflow.log_artifact(os.path.join(train_outdir, "model_final.pt"))

        config["outdir"] = predict_outdir
        config["inputdir"] = train_outdir
        config["hierarchy"] = os.path.join(train_outdir, "hierarchy.cx2")
        config["std"] = None
        predict_config = AttrDict(config)
        
        predict_cmd = VNNPredict(predict_config)
        predict_cmd.run()

        mlflow.log_artifact(os.path.join(predict_outdir, "predict.txt"))
        mlflow.log_artifact(os.path.join(predict_outdir,"gene_rho.out"))
        mlflow.log_artifact(os.path.join(predict_outdir,"rlipp.out"))

        predict_file_path = os.path.join(predict_outdir, "predict.txt")
        y_pred = np.loadtxt(predict_file_path)

        predict_data_file = config["predict_data"]
        predict_data = pd.read_csv(predict_data_file, delimiter="\t", header=None)
        y_true = predict_data.iloc[:, 2].values

        error = calculate_error(y_true, y_pred)
        mlflow.log_metric("mse", error)

        config["outdir"] = annotate_outdir
        config["inputdir"] = predict_outdir
        annotate_config = AttrDict(config)
        annotate_cmd = VNNAnnotate(annotate_config)
        annotate_cmd.run()

        mlflow.log_artifact(os.path.join(annotate_outdir, "rlipp.out"))
        mlflow.log_artifact(os.path.join(annotate_outdir, "hierarchy.cx2"))

        return error, run.info.run_id

def objective(trial):
    params = {
        "epoch": trial.suggest_categorical("epoch", [10, 25]),
        "lr": trial.suggest_categorical("lr", [0.01, 0.001]),
        "wd": trial.suggest_categorical("wd", [0.01, 0.001]),
        "alpha": trial.suggest_categorical("alpha", [0.1, 0.3]),
        "trial_number": trial.number
    }
    
#    params = {
#        "epoch": trial.suggest_int("epoch", 20, 50),  # search between 20 and 50 epochs
#        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),  # search learning rate between 0.0001 and 0.01, log scale
#        "wd": trial.suggest_float("wd", 1e-5, 1e-2, log=True),  # search weight decay between 0.00001 and 0.01, log scale
#        "alpha": trial.suggest_float("alpha", 0.01, 0.3),  # search alpha between 0.01 and 0.3
#        "trial_number": trial.number
#    }

    error, run_id = train(params)
    optuna_mlflow_mapping[trial.number] = run_id

    return error

if __name__ == "__main__":
    search_space = {
        "epoch": [10, 25],
        "lr": [0.01, 0.001],
        "wd": [0.01, 0.001],        
        "alpha": [0.1, 0.3]
    }

    n_trials = len(list(product(*search_space.values())))

#    n_trials = 5

    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(direction='minimize', sampler=sampler)

#    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params["best_study_id"] = optuna_mlflow_mapping[study.best_trial._trial_id]
    print("Best hyperparameters:", best_params)
    mlflow.log_dict(best_params, "best_params.json")

