import os

import numpy as np
import pandas as pd

from cellmaps_vnn.annotate import VNNAnnotate
from cellmaps_vnn.predict_mlflow import VNNPredict
from cellmaps_vnn.train_mlflow import VNNTrain
import mlflow


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


config = {
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
    "epoch": 25,
    "lr": 0.001,
    "wd": 0.001,
    "alpha": 0.3,
    "genotype_hiddens": 4,
    "patience": 30,
    "delta": 0.001,
    "min_dropout_layer": 2,
    "dropout_fraction": 0.3
}

base_outdir = config["outdir"]
train_outdir = os.path.join(config["outdir"], "train")
predict_outdir = os.path.join(config["outdir"], "predict")
config["model_predictions"] = [predict_outdir]

annotate_outdir = os.path.join(config["outdir"], "annotate")
os.makedirs(train_outdir, exist_ok=True)
os.makedirs(predict_outdir, exist_ok=True)
os.makedirs(annotate_outdir, exist_ok=True)

experiment_name = "vnn-challenge-experiment"
mlflow.set_experiment(experiment_name)

with mlflow.start_run() as parent_run:
    mlflow.log_params(config)
    
    config["outdir"] = train_outdir
    train_config = AttrDict(config)

    train_cmd = VNNTrain(train_config)
    train_cmd.run()

    mlflow.log_artifact(os.path.join(train_outdir,"hierarchy.cx2"))
    mlflow.log_artifact(os.path.join(train_outdir,"model_final.pt"))

    config["outdir"] = predict_outdir
    config["inputdir"] = train_outdir
    config["hierarchy"] = os.path.join(train_outdir, "hierarchy.cx2")
    config["std"] = None
    predict_config = AttrDict(config)
    
    predict_cmd = VNNPredict(predict_config)
    predict_cmd.run()

    mlflow.log_artifact(os.path.join(predict_outdir,"gene_rho.out"))
    mlflow.log_artifact(os.path.join(predict_outdir,"rlipp.out"))
    mlflow.log_artifact(os.path.join(predict_outdir,"predict.txt"))

    with open(os.path.join(predict_outdir,"predict.txt"), 'r') as file:
        lines = [line.strip() for line in file]
        first_five = lines[:5]
        for line in first_five:
            print(line.strip())
    
    predict_file_path = os.path.join(predict_outdir, "predict.txt")
    y_pred = np.loadtxt(predict_file_path)

    predict_data_file = config["predict_data"]
    predict_data = pd.read_csv(predict_data_file, delimiter="\t", header=None)
    y_true = predict_data.iloc[:, 2].values

    mse = np.mean((y_true - y_pred) ** 2)
    print(mse)

    config["outdir"] = annotate_outdir
    config["inputdir"] = predict_outdir
    annotate_config = AttrDict(config)
    annotate_cmd = VNNAnnotate(annotate_config)
    annotate_cmd.run()

    mlflow.log_artifact(os.path.join(annotate_outdir,"rlipp.out"))
    mlflow.log_artifact(os.path.join(annotate_outdir,"hierarchy.cx2"))
