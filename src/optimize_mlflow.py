import os
import mlflow
import optuna

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

def objective(trial):
    config = base_config.copy()

    config["epoch"] = trial.suggest_int("epoch", 10, 100)
    config["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    config["wd"] = trial.suggest_float("wd", 1e-5, 1e-2, log=True)
    config["alpha"] = trial.suggest_float("alpha", 0.1, 1.0)

    base_outdir = config["outdir"]
    train_outdir = os.path.join(config["outdir"], "train")
    predict_outdir = os.path.join(config["outdir"], "predict")
    config["model_predictions"] = [predict_outdir]

    annotate_outdir = os.path.join(config["outdir"], "annotate")
    os.makedirs(train_outdir, exist_ok=True)
    os.makedirs(predict_outdir, exist_ok=True)
    os.makedirs(annotate_outdir, exist_ok=True)

    with mlflow.start_run():
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

        min_loss_path = os.path.join(train_outdir, "min_val_loss.txt")
        if os.path.exists(min_loss_path):
            with open(min_loss_path, "r") as f:
                min_val_loss = float(f.read().strip())
        else:
            min_val_loss = float("inf")

        mlflow.log_metric("min_val_loss", min_val_loss)

        config["outdir"] = annotate_outdir
        config["inputdir"] = predict_outdir
        annotate_config = AttrDict(config)
        annotate_cmd = VNNAnnotate(annotate_config)
        annotate_cmd.run()

        mlflow.log_artifact(os.path.join(annotate_outdir, "rlipp.out"))
        mlflow.log_artifact(os.path.join(annotate_outdir, "hierarchy.cx2"))

    return min_val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    print(study.best_trial)
