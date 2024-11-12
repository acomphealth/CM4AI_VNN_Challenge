import os

from cellmaps_vnn.annotate import VNNAnnotate
from cellmaps_vnn.predict import VNNPredict
from cellmaps_vnn.train import VNNTrain
import wandb


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
    "epoch": 5,
    "lr": 0.001,
    "wd": 0.001,
    "alpha": 0.3,
    "genotype_hiddens": 4,
    "optimize": 1,
    "patience": 30,
    "delta": 0.001,
    "min_dropout_layer": 2,
    "dropout_fraction": 0.3,
    "cpu_count": 12,
    "drug_count": 0,
    "hierarchy": None
}

base_outdir = config["outdir"]
train_outdir = os.path.join(config["outdir"], "train")
predict_outdir = os.path.join(config["outdir"], "predict")
config["model_predictions"] = [predict_outdir]
annotate_outdir = os.path.join(config["outdir"], "annotate")
os.makedirs(train_outdir, exist_ok=True)
os.makedirs(predict_outdir, exist_ok=True)
os.makedirs(annotate_outdir, exist_ok=True)

with wandb.init(entity="b2ai-cm4ai", project="vnn-challenge-wls", config=config) as run:
    config["outdir"] = train_outdir
    train_config = AttrDict(config)

    train_cmd = VNNTrain(train_config)
    train_cmd.run()
    wandb.save(os.path.join(train_outdir,"hierarchy.cx2"))
    wandb.save(os.path.join(train_outdir,"model_final.pt"))

    config["outdir"] = predict_outdir
    config["inputdir"] = train_outdir
    config["std"] = None
    predict_config = AttrDict(config)
    
    predict_cmd = VNNPredict(predict_config)
    predict_cmd.run()

    config["outdir"] = annotate_outdir
    config["inputdir"] = predict_outdir
    annotate_config = AttrDict(config)
    annotate_cmd = VNNAnnotate(annotate_config)
    annotate_cmd.run()