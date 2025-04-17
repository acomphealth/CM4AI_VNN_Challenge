import time
import logging
import torch.utils.data as du
from torch.autograd import Variable

from cellmaps_vnn.data_wrapper import *
from cellmaps_vnn.vnn import *
from cellmaps_vnn.ccc_loss import *
import mlflow

logger = logging.getLogger(__name__)


class VNNTrainer:
    TRAINING_PROGRESS_FILE = 'training_progress.tsv'

    def __init__(self, data_wrapper):
        """
        Initialize the VNN Trainer.

        :param data_wrapper: data wrapper containing data necessary for training
        :type data_wrapper: TrainingDataWrapper
        """
        self.model = None
        self.data_wrapper = data_wrapper
        self.train_feature = self.data_wrapper.train_feature
        self.train_label = self.data_wrapper.train_label
        self.val_feature = self.data_wrapper.val_feature
        self.val_label = self.data_wrapper.val_label
        self.use_cuda = torch.cuda.is_available() and self.data_wrapper.cuda is not None

    def _to_device(self, tensor):
        """
        Moves a tensor to the appropriate device (GPU or CPU).

        :param tensor: a tensor to be moved.
        :type tensor: torch.Tensor
        :returns: a tensor on GPU or CPU.
        :rtype: torch.Tensor
        """
        if self.use_cuda:
            return tensor.cuda(self.data_wrapper.cuda)
        return tensor

    def train_model(self):
        """
        Trains the VNN model.

        :returns min_loss: The minimum validation loss achieved during training.
        :rtype min_loss: float
        """
        self.model = VNN(self.data_wrapper)
        if self.use_cuda:
            self.model.cuda(self.data_wrapper.cuda)

        term_mask_map = util.create_term_mask(
            self.model.term_direct_gene_map, self.model.gene_dim, self.data_wrapper.cuda
        )
        self._initialize_model_parameters(term_mask_map)

        train_loader = du.DataLoader(du.TensorDataset(self.train_feature, self.train_label),
                                     batch_size=self.data_wrapper.batchsize, shuffle=True, drop_last=True)
        val_loader = du.DataLoader(du.TensorDataset(self.val_feature, self.val_label),
                                   batch_size=self.data_wrapper.batchsize, shuffle=True)

        optimizer = self._configure_optimizer()

        with open(os.path.join(self.data_wrapper.outdir, VNNTrainer.TRAINING_PROGRESS_FILE), "w") as f:
            f.write("epoch\ttrain_corr\ttrain_loss\ttrue_auc\tpred_auc\tval_corr\tval_loss\tgrad_norm\telapsed_time\n")
            min_loss = None

            for epoch in range(self.data_wrapper.epochs):
                epoch_start_time = time.time()
                train_predict, total_loss, gradnorms, train_label_gpu = self._train_epoch(train_loader, optimizer,
                                                                                          term_mask_map)
                val_predict, val_loss, val_label_gpu = self._validate_epoch(val_loader)

                train_corr, val_corr, true_auc, pred_auc = self._calculate_metrics(train_predict, train_label_gpu,
                                                                                   val_predict, val_label_gpu)
                epoch_end_time = time.time()

                elapsed_time = epoch_end_time - epoch_start_time
                f.write(f"{epoch}\t{train_corr:.4f}\t{total_loss:.4f}\t{true_auc:.4f}\t{pred_auc:.4f}\t"
                            f"{val_corr:.4f}\t{val_loss:.4f}\t{gradnorms:.4f}\t{elapsed_time:.4f}\n")

                metrics = {
                    "epoch": epoch,
                    "train_corr": train_corr,
                    "total_loss": total_loss,
                    "true_auc": true_auc,
                    "pred_auc": pred_auc,
                    "val_corr": val_corr,
                    "val_loss": val_loss,
                    "gradnorms": gradnorms,
                    "elapsed_time": elapsed_time
                }

                mlflow.log_metics(metrics, step=epoch)

                min_loss = self._save_model_if_improved(min_loss, val_loss, epoch)

        mlflow.log_metric("min_val_loss", min_loss)
        return min_loss

    def _train_epoch(self, train_loader, optimizer, term_mask_map):
        """
        Conducts a training epoch over the training dataset.

        :param train_loader: Data loader for training data.
        :type train_loader: DataLoader
        :param optimizer: Optimizer for the model.
        :type optimizer: Optimizer
        :param term_mask_map: Mask map for term weights.
        :type term_mask_map: Tensor

        :returns : Tuple containing training predictions, total loss, and gradient norms.
        :rtype : Tuple[Tensor, float, float]
        """
        self.model.train()

        train_predict = torch.zeros(0, 0)
        if self.use_cuda:
            train_predict = train_predict.cuda(self.data_wrapper.cuda)
        # tensor for accumulating grad norms from each batch in this epoch
        _gradnorms = torch.empty(len(train_loader))
        if self.use_cuda:
            _gradnorms = _gradnorms.cuda(self.data_wrapper.cuda)
        total_loss = 0
        train_label_gpu = None

        for i, (inputdata, labels) in enumerate(train_loader):
            features, cuda_labels = self._prepare_batch_data(inputdata, labels)
            optimizer.zero_grad()
            aux_out_map, _ = self.model(features)

            if train_predict.size()[0] == 0:
                train_predict = aux_out_map['final'].data
                train_label_gpu = cuda_labels
            else:
                train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)
                train_label_gpu = torch.cat([train_label_gpu, cuda_labels], dim=0)

            total_loss = 0
            for name, output in aux_out_map.items():
                loss = CCCLoss()
                if name == 'final':
                    total_loss += loss(output, cuda_labels)
                else:
                    total_loss += self.data_wrapper.alpha * loss(output, cuda_labels)
            total_loss.backward()

            for name, param in self.model.named_parameters():
                if '_direct_gene_layer.weight' not in name:
                    continue
                term_name = name.split('_')[0]
                param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

            # Save gradnorm for batch
            _gradnorms[i] = util.get_grad_norm(self.model.parameters(), 2.0).unsqueeze(0)
            optimizer.step()

        # Save total gradnorm for epoch
        gradnorms = sum(_gradnorms).unsqueeze(0).cpu().numpy()[0]

        return train_predict, total_loss, gradnorms, train_label_gpu

    def _validate_epoch(self, val_loader):
        """
        Conducts a validation epoch over the validation dataset.

        :param val_loader: Data loader for validation data.
        :type val_loader: DataLoader

        :returns : Tuple containing validation predictions and validation loss.
        :rtype : Tuple[Tensor, float]
        """
        self.model.eval()
        val_predict = torch.zeros(0, 0)
        if self.use_cuda:
            val_predict = val_predict.cuda(self.data_wrapper.cuda)
        val_loss = 0
        val_label_gpu = None

        for inputdata, labels in val_loader:
            features, cuda_labels = self._prepare_batch_data(inputdata, labels)
            aux_out_map, _ = self.model(features)

            if val_predict.size()[0] == 0:
                val_predict = aux_out_map['final'].data
                val_label_gpu = cuda_labels
            else:
                val_predict = torch.cat([val_predict, aux_out_map['final'].data], dim=0)
                val_label_gpu = torch.cat([val_label_gpu, cuda_labels], dim=0)

            for name, output in aux_out_map.items():
                loss = CCCLoss()
                if name == 'final':
                    val_loss += loss(output, cuda_labels)

        return val_predict, val_loss, val_label_gpu

    def _prepare_batch_data(self, inputdata, labels):
        """
        Prepares batch data for training or validation.

        :param inputdata: Input data tensor.
        :type inputdata: Tensor
        :param labels: Labels tensor.
        :type labels: Tensor

        :returns : Tuple containing features and labels as Variables.
        :rtype : Tuple[Variable, Variable]
        """
        features = util.build_input_vector(inputdata, self.data_wrapper.cell_features)
        cuda_features = Variable(self._to_device(features))
        cuda_labels = Variable(self._to_device(labels))
        return cuda_features, cuda_labels

    def _initialize_model_parameters(self, term_mask_map):
        """
        Initializes the model parameters, applying term masks and scaling.

        :param term_mask_map: Mask map for term weights.
        :type term_mask_map: Tensor
        """
        for name, param in self.model.named_parameters():
            term_name = name.split('_')[0]
            if '_direct_gene_layer.weight' in name:
                param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
            else:
                param.data = param.data * 0.1

    def _configure_optimizer(self):
        """
        Configures the optimizer for the model.

        :returns : Configured optimizer.
        :rtype : torch.optim.Optimizer
        """
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.data_wrapper.lr,
            betas=(0.9, 0.99),
            eps=1e-05,
            weight_decay=self.data_wrapper.lr
        )

    @staticmethod
    def _calculate_metrics(train_predict, train_label_gpu, val_predict, val_label_gpu):
        """
        Calculates performance metrics for the training and validation predictions.

        :param train_predict: Training predictions.
        :type train_predict: Tensor
        :param train_label_gpu: Training labels.
        :type train_label_gpu: Tensor
        :param val_predict: Validation predictions.
        :type val_predict: Tensor
        :param val_label_gpu: Validation labels.
        :type val_label_gpu: Tensor

        :returns : Tuple containing training correlation, validation correlation,
                    true average under the curve, predicted average under the curve.
        :rtype : Tuple[float, float, float, float]
        """
        train_corr = util.pearson_corr(train_predict, train_label_gpu)
        val_corr = util.pearson_corr(val_predict, val_label_gpu)
        true_auc = torch.mean(train_label_gpu).item()
        pred_auc = torch.mean(train_predict).item()

        return train_corr, val_corr, true_auc, pred_auc

    def _save_model_if_improved(self, min_loss, val_loss, epoch):
        """
        Saves the model if the validation loss has improved.

        :param min_loss: Minimum loss from previous epochs.
        :type min_loss: float
        :param val_loss: Current epoch's validation loss.
        :type val_loss: float
        :param epoch: Current epoch number.
        :type epoch: int

        :returns : Updated minimum loss.
        :rtype : float
        """
        if min_loss is None or val_loss < min_loss - self.data_wrapper.delta:
            min_loss = val_loss
            torch.save(self.model, self.data_wrapper.modelfile)
            logger.info(f"Model saved at epoch {epoch}")

        return min_loss
