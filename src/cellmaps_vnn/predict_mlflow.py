import os
import logging
import shutil
from datetime import date

from cellmaps_vnn.util import copy_and_register_gene2id_file
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.utils.data as du
from torch.autograd import Variable
from cellmaps_utils import constants
import cellmaps_vnn.constants as vnnconstants
from ndex2.cx2 import RawCX2NetworkFactory

import cellmaps_vnn
from cellmaps_vnn import util
from cellmaps_vnn.exceptions import CellmapsvnnError
from cellmaps_vnn.rlipp_calculator import RLIPPCalculator
import wandb


logger = logging.getLogger(__name__)


class VNNPredict:
    COMMAND = 'predict'

    def __init__(self, theargs):
        """
        Constructor for predicting with a trained model.
        """
        if not os.path.exists(os.path.join(theargs.inputdir, 'model_final.pt')):
            theargs.inputdir = os.path.join(theargs.inputdir, 'out_train')
        self._theargs = theargs
        self._outdir = os.path.abspath(theargs.outdir)
        self._number_feature_grads = 0
        self.use_cuda = torch.cuda.is_available() and self._theargs.cuda is not None

        self.excluded_terms = []
        excluded_terms_path = os.path.join(theargs.inputdir, 'vnn_excluded_terms.txt')
        if os.path.exists(excluded_terms_path):
            with open(excluded_terms_path, 'r') as file:
                self.excluded_terms = set(int(line.strip()) for line in file if line.strip().isdigit())

    @staticmethod
    def add_subparser(subparsers):
        """
        Adds a subparser for the 'predict' command.
        """
        # TODO: modify description later
        desc = """
        Version: todo

        The 'predict' command takes a trained model and input data to run predictions.
        The results are stored in a specified output directory.
        """
        parser = subparsers.add_parser(VNNPredict.COMMAND,
                                       help='Run prediction using a trained model',
                                       description=desc,
                                       formatter_class=constants.ArgParseFormatter)
        parser.add_argument('outdir', help='Directory to write results to')
        parser.add_argument('--inputdir', required=True, help='Path to RO-Crate with the trained model', type=str)
        parser.add_argument('--config_file', help='Config file that can be used to populate arguments for training. '
                                                  'If a given argument is set, it will override the default value.')
        parser.add_argument('--predict_data', help='Path to the dataset to be predicted', type=str)
        parser.add_argument('--gene2id', help='Gene to ID mapping file', type=str)
        parser.add_argument('--cell2id', help='Cell to ID mapping file', type=str)
        parser.add_argument('--mutations', help='Mutation information for cell lines', type=str)
        parser.add_argument('--cn_deletions', help='Copy number deletions for cell lines', type=str)
        parser.add_argument('--cn_amplifications', help='Copy number amplifications for cell lines',
                            type=str)
        parser.add_argument('--batchsize', help='Batchsize', type=int, default=1000)
        parser.add_argument('--zscore_method', help='zscore method (zscore/robustz)', type=str, default='auc')
        parser.add_argument('--cpu_count', help='No of available cores', type=int, default=1)
        parser.add_argument('--drug_count', help='No of top performing drugs', type=int, default=0)
        parser.add_argument('--genotype_hiddens',
                            help='Mapping for the number of neurons in each term in genotype parts', type=int,
                            default=4)
        parser.add_argument('--cuda', help='Specify GPU', type=int, default=0)
        parser.add_argument('--std', help='Standardization File (if not set standardization file from RO-Crate '
                                          'will be used)', type=str)
        parser.add_argument('--slurm', help='If set, slurm script for training will be generated.',
                            action='store_true')
        parser.add_argument('--use_gpu', help='If set, slurm script will be adjusted to run on GPU.',
                            action='store_true')
        parser.add_argument('--slurm_partition', help='Slurm partition. If use_gpu is set, the default is nrnb-gpu.',
                            type=str)
        parser.add_argument('--slurm_account', help='Slurm account. If use_gpu is set, the default is nrnb-gpu.',
                            type=str)
        return parser

    def run(self):
        """
        The logic for running predictions with the model. It executes the prediction process
        using the trained model and input data.

        :raises CellmapsvnnError: If an error occurs during the prediction process.
        """
        try:
            model = os.path.join(self._theargs.inputdir, 'model_final.pt')
            std = os.path.join(self._theargs.inputdir, 'std.txt') if self._theargs.std is None else self._theargs.std
            torch.set_printoptions(precision=5)

            # Load data and model for prediction
            predict_data, cell2id_mapping = self._prepare_predict_data(
                self._theargs.predict_data, self._theargs.cell2id, self._theargs.zscore_method, std)

            # Load cell features
            cell_features = util.load_cell_features(self._theargs.mutations, self._theargs.cn_deletions,
                                                    self._theargs.cn_amplifications)

            hidden_dir = self._get_hidden_dir_path()
            if not os.path.exists(hidden_dir):
                os.mkdir(hidden_dir)

            # Perform prediction
            self.predict(predict_data, model, hidden_dir, self._theargs.batchsize,
                         cell_features)

            hierarchy_file = os.path.join(self._theargs.inputdir, vnnconstants.HIERARCHY_FILENAME)
            factory = RawCX2NetworkFactory()
            hierarchy = factory.get_cx2network(hierarchy_file)
            rlipp_file = os.path.join(self._outdir, vnnconstants.RLIPP_OUTPUT_FILE)
            gene_rho_file = os.path.join(self._outdir, 'gene_rho.out')
            # Perform interpretation
            calc = RLIPPCalculator(hierarchy, self._theargs.predict_data, self._get_predict_dest_file(),
                                   self._theargs.gene2id, self._theargs.cell2id, hidden_dir,
                                   rlipp_file, gene_rho_file, self._theargs.cpu_count, self._theargs.genotype_hiddens,
                                   self._theargs.drug_count, self.excluded_terms)
            calc.calc_scores()
            logger.info('Prediction and interpretation executed successfully')
            print('Prediction and interpretation executed successfully')
        except Exception as e:
            logger.error(f"Error in prediction flow: {e}")
            raise CellmapsvnnError(f"Encountered problem in prediction flow: {e}")

    def _prepare_predict_data(self, test_file, cell2id_mapping_file, zscore_method, std_file):
        """
        Prepares the prediction data for the model.

        :param test_file: Path to the file containing the test dataset.
        :type test_file: str
        :param cell2id_mapping_file: Path to the file containing the cell to ID mapping.
        :type cell2id_mapping_file: str
        :param zscore_method: Method used for z-score standardization.
        :type zscore_method: str
        :param std_file: Path to the standardization file.
        :type std_file: str

        :return: A tuple containing test features and labels as tensors, and the cell2id mapping.
        :rtype: Tuple(Tensor, Tensor), dict
        """
        cell2id_mapping = util.load_mapping(cell2id_mapping_file, 'cell lines')
        test_features, test_labels = self._load_pred_data(test_file, cell2id_mapping, zscore_method, std_file)
        return (torch.Tensor(test_features), torch.Tensor(test_labels)), cell2id_mapping

    @staticmethod
    def _load_pred_data(test_file, cell2id, zscore_method, train_std_file):
        """
        Loads and processes prediction data from a file.

        :param test_file: Path to the file containing the test dataset.
        :type test_file: str
        :param cell2id: Dictionary mapping cell lines to their respective IDs.
        :type cell2id: dict
        :param zscore_method: Method used for z-score standardization.
        :type zscore_method: str
        :param train_std_file: Path to the training standardization file.
        :type train_std_file: str

        :return: Features and labels for the prediction data.
        :rtype: List, List
        """
        train_std_df = pd.read_csv(train_std_file, sep='\t', header=None, names=['dataset', 'center', 'scale'])
        all_df = pd.read_csv(test_file, sep='\t', header=None, names=['cell_line', 'smiles', 'auc', 'dataset'])
        test_df = all_df[all_df['cell_line'].isin(cell2id.keys())]
        test_std_df = util.calc_std_vals(test_df, zscore_method)
        for i, row in test_std_df.iterrows():
            dataset = row['dataset']
            train_entry = train_std_df.query('dataset == @dataset')
            if not train_entry.empty:
                test_std_df.loc[i, 'center'] = float(train_entry['center'].iloc[0])
                test_std_df.loc[i, 'scale'] = float(train_entry['scale'].iloc[0])
        test_df = util.standardize_data(test_df, test_std_df)

        feature = []
        label = []
        for row in test_df.values:
            feature.append([cell2id[row[0]]])
            label.append([float(row[2])])
        return feature, label

    def _get_predict_dest_file(self):
        """
        Returns the file path for saving the prediction results.

        :return: The file path to the prediction results file.
        """
        return os.path.join(self._outdir, 'predict.txt')

    def _get_feature_grad_dest_file(self, grad):
        """
        Returns the file path for saving the gradient of a specific feature.

        :return: The file path to the prediction feature grad file.
        """
        return os.path.join(self._outdir, f'predict_feature_grad_{grad}.txt')

    def _get_hidden_dir_path(self):
        """
        Returns the path to the directory where hidden layer outputs will be stored.

        :return: The file path to the hidden directory.
        """
        return os.path.join(self._outdir, 'hidden/')

    def _to_device(self, tensor):
        if self.use_cuda:
            return tensor.cuda(self._theargs.cuda)
        return tensor

    def predict(self, predict_data, model_file, hidden_folder, batch_size, cell_features=None):
        """
        Perform prediction using the trained model.

        :param predict_data: Tuple of features and labels for prediction.
        :param model_file: Path to the trained model file.
        :param hidden_folder: Directory to store hidden layer outputs.
        :param batch_size: Size of each batch for prediction.
        :param cell_features: Additional cell features for prediction.
        """
        try:
            logger.info('Starting prediction process')
            print('Starting prediction process')
            model = self._load_model(model_file)
            test_loader = self._create_data_loader(predict_data, batch_size)
            test_predict, saved_grads = self._predict(model, test_loader, cell_features, hidden_folder)

            predict_label_gpu = self._to_device(predict_data[1])
            test_corr = util.pearson_corr(test_predict, predict_label_gpu)
            logger.info(f"Test correlation {model.root}: {test_corr:.4f}")
            wandb.run.summary["test_corr"] = test_corr

            np.savetxt(self._get_predict_dest_file(), test_predict.cpu().numpy(), '%.4e')

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise CellmapsvnnError(f"Encountered problem in prediction: {e}")

    def _load_model(self, model_file):
        """
        Load the trained model for prediction.

        :param model_file: Path to the trained model file.
        :return: Loaded model.
        """
        model = torch.load(model_file,
                           map_location=f'cuda:{self._theargs.cuda}' if self.use_cuda else torch.device("cpu"))
        if self.use_cuda:
            model.cuda(self._theargs.cuda)
        model.eval()
        return model

    def _create_data_loader(self, predict_data, batch_size):
        """
        Create a DataLoader for the prediction data.

        :param predict_data: Tuple of features and labels for prediction.
        :param batch_size: Size of each batch for prediction.
        :return: DataLoader for the prediction data.
        """
        predict_feature, predict_label = predict_data
        return du.DataLoader(du.TensorDataset(predict_feature, predict_label), batch_size=batch_size, shuffle=False)

    def _predict(self, model, data_loader, cell_features, hidden_folder):
        """
        Run the prediction process and save outputs.

        :param model: Trained model for prediction.
        :param data_loader: DataLoader containing the prediction data.
        :param cell_features: Additional cell features for prediction.
        :param hidden_folder: Directory to store hidden layer outputs.
        :return: Tuple of prediction results and saved gradients.
        """
        test_predict = torch.zeros(0, 0)
        if self.use_cuda:
            test_predict = test_predict.cuda(self._theargs.cuda)
        saved_grads = {}

        for i, (inputdata, labels) in enumerate(data_loader):
            cuda_features = self._process_input(inputdata, cell_features)
            aux_out_map, hidden_embeddings_map = model(cuda_features)
            test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0) \
                if test_predict.size()[0] else aux_out_map['final'].data

            self._save_hidden_outputs(hidden_embeddings_map, hidden_folder)
            self._register_gradient_hooks(hidden_embeddings_map, saved_grads)
            self._backpropagate(aux_out_map)

            self._save_gradients(cuda_features)
            self._save_hidden_gradients(saved_grads, hidden_folder)

        return test_predict, saved_grads

    def _process_input(self, inputdata, cell_features):
        """
        Process input data for the model.

        :param inputdata: Input data for the model.
        :param cell_features: Additional cell features for prediction.
        :return: Processed features as CUDA variables.
        """
        features = util.build_input_vector(inputdata, cell_features)
        return Variable(self._to_device(features), requires_grad=True)

    def _save_hidden_outputs(self, hidden_embeddings_map, hidden_folder):
        """
        Save outputs from hidden layers.

        :param hidden_embeddings_map: Dictionary of hidden layer outputs.
        :param hidden_folder: Directory to save hidden layer outputs.
        """
        for element, hidden_map in hidden_embeddings_map.items():
            hidden_file = os.path.join(hidden_folder, element + '.hidden')
            with open(hidden_file, 'ab') as f:
                np.savetxt(f, hidden_map.data.cpu().numpy(), '%.4e')

    def _register_gradient_hooks(self, hidden_embeddings_map, saved_grads):
        """
        Register gradient hooks to save gradients of hidden layers.

        :param hidden_embeddings_map: Dictionary of hidden layer outputs.
        :param saved_grads: Dictionary to store saved gradients.
        """

        def save_grad(elem):
            def savegrad_hook(grad):
                saved_grads[elem] = grad

            return savegrad_hook

        for element, _ in hidden_embeddings_map.items():
            hidden_embeddings_map[element].register_hook(save_grad(element))

    def _backpropagate(self, aux_out_map):
        """
        Perform backpropagation.

        :param aux_out_map: Auxiliary output map from the model.
        """
        aux_out_map['final'].backward(torch.ones_like(aux_out_map['final']))

    def _save_gradients(self, cuda_features):
        """
        Save gradients for each feature.

        :param cuda_features: CUDA features variable.
        """
        self._number_feature_grads = len(cuda_features[0, 0, :])
        for i in range(self._number_feature_grads):
            feature_grad = cuda_features.grad.data[:, :, i]
            grad_file = self._get_feature_grad_dest_file(i)
            with open(grad_file, 'ab') as f:
                np.savetxt(f, feature_grad.cpu().numpy(), '%.4e', delimiter='\t')

    def _save_hidden_gradients(self, saved_grads, hidden_folder):
        """
        Save the gradients of the hidden layer outputs.

        :param saved_grads: Dictionary containing the saved gradients.
        :param hidden_folder: Directory to save the hidden layer gradients.
        """
        for element, hidden_grad in saved_grads.items():
            hidden_grad_file = os.path.join(hidden_folder, f'{element}.hidden_grad')
            with open(hidden_grad_file, 'ab') as f:
                np.savetxt(f, hidden_grad.data.cpu().numpy(), '%.4e', delimiter='\t')

    def register_outputs(self, outdir, description, keywords, provenance_utils):
        """
        Registers all output files (predictions, feature gradients, and hidden files)
        with the FAIRSCAPE service for data provenance.

        :param outdir: The directory where the output files are stored.
        :param description: Description for the output files.
        :param keywords: List of keywords associated with the files.
        :param provenance_utils: The utility class for provenance registration.

        :return: A list of dataset IDs for the registered files.
        """
        output_ids = [copy_and_register_gene2id_file(self._theargs.gene2id, outdir, description, keywords,
                                                     provenance_utils),
                      self._register_predict_file(outdir, description, keywords, provenance_utils)]
        for i in range(self._number_feature_grads):
            output_ids.append(self._register_feature_grad_file(outdir, description, keywords, provenance_utils, i))
        output_ids.extend(self._register_hidden_files(outdir, description, keywords, provenance_utils))
        orginal_hierarchy_id = self._copy_and_register_original_hierarchy(outdir, description, keywords, provenance_utils)
        if orginal_hierarchy_id is not None:
            output_ids.append(orginal_hierarchy_id)
        output_ids.append(self._copy_and_register_hierarchy(outdir, description, keywords, provenance_utils))
        id_parent = self._copy_and_register_hierarchy_parent(outdir, description, keywords, provenance_utils)
        if id_parent is not None:
            output_ids.append(id_parent)
        return output_ids

    def _register_predict_file(self, outdir, description, keywords, provenance_utils):
        """
        Registers the prediction result file with the FAIRSCAPE service for data provenance.

        :param outdir: The output directory where the outputs are stored.
        :param description: Description of the file for provenance registration.
        :param keywords: List of keywords associated with the file.
        :param provenance_utils: The utility class for provenance registration.

        :return: The dataset ID assigned to the registered file.
        """
        dest_path = self._get_predict_dest_file()
        description = description
        description += ' prediction result file'
        keywords = keywords
        keywords.extend(['file'])
        data_dict = {'name': os.path.basename(dest_path) + ' prediction result file',
                     'description': description,
                     'keywords': keywords,
                     'data-format': 'txt',
                     'author': cellmaps_vnn.__name__,
                     'version': cellmaps_vnn.__version__,
                     'date-published': date.today().strftime(provenance_utils.get_default_date_format_str())}
        dataset_id = provenance_utils.register_dataset(outdir,
                                                       source_file=dest_path,
                                                       data_dict=data_dict)
        return dataset_id

    def _register_feature_grad_file(self, outdir, description, keywords, provenance_utils, grad):
        """
        Registers the feature gradient file with the FAIRSCAPE service for data provenance.

        :param outdir: The output directory where the file is stored.
        :param description: Description of the file for provenance registration.
        :param keywords: List of keywords associated with the file.
        :param provenance_utils: The utility class for provenance registration.
        :param grad: The specific gradient index for the feature.

        :return: The dataset ID assigned to the registered file.
        """
        dest_path = self._get_feature_grad_dest_file(grad)
        description = description
        description += f' prediction feature grad {grad} file'
        keywords = keywords
        keywords.extend(['file'])
        data_dict = {'name': os.path.basename(dest_path) + f' prediction feature grad {grad} file',
                     'description': description,
                     'keywords': keywords,
                     'data-format': 'txt',
                     'author': cellmaps_vnn.__name__,
                     'version': cellmaps_vnn.__version__,
                     'date-published': date.today().strftime(provenance_utils.get_default_date_format_str())}
        dataset_id = provenance_utils.register_dataset(outdir,
                                                       source_file=dest_path,
                                                       data_dict=data_dict)
        return dataset_id

    def _register_hidden_files(self, outdir, description, keywords, provenance_utils):
        """
        Registers the output files from the hidden layers with the FAIRSCAPE service for data provenance.

        :param outdir: The directory where the hidden layer output files are stored.
        :param description: A general description for the hidden files.
        :param keywords: A list of keywords associated with the hidden files.
        :param provenance_utils: An instance of the utility class used for handling the provenance registration.

        :return: A list of dataset IDs, each corresponding to a registered hidden file.
        """
        data_dict = {'name': cellmaps_vnn.__name__ + ' hidden layer output file',
                     'description': description + ' hidden layer output file',
                     'author': cellmaps_vnn.__name__,
                     'version': cellmaps_vnn.__version__,
                     'date-published': date.today().strftime(provenance_utils.get_default_date_format_str())}

        hidden_files_ids = list()

        hidden_dir = self._get_hidden_dir_path()
        cntr = 0
        for entry in tqdm(os.listdir(hidden_dir), desc='FAIRSCAPE hidden files registration'):
            data_dict['data-format'] = entry.split('.')[-1]
            dest_path = os.path.join(hidden_dir, entry)
            data_dict['name'] = os.path.basename(dest_path) + f' hidden file'
            data_dict['keywords'] = ['hidden', 'file']
            dataset_id = provenance_utils.register_dataset(outdir,
                                                           source_file=dest_path,
                                                           data_dict=data_dict)
            hidden_files_ids.append(dataset_id)
            cntr += 1
            if cntr > 5:
                # Todo: https://github.com/fairscape/fairscape-cli/issues/9
                logger.warning('FAIRSCAPE cannot handle too many files, skipping rest')
                break
        return hidden_files_ids

    def _copy_and_register_original_hierarchy(self, outdir, description, keywords, provenance_utils):
        hierarchy_out_file = os.path.join(outdir, vnnconstants.ORIGINAL_HIERARCHY_FILENAME)
        hierarchy_in_file = os.path.join(self._theargs.inputdir, vnnconstants.ORIGINAL_HIERARCHY_FILENAME)
        if not os.path.exists(hierarchy_in_file):
            return None
        shutil.copy(hierarchy_in_file, hierarchy_out_file)

        data_dict = {'name': os.path.basename(hierarchy_out_file) + ' Hierarchy network file',
                     'description': description + ' Hierarchy network file',
                     'keywords': keywords,
                     'data-format': 'CX2',
                     'author': cellmaps_vnn.__name__,
                     'version': cellmaps_vnn.__version__,
                     'date-published': date.today().strftime('%m-%d-%Y')}
        dataset_id = provenance_utils.register_dataset(outdir,
                                                       source_file=hierarchy_out_file,
                                                       data_dict=data_dict)
        return dataset_id

    def _copy_and_register_hierarchy(self, outdir, description, keywords, provenance_utils):
        hierarchy_out_file = os.path.join(outdir, vnnconstants.HIERARCHY_FILENAME)
        shutil.copy(os.path.join(self._theargs.inputdir, vnnconstants.HIERARCHY_FILENAME), hierarchy_out_file)

        data_dict = {'name': os.path.basename(hierarchy_out_file) + ' Hierarchy network file used to build VNN',
                     'description': description + ' Hierarchy network file used to build VNN',
                     'keywords': keywords,
                     'data-format': 'CX2',
                     'author': cellmaps_vnn.__name__,
                     'version': cellmaps_vnn.__version__,
                     'date-published': date.today().strftime('%m-%d-%Y')}
        dataset_id = provenance_utils.register_dataset(outdir,
                                                       source_file=hierarchy_out_file,
                                                       data_dict=data_dict)
        return dataset_id

    def _copy_and_register_hierarchy_parent(self, outdir, description, keywords, provenance_utils):
        hierarchy_parent_in_file = os.path.join(self._theargs.inputdir, vnnconstants.PARENT_NETWORK_NAME)
        if not os.path.exists(hierarchy_parent_in_file):
            return None
        hierarchy_parent_out_file = os.path.join(outdir, vnnconstants.PARENT_NETWORK_NAME)
        shutil.copy(hierarchy_parent_in_file, hierarchy_parent_out_file)

        data_dict = {'name': os.path.basename(hierarchy_parent_out_file) + ' Hierarchy parent network file',
                     'description': description + ' Hierarchy parent network file',
                     'keywords': keywords,
                     'data-format': 'CX2',
                     'author': cellmaps_vnn.__name__,
                     'version': cellmaps_vnn.__version__,
                     'date-published': date.today().strftime('%m-%d-%Y')}
        dataset_id = provenance_utils.register_dataset(outdir,
                                                       source_file=hierarchy_parent_out_file,
                                                       data_dict=data_dict)
        return dataset_id

    def _register_rlipp_file(self, outdir, description, keywords, provenance_utils):
        rlipp_file = os.path.join(outdir, vnnconstants.RLIPP_OUTPUT_FILE)

        data_dict = {'name': os.path.basename(rlipp_file) + ' RLIPP output file',
                     'description': description + ' RLIPP output file',
                     'keywords': keywords,
                     'data-format': 'txt',
                     'author': cellmaps_vnn.__name__,
                     'version': cellmaps_vnn.__version__,
                     'date-published': date.today().strftime('%m-%d-%Y')}
        dataset_id = provenance_utils.register_dataset(outdir,
                                                       source_file=rlipp_file,
                                                       data_dict=data_dict)
        return dataset_id

    def _register_gene_rho_file(self, outdir, description, keywords, provenance_utils):
        gene_rho_file = os.path.join(outdir, 'gene_rho.out')

        data_dict = {'name': os.path.basename(gene_rho_file) + ' Gene Rho file',
                     'description': description + ' Gene Rho file',
                     'keywords': keywords,
                     'data-format': 'txt',
                     'author': cellmaps_vnn.__name__,
                     'version': cellmaps_vnn.__version__,
                     'date-published': date.today().strftime('%m-%d-%Y')}
        dataset_id = provenance_utils.register_dataset(outdir,
                                                       source_file=gene_rho_file,
                                                       data_dict=data_dict)
        return dataset_id
