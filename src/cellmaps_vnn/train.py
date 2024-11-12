# train.py
import os
import shutil
from datetime import date

from cellmaps_utils import constants as constants
import cellmaps_vnn.constants as vnnconstants
import logging

import cellmaps_vnn
from cellmaps_vnn.data_wrapper import TrainingDataWrapper
from cellmaps_vnn.exceptions import CellmapsvnnError
from cellmaps_vnn.vnn_trainer import VNNTrainer
from cellmaps_vnn.util import copy_and_register_gene2id_file

logger = logging.getLogger(__name__)


class VNNTrain:
    COMMAND = 'train'

    def __init__(self, theargs):
        """
        Constructor for training Visual Neural Network.
        """
        self._theargs = theargs
        self._outdir = os.path.abspath(theargs.outdir)
        self._theargs.modelfile = self._get_model_dest_file()
        self._theargs.stdfile = self._get_std_dest_file()

    @staticmethod
    def add_subparser(subparsers):
        """
        Adds a subparser for the 'train' command.
        """
        # TODO: modify description later
        desc = """
        Version: todo

        The 'train' command trains a Visual Neural Network using specified hierarchy files
        and data from drugcell or NeSTVNN repository. The resulting model is stored in a
        directory specified by the user.
        """
        parser = subparsers.add_parser(VNNTrain.COMMAND,
                                       help='Train a Visual Neural Network',
                                       description=desc,
                                       formatter_class=constants.ArgParseFormatter)
        parser.add_argument('outdir', help='Directory to write results to')
        parser.add_argument('--inputdir', required=True, help='Path to directory or RO-Crate with hierarchy.cx2 file.'
                                                              'Note that the name of the hierarchy should be '
                                                              'hierarchy.cx2.')
        parser.add_argument('--gene_attribute_name', help='Name of the node attribute of the hierarchy '
                                                          'with list of genes/ proteins of this node. '
                                                          'Default: CD_MemberList.', type=str,
                            default=vnnconstants.GENE_SET_COLUMN_NAME)
        parser.add_argument('--config_file', help='Config file that can be used to populate arguments for training. '
                                                  'If a given argument is set, it will override the default value.')
        parser.add_argument('--training_data', help='Training data')
        parser.add_argument('--gene2id', help='Gene to ID mapping file', type=str)
        parser.add_argument('--cell2id', help='Cell to ID mapping file', type=str)
        parser.add_argument('--mutations', help='Mutation information for cell lines', type=str)
        parser.add_argument('--cn_deletions', help='Copy number deletions for cell lines', type=str)
        parser.add_argument('--cn_amplifications', help='Copy number amplifications for cell lines',
                            type=str)
        parser.add_argument('--batchsize', help='Batchsize', type=int, default=64)
        parser.add_argument('--zscore_method', help='zscore method (zscore/robustz)', type=str, default='auc')
        parser.add_argument('--epoch', help='Training epochs for training', type=int, default=300)
        parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
        parser.add_argument('--wd', help='Weight decay', type=float, default=0.001)
        parser.add_argument('--alpha', help='Loss parameter alpha', type=float, default=0.3)
        parser.add_argument('--genotype_hiddens',
                            help='Mapping for the number of neurons in each term in genotype parts',
                            type=int, default=4)
        parser.add_argument('--patience', help='Early stopping epoch limit', type=int, default=30)
        parser.add_argument('--delta', help='Minimum change in loss to be considered an improvement', type=float,
                            default=0.001)
        parser.add_argument('--min_dropout_layer', help='Start dropout from this Layer number', type=int, default=2)
        parser.add_argument('--dropout_fraction', help='Dropout Fraction', type=float, default=0.3)
        parser.add_argument('--optimize', help='Hyper-parameter optimization', type=int, default=1)
        parser.add_argument('--cuda', help='Specify GPU', type=int, default=0)
        parser.add_argument('--std', help='Standardization File name', type=str, default='std.txt')
        parser.add_argument('--skip_parent_copy', help='If set, hierarchy parent (interactome) will not be copied',
                            action='store_true')
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
        The logic for training the Visual Neural Network.
        """
        try:
            data_wrapper = TrainingDataWrapper(self._theargs)
            if self._theargs.optimize == 1:
                VNNTrainer(data_wrapper).train_model()
            else:
                logger.error(f"The value {self._theargs.optimize} is wrong value for optimize.")
                raise CellmapsvnnError(f"The value {self._theargs.optimize} is wrong value for optimize.")
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise CellmapsvnnError(f"Encountered problem in training: {e}")

    def _get_model_dest_file(self):
        """
        Returns the file path for saving the trained model file.

        :return: The file path for the model file.
        """
        return os.path.join(self._outdir, 'model_final.pt')

    def _get_std_dest_file(self):
        """
        Returns the file path for saving the standard deviation file.

        :return: The file path for the standard deviation file.
        """
        return os.path.join(self._outdir, self._theargs.std)

    def register_outputs(self, outdir, description, keywords, provenance_utils):
        """
        Registers the model and standard deviation files with the FAIRSCAPE service for data provenance.
        It generates dataset IDs for each registered file.

        :param outdir: The directory where the output files are stored.
        :param description: Description for the output files.
        :param keywords: List of keywords associated with the files.
        :param provenance_utils: The utility class for provenance registration.

        :return: A list of dataset IDs for the registered model and standard deviation files.
        """
        return_ids = [self._register_model_file(outdir, description, keywords, provenance_utils),
                      self._register_std_file(outdir, description, keywords, provenance_utils),
                      self._copy_and_register_hierarchy(outdir, description, keywords, provenance_utils),
                      self._register_pruned_hierarchy(outdir, description, keywords, provenance_utils),
                      copy_and_register_gene2id_file(self._theargs.gene2id, outdir, description, keywords,
                                                     provenance_utils)]
        if not self._theargs.skip_parent_copy:
            id_hierarchy_parent = self._copy_and_register_hierarchy_parent(outdir, description, keywords,
                                                                           provenance_utils)
            if id_hierarchy_parent is not None:
                return_ids.append(id_hierarchy_parent)

        return return_ids

    def _register_model_file(self, outdir, description, keywords, provenance_utils):
        """
        Registers the trained model file with the FAIRSCAPE service for data provenance.

        :param outdir: The output directory where the model file is stored.
        :param description: Description of the model file for provenance registration.
        :param keywords: List of keywords associated with the model file.
        :param provenance_utils: The utility class for provenance registration.

        :return: The dataset ID assigned to the registered model file.
        """
        dest_path = self._get_model_dest_file()
        description = description
        description += ' Model file'
        keywords = keywords
        keywords.extend(['file'])
        data_dict = {'name': os.path.basename(dest_path) + ' trained model file',
                     'description': description,
                     'keywords': keywords,
                     'data-format': 'pt',
                     'author': cellmaps_vnn.__name__,
                     'version': cellmaps_vnn.__version__,
                     'date-published': date.today().strftime(provenance_utils.get_default_date_format_str())}
        dataset_id = provenance_utils.register_dataset(outdir,
                                                       source_file=dest_path,
                                                       data_dict=data_dict)
        return dataset_id

    def _register_std_file(self, outdir, description, keywords, provenance_utils):
        """
        Registers the standard deviation file with the FAIRSCAPE service for data provenance.

        :param outdir: The output directory where the standard deviation file is stored.
        :param description: Description of the standard deviation file for provenance registration.
        :param keywords: List of keywords associated with the standard deviation file.
        :param provenance_utils: The utility class for provenance registration.

        :return: The dataset ID assigned to the registered standard deviation file.
        """
        dest_path = self._get_std_dest_file()
        description = description
        description += ' standard deviation file'
        keywords = keywords
        keywords.extend(['file'])
        data_dict = {'name': os.path.basename(dest_path) + ' standard deviation file',
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

    def _copy_and_register_hierarchy(self, outdir, description, keywords, provenance_utils):
        hierarchy_out_file = os.path.join(outdir, vnnconstants.ORIGINAL_HIERARCHY_FILENAME)
        shutil.copy(os.path.join(self._theargs.inputdir, vnnconstants.HIERARCHY_FILENAME), hierarchy_out_file)

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

    def _register_pruned_hierarchy(self, outdir, description, keywords, provenance_utils):
        hierarchy_out_file = os.path.join(outdir, vnnconstants.HIERARCHY_FILENAME)

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
            logger.warning("No hierarchy parent in the input directory. Cannot copy.")
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

    @property
    def theargs(self):
        return self._theargs
