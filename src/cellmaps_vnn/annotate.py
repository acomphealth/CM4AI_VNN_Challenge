import logging
import os
import shutil
from datetime import date
import getpass
import numpy as np
import pandas as pd
from cellmaps_utils.ndexupload import NDExHierarchyUploader
from cellmaps_utils import constants
from cellmaps_vnn.util import copy_and_register_gene2id_file

import cellmaps_vnn
import cellmaps_vnn.constants as vnnconstants
from ndex2.cx2 import RawCX2NetworkFactory

from cellmaps_vnn.exceptions import CellmapsvnnError

logger = logging.getLogger(__name__)


class VNNAnnotate:
    COMMAND = 'annotate'

    def __init__(self, theargs):
        """
        Constructor. Sets up the hierarchy path either directly from the arguments or by looking for
        a hierarchy.cx2 file in the first RO-Crate directory provided. If neither is found, raises an error.

        :param theargs: The arguments provided to the command line interface.
        :type theargs: argparse.Namespace
        :raises CellmapsvnnError: If no hierarchy path is specified or found.
        """
        self._theargs = theargs
        self._outdir = os.path.abspath(theargs.outdir)
        self.original_hierarchy = None
        if not os.path.exists(os.path.join(theargs.model_predictions[0], vnnconstants.RLIPP_OUTPUT_FILE)):
            theargs.model_predictions[0] = os.path.join(theargs.model_predictions[0], 'out_predict')
        if theargs.hierarchy is not None:
            self.hierarchy = theargs.hierarchy
        else:
            hierarchy_path = os.path.join(theargs.model_predictions[0], vnnconstants.HIERARCHY_FILENAME)
            if os.path.exists(hierarchy_path):
                self.hierarchy = hierarchy_path
            else:
                raise CellmapsvnnError("No hierarchy was specified or found in first ro-crate")
            original_hierarchy_path = os.path.join(theargs.model_predictions[0],
                                                   vnnconstants.ORIGINAL_HIERARCHY_FILENAME)
            if os.path.exists(original_hierarchy_path):
                self.original_hierarchy = original_hierarchy_path
        if theargs.parent_network is not None:
            self.parent_network = theargs.parent_network
        else:
            parent_network_path = os.path.join(theargs.model_predictions[0], vnnconstants.PARENT_NETWORK_NAME)
            if os.path.exists(parent_network_path):
                self.parent_network = parent_network_path
            else:
                self.parent_network = None
        if self.parent_network is not None and os.path.isfile(self.parent_network):
            self.parent_network = os.path.abspath(self.parent_network)

    @staticmethod
    def add_subparser(subparsers):
        """
        Adds a subparser for the 'annotate' command.
        """
        # TODO: modify description later
        desc = """
        Version: todo

        The 'annotate' command takes ..
        """
        parser = subparsers.add_parser(VNNAnnotate.COMMAND,
                                       help='Run prediction using a trained model',
                                       description=desc,
                                       formatter_class=constants.ArgParseFormatter)
        parser.add_argument('outdir', help='Directory to write results to')
        parser.add_argument('--model_predictions', nargs='+', required=True,
                            help='Path to one or multiple RO-Crate with the predictions and interpretations '
                                 'obtained from predict step',
                            type=str)
        parser.add_argument('--disease', help='Specify the disease or cancer type for which the annotations will be '
                                              'performed. This allows the annotation process to tailor the results '
                                              'according to the particular disease or cancer type. If not set, '
                                              'prediction scores for all diseases will be aggregated.', type=str)
        parser.add_argument('--hierarchy', help='Path to hierarchy (optional), if not set the hierarchy will be '
                                                'selected from the first RO-Crate passed in --model_predictions '
                                                'argument', type=str)
        parser.add_argument('--parent_network', help='Path to interactome (parent network) of the annotated hierarchy '
                                                     'or NDEx UUID of parent network (required if uploading '
                                                     'HCX to NDEx)', type=str)
        parser.add_argument('--ndexserver', default='ndexbio.org',
                            help='Server where annotated hierarchy will be uploaded to')
        parser.add_argument('--ndexuser',
                            help='NDEx user account. Required if uploading to NDEx.')
        parser.add_argument('--ndexpassword',
                            help='NDEx password. Enter "-" to input password interactively, or provide a file '
                                 'containing the password. Required if uploading to NDEx.')
        parser.add_argument('--visibility', action='store_true',
                            help='If set, makes Hierarchy and interactome network loaded onto '
                                 'NDEx publicly visible')
        parser.add_argument('--slurm', help='If set, slurm script for training will be generated.',
                            action='store_true')
        parser.add_argument('--slurm_partition', help='Slurm partition', type=str)
        parser.add_argument('--slurm_account', help='Slurm account', type=str)

    def _get_rlipp_out_dest_file(self):
        """
        Constructs the file path for the RLIPP output file within the specified output directory.

        :return: The file path for the RLIPP output file.
        :rtype: str
        """
        return os.path.join(self._outdir, vnnconstants.RLIPP_OUTPUT_FILE)

    def _get_hierarchy_dest_file(self):
        """
        Constructs the file path for the hierarchy output file within the specified output directory.

        :return: The file path for the hierarchy output file.
        :rtype: str
        """
        return os.path.join(self._outdir, vnnconstants.HIERARCHY_FILENAME)

    def _get_original_hierarchy_dest_file(self):
        """
        Constructs the file path for the hierarchy output file within the specified output directory.

        :return: The file path for the hierarchy output file.
        :rtype: str
        """
        return os.path.join(self._outdir, vnnconstants.ORIGINAL_HIERARCHY_FILENAME)

    def _aggregate_prediction_scores_from_models(self):
        """
        Aggregates prediction scores from multiple models' outputs by averaging them.
        The aggregated scores are then saved to the RLIPP output destination file.
        """
        data = {}

        for directory in self._theargs.model_predictions:
            if not os.path.exists(os.path.join(directory, vnnconstants.RLIPP_OUTPUT_FILE)):
                directory = os.path.join(directory, 'out_predict')
            filepath = os.path.join(directory, vnnconstants.RLIPP_OUTPUT_FILE)
            has_disease = False
            with open(filepath, 'r') as file:
                for line in file:
                    if line.startswith('Term') or not line.strip():
                        if 'Disease' in line:
                            has_disease = True
                        continue

                    parts = line.strip().split('\t')
                    if has_disease:
                        key = (parts[0], parts[-1])  # (Term, Disease)
                        values = np.array([float(v) for v in parts[1:-1]])
                    else:
                        key = (parts[0], 'unspecified')
                        values = np.array([float(v) for v in parts[1:]])

                    if key not in data:
                        data[key] = []
                    data[key].append(values)

        averaged_data = {k: np.mean(v, axis=0) for k, v in data.items()}

        with open(self._get_rlipp_out_dest_file(), 'w') as outfile:
            outfile.write("Term\tP_rho\tP_pval\tC_rho\tC_pval\tRLIPP\tDisease\n")
            for (term, disease), values in averaged_data.items():
                outfile.write(f"{term}\t" + "\t".join([f"{v:.5e}" for v in values]) + f"\t{disease}\n")

    @staticmethod
    def _aggregate_scores_from_diseases(data):
        """
        Aggregates the prediction scores for all diseases by averaging P_rho score.

        :return: A dictionary mapping each term to its averaged P_rho score across all diseases.
        :rtype: dict
        """
        aggregated_data = data.groupby('Term').agg({
            vnnconstants.PRHO_SCORE: 'mean',
            vnnconstants.P_PVAL_SCORE: 'mean',
            vnnconstants.CRHO_SCORE: 'mean',
            vnnconstants.C_PVAL_SCORE: 'mean',
            vnnconstants.RLIPP_SCORE: 'mean'
        })

        aggregated_dict = {
            term: [row[vnnconstants.PRHO_SCORE], row[vnnconstants.P_PVAL_SCORE],
                   row[vnnconstants.CRHO_SCORE], row[vnnconstants.C_PVAL_SCORE], row[vnnconstants.RLIPP_SCORE]]
            for term, row in aggregated_data.iterrows()
        }

        return aggregated_dict

    @staticmethod
    def _get_scores_for_disease(disease, data):
        """
        Retrieves prediction scores for a specific disease, returning a dictionary mapping
        each term to its P_rho score for the given disease.

        :param disease: The disease or cancer type for which scores are requested.
        :type disease: str
        :return: A dictionary with Term as keys and P_rho scores as values for the specified disease.
        :rtype: dict
        """
        filtered_data = data[data['Disease'] == disease]
        if filtered_data.empty:
            return {}

        scores = {
            term: [row[vnnconstants.PRHO_SCORE], row[vnnconstants.P_PVAL_SCORE],
                   row[vnnconstants.CRHO_SCORE], row[vnnconstants.C_PVAL_SCORE], row[vnnconstants.RLIPP_SCORE]]
            for term, row in filtered_data.set_index('Term').iterrows()
        }

        return scores

    def _upload_to_ndex_if_credentials_provided(self):
        """
        Uploads hierarchy and parent network to NDEx if credentials are provided.

        This method checks if the NDEx server, user, and password credentials are provided.
        If they are, it uploads the hierarchy and parent network to NDEx. If the parent
        network is not specified, it raises an error. If the password is specified as '-',
        it prompts the user to enter the password interactively.
        """
        if self._theargs.ndexserver and self._theargs.ndexuser and self._theargs.ndexpassword:

            if self._theargs.ndexpassword == '-':
                self._theargs.ndexpassword = getpass.getpass(prompt="Enter NDEx Password: ")

            ndex_uploader = NDExHierarchyUploader(self._theargs.ndexserver, self._theargs.ndexuser,
                                                  self._theargs.ndexpassword, self._theargs.visibility)

            if self.parent_network is None:
                logger.warning("Parent network was not specified. Hierarchy will not be in cell view.")
                cx_factory = RawCX2NetworkFactory()
                hierarchy_network = cx_factory.get_cx2network(self._get_hierarchy_dest_file())
                _, hierarchyurl = ndex_uploader._save_network(hierarchy_network)
            else:
                if os.path.isfile(self.parent_network):
                    _, _, _, hierarchyurl = ndex_uploader.upload_hierarchy_and_parent_network_from_files(
                        hierarchy_path=self._get_hierarchy_dest_file(), parent_path=self.parent_network)
                else:
                    cx_factory = RawCX2NetworkFactory()
                    hierarchy_network = cx_factory.get_cx2network(self._get_hierarchy_dest_file())
                    _, _, _, hierarchyurl = ndex_uploader.save_hierarchy_and_parent_network(hierarchy_network,
                                                                                            self.parent_network)

            print(f'Hierarchy uploaded. To view hierarchy on NDEx please paste this URL in your '
                  f'browser {hierarchyurl}. To view Hierarchy on new experimental Cytoscape on the Web, go to '
                  f'{ndex_uploader.get_cytoscape_url(hierarchyurl)}')

    @staticmethod
    def _annotate_with_score(hierarchy, original_hierarchy, node_id, score_name, score):
        hierarchy.add_node_attribute(node_id, score_name, score, datatype='double')
        if original_hierarchy is not None:
            original_hierarchy.add_node_attribute(node_id, score_name, score, datatype='double')

    def annotate(self, annotation_dict):
        """
        Annotates the hierarchy with P_rho scores from the given annotation dictionary,
        updating node attributes within the hierarchy file.

        :param annotation_dict: A dictionary mapping terms to their P_rho scores.
        :type annotation_dict: dict
        """
        factory = RawCX2NetworkFactory()
        hierarchy = factory.get_cx2network(self.hierarchy)
        original_hierarchy = None
        if self.original_hierarchy is not None:
            original_hierarchy = factory.get_cx2network(self.original_hierarchy)

        for term, score in annotation_dict.items():
            node_id = term
            if not isinstance(term, int):
                node_id = hierarchy.lookup_node_id_by_name(term)
            if node_id is not None:
                self._annotate_with_score(hierarchy, original_hierarchy, node_id, vnnconstants.PRHO_SCORE, score[0])
                self._annotate_with_score(hierarchy, original_hierarchy, node_id, vnnconstants.P_PVAL_SCORE, score[1])
                self._annotate_with_score(hierarchy, original_hierarchy, node_id, vnnconstants.CRHO_SCORE, score[2])
                self._annotate_with_score(hierarchy, original_hierarchy, node_id, vnnconstants.C_PVAL_SCORE, score[3])
                self._annotate_with_score(hierarchy, original_hierarchy, node_id, vnnconstants.RLIPP_SCORE, score[4])

        # TODO: apply style to the hierarchy
        path_to_style_network = os.path.join(os.path.dirname(cellmaps_vnn.__file__), 'nest_style.cx2')
        style_network = factory.get_cx2network(path_to_style_network)
        vis_prop = style_network.get_visual_properties()
        hierarchy.set_visual_properties(vis_prop)
        hierarchy.write_as_raw_cx2(self._get_hierarchy_dest_file())
        if original_hierarchy is not None:
            original_hierarchy.set_visual_properties(vis_prop)
            original_hierarchy.write_as_raw_cx2(self._get_original_hierarchy_dest_file())

    def run(self):
        """
        The logic for annotating hierarchy with prediction results from cellmaps_vnn. It aggregates prediction scores
        from models, optionally filters them for a specific disease, and annotates the hierarchy with these scores.
        """
        self._aggregate_prediction_scores_from_models()
        filepath = self._get_rlipp_out_dest_file()
        data = pd.read_csv(filepath, sep='\t')
        if self._theargs.disease is None:
            annotation_dict = self._aggregate_scores_from_diseases(data)
        else:
            annotation_dict = self._get_scores_for_disease(self._theargs.disease, data)
        if len(annotation_dict) == 0:
            logger.error("No system importance scores available for annotation.")
            raise CellmapsvnnError("No system importance scores available for annotation. "
                                   "Please ensure valid data is provided for the hierarchy annotation.")
        self.annotate(annotation_dict)
        self._upload_to_ndex_if_credentials_provided()

    def register_outputs(self, outdir, description, keywords, provenance_utils):
        """
        Registers the output files of the annotation process with the FAIRSCAPE service for data provenance.
        This includes the annotated hierarchy and the RLIPP output files.

        :param outdir: The output directory where the files are stored.
        :type outdir: str
        :param description: A description of the files for provenance registration.
        :type description: str
        :param keywords: A list of keywords associated with the files.
        :type keywords: list
        :param provenance_utils: The utility class for provenance registration.
        :type provenance_utils: ProvenanceUtility
        :return: A list of dataset IDs assigned to the registered files.
        :rtype: list
        """
        hierarchy_id = self._register_hierarchy(outdir, description, keywords, provenance_utils)
        rlipp_id = self._register_rlipp_file(outdir, description, keywords, provenance_utils)
        return_ids = [hierarchy_id, rlipp_id]
        gene2ind_path = os.path.join(self._theargs.model_predictions[0], 'gene2ind.txt')
        if os.path.exists(gene2ind_path):
            gene2ind_id = copy_and_register_gene2id_file(gene2ind_path, outdir, description, keywords,
                                                         provenance_utils)
            return_ids.append(gene2ind_id)
        if self.original_hierarchy is not None:
            original_hierarchy_id = self._register_original_hierarchy(outdir, description, keywords, provenance_utils)
            return_ids.append(original_hierarchy_id)
        if self.parent_network is not None and os.path.isfile(self.parent_network):
            hierarchy_parent_id = self._copy_and_register_hierarchy_parent(outdir, description, keywords,
                                                                           provenance_utils)
            return_ids.append(hierarchy_parent_id)
        return return_ids

    def _register_hierarchy(self, outdir, description, keywords, provenance_utils):
        """
        Register annotated hierarchy file with the FAIRSCAPE service for data provenance.

        :param outdir: The output directory where the outputs are stored.
        :param description: Description of the file for provenance registration.
        :param keywords: List of keywords associated with the file.
        :param provenance_utils: The utility class for provenance registration.

        :return: The dataset ID assigned to the registered file.
        """
        hierarchy_out_file = self._get_original_hierarchy_dest_file()

        data_dict = {'name': os.path.basename(hierarchy_out_file) + ' Annotated hierarchy file that was used to build '
                                                                    'VNN',
                     'description': description + ' Annotated hierarchy file that was used to build VNN',
                     'keywords': keywords,
                     'data-format': 'CX2',
                     'author': cellmaps_vnn.__name__,
                     'version': cellmaps_vnn.__version__,
                     'date-published': date.today().strftime('%m-%d-%Y')}
        dataset_id = provenance_utils.register_dataset(outdir,
                                                       source_file=hierarchy_out_file,
                                                       data_dict=data_dict)
        return dataset_id

    def _register_original_hierarchy(self, outdir, description, keywords, provenance_utils):
        """
        Register annotated hierarchy file with the FAIRSCAPE service for data provenance.

        :param outdir: The output directory where the outputs are stored.
        :param description: Description of the file for provenance registration.
        :param keywords: List of keywords associated with the file.
        :param provenance_utils: The utility class for provenance registration.

        :return: The dataset ID assigned to the registered file.
        """
        hierarchy_out_file = self._get_hierarchy_dest_file()

        data_dict = {'name': os.path.basename(hierarchy_out_file) + ' Annotated hierarchy file',
                     'description': description + ' Annotated hierarchy file',
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
        hierarchy_parent_out_file = os.path.join(outdir, 'hierarchy_parent.cx2')
        shutil.copy(self.parent_network, hierarchy_parent_out_file)

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
        """
        Registers the rlipp aggregated file with the FAIRSCAPE service for data provenance.

        :param outdir: The output directory where the outputs are stored.
        :param description: Description of the file for provenance registration.
        :param keywords: List of keywords associated with the file.
        :param provenance_utils: The utility class for provenance registration.

        :return: The dataset ID assigned to the registered file.
        """
        dest_path = self._get_rlipp_out_dest_file()
        description = description
        description += ' rlipp results file averaged with multiple models'
        keywords = keywords
        keywords.extend(['file'])
        data_dict = {'name': os.path.basename(dest_path) + ' rlipp aggregated file',
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
