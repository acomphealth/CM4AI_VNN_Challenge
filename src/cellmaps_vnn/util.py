import math
import os
import logging
import shutil
from datetime import date

import cellmaps_vnn
import numpy as np
import pandas as pd
import torch
from torch import inf

from cellmaps_vnn.exceptions import CellmapsvnnError

logger = logging.getLogger(__name__)


def calc_std_vals(df, zscore_method):
    """
    Calculates standard deviation values for a given DataFrame based on the specified z-score method
    ('zscore' and 'robustz').

    :param df: the data to be standardized.
    :type df: pandas.DataFrame
    :param zscore_method: Method to use for standardization ('zscore' or 'robustz').
    :type zscore_method: str

    :returns std_df: DataFrame with standard deviation values for each dataset.
    :rtype std_df: pandas.DataFrame
    """
    std_df = pd.DataFrame(columns=['dataset', 'center', 'scale'])
    std_list = []

    if zscore_method == 'zscore':
        for name, group in df.groupby(['dataset'])['auc']:
            if isinstance(name, tuple) and len(name) == 1:
                name = name[0]
            center = group.mean()
            scale = group.std()
            if math.isnan(scale) or scale == 0.0:
                scale = 1.0
            temp = pd.DataFrame([[name, center, scale]], columns=std_df.columns)
            std_list.append(temp)

    elif zscore_method == 'robustz':
        for name, group in df.groupby(['dataset'])['auc']:
            if isinstance(name, tuple) and len(name) == 1:
                name = name[0]
            center = group.median()
            scale = group.quantile(0.75) - group.quantile(0.25)
            if math.isnan(scale) or scale == 0.0:
                scale = 1.0
            temp = pd.DataFrame([[name, center, scale]], columns=std_df.columns)
            std_list.append(temp)
    else:
        for name, group in df.groupby(['dataset'])['auc']:
            if isinstance(name, tuple) and len(name) == 1:
                name = name[0]
            temp = pd.DataFrame([[name, 0.0, 1.0]], columns=std_df.columns)
            std_list.append(temp)

    std_df = pd.concat(std_list, ignore_index=True)
    return std_df


def standardize_data(df, std_df):
    """
    Standardizes the data based on provided standard deviation values. This function applies z-score standardization
    to the 'auc' column of the DataFrame, using the standard deviation values provided.

    :param df: the data to be standardized.
    :type df: pandas.DataFrame
    :param std_df: the standard deviation values.
    :type std_df: pandas.DataFrame

    :returns merged: DataFrame with the standardized 'z' values.
    :rtype merged: pandas.DataFrame
    """
    merged = pd.merge(df, std_df, how="left", on=['dataset'], sort=False)
    merged['z'] = (merged['auc'] - merged['center']) / merged['scale']
    merged = merged[['cell_line', 'smiles', 'z']]
    return merged


def load_numpy_data(file_path):
    """
    Reads a file at the specified path and attempts to convert it into a NumPy array.
    If the file is not found or any other error occurs, an exception is raised.

    :param file_path: Path to the file to be loaded.
    :type file_path: str

    :returns: Data loaded from the file.
    :rtype: numpy.ndarray

    :raises CellmapsvnnError: If the file is not found or an error occurs during loading.
    """
    if not os.path.isfile(file_path):
        raise CellmapsvnnError(f"File {file_path} not found.")

    try:
        return np.genfromtxt(file_path, delimiter=',')
    except Exception as e:
        raise CellmapsvnnError(f"Error loading data from {file_path}: {e}")


def load_cell_features(mutations, cn_deletions, cn_amplifications):
    """
    Loads and combines cell/drug features from given mutation, CN deletion, and CN amplification files.

    Each feature set is loaded as a NumPy array and then combined into a single array.

    :param mutations: Path to the mutations data file.
    :type mutations: str
    :param cn_deletions: Path to the CN deletions data file.
    :type cn_deletions: str
    :param cn_amplifications: Path to the CN amplifications data file.
    :type cn_amplifications: str

    :returns: Combined cell features.
    :rtype: numpy.ndarray
    """
    mutations = load_numpy_data(mutations)
    cn_deletions = load_numpy_data(cn_deletions)
    cn_amplifications = load_numpy_data(cn_amplifications)
    return np.dstack([mutations, cn_deletions, cn_amplifications])


def load_mapping(mapping_file, mapping_type):
    """
    Loads a mapping from a file and returns it as a dictionary.

    :param mapping_file: Path to the mapping file.
    :type mapping_file: str
    :param mapping_type: Description of the mapping (e.g., 'gene to ID').
    :type mapping_type: str

    :returns mapping: Dictionary containing the mapping from the file.
    :rtype mapping: dict

    :raises CellmapsvnnError: If the mapping file is not found.
    """
    if not os.path.isfile(mapping_file):
        raise CellmapsvnnError(f"Mapping file {mapping_file} not found.")

    mapping = {}
    file_handle = open(mapping_file)
    for line in file_handle:
        line = line.rstrip().split()
        mapping[line[1]] = int(line[0])

    file_handle.close()
    logger.info('Total number of {} = {}'.format(mapping_type, len(mapping)))
    return mapping


def create_term_mask(term_direct_gene_map, gene_dim, cuda_id=None):
    """
    Creates a term mask map for gene sets. This function generates a mask for each term where the mask is
    a matrix with rows equal to the number of relevant gene set and columns equal to the total number of genes.
    Each element is set to 1 if the corresponding gene is one of the relevant genes.

    :param term_direct_gene_map: Mapping of terms to their respective gene sets.
    :type term_direct_gene_map: dict
    :param gene_dim: Total number of genes.
    :type gene_dim: int
    :param cuda_id: CUDA ID for tensor operations.
    :type cuda_id: int

    :returns term_mask_map: Dictionary of term masks.
    :rtype term_mask_map: dict
    """
    term_mask_map = {}
    for term, gene_set in term_direct_gene_map.items():
        mask = torch.zeros(len(gene_set), gene_dim)
        if cuda_id is not None and torch.cuda.is_available():
            mask = mask.cuda(cuda_id)
        for i, gene_id in enumerate(gene_set):
            mask[i, gene_id] = 1
        term_mask_map[term] = mask
    return term_mask_map


def build_input_vector(input_data, cell_features):
    """
    Builds an input vector for model training using cell features.

    :param input_data: Input data containing cell indices.
    :type input_data: Tensor
    :param cell_features: Cell features array.
    :type cell_features: numpy.ndarray

    :returns feature: Input feature tensor for the model.
    :rtype feature: Tensor
    """
    genedim = len(cell_features[0, :])
    featdim = len(cell_features[0, 0, :])
    feature = np.zeros((input_data.size()[0], genedim, featdim))

    for i in range(input_data.size()[0]):
        feature[i] = cell_features[int(input_data[i, 0])]

    feature = torch.from_numpy(feature).float()
    return feature


def get_grad_norm(model_params, norm_type):
    """
    Computes the gradient norm of model parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    :param model_params: Iterable of model parameters or a single Tensor that will have gradients normalized.
    :type model_params: Iterable[Tensor] or Tensor
    :param norm_type: Type of the p-norm to use (can be 'inf' for infinity norm).
    :type norm_type: float or int

    :returns: Total norm of the model parameters (viewed as a single vector).
    :rtype: Tensor
    """
    if isinstance(model_params, torch.Tensor):  # check if parameters are tensorobject
        model_params = [model_params]  # change to list
    model_params = [p for p in model_params if p.grad is not None]  # get list of params with grads
    norm_type = float(norm_type)  # make sure norm_type is of type float
    if len(model_params) == 0:  # if no params provided, return tensor of 0
        return torch.tensor(0.)

    device = model_params[0].grad.device if torch.cuda.is_available() else torch.device("cpu")  # get device
    if norm_type == inf:  # infinity norm
        total_norm = max(p.grad.detach().abs().max().to(device) for p in model_params)
    else:  # total norm
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in model_params]),
                                norm_type)
    return total_norm


def pearson_corr(x, y):
    """
    Computes the Pearson correlation coefficient between two tensors.

    :param x: First variable tensor.
    :type x: Tensor
    :param y: Second variable tensor.
    :type y: Tensor

    :returns: Pearson correlation coefficient.
    :rtype: Tensor
    """
    xx = x - torch.mean(x)
    yy = y - torch.mean(y)

    return torch.sum(xx * yy) / (torch.norm(xx, 2) * torch.norm(yy, 2))


def copy_and_register_gene2id_file(genet2id_in_file, outdir, description, keywords, provenance_utils):
    gene2id_out_file = os.path.join(outdir, 'gene2ind.txt')
    shutil.copy(genet2id_in_file, gene2id_out_file)

    data_dict = {'name': os.path.basename(gene2id_out_file) + ' gene to index mapping file',
                 'description': description + ' gene to index mapping file',
                 'keywords': keywords,
                 'data-format': 'txt',
                 'author': cellmaps_vnn.__name__,
                 'version': cellmaps_vnn.__version__,
                 'date-published': date.today().strftime('%m-%d-%Y')}
    dataset_id = provenance_utils.register_dataset(outdir,
                                                   source_file=gene2id_out_file,
                                                   data_dict=data_dict)
    return dataset_id
