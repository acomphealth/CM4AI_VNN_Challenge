import logging
import copy
import torch
import torch.nn as nn
from cellmaps_vnn.data_wrapper import TrainingDataWrapper
from cellmaps_vnn.exceptions import CellmapsvnnError

logger = logging.getLogger(__name__)


class VNN(nn.Module):

    def __init__(self, data_wrapper: TrainingDataWrapper):
        """
        Initializes the VNN model with the provided data wrapper.

        This constructor sets up components of the VNN model, including term maps, gene mappings, dropout
        parameters, and initializes neural network layers based on the given data structure. It also calculates
        the dimensions for each term and constructs the direct gene layers and the neural network graph.

        :param data_wrapper: The necessary data and configurations for initializing the VNN model.
        :type data_wrapper: TrainingDataWrapper
        :raises CellmapsvnnError: If an error occurs during the initialization of the neural network.
        """
        super().__init__()

        self.term_neighbor_map = None
        self.term_layer_list = None
        self.term_dim_map = None
        self.root = data_wrapper.root
        self.num_hiddens_genotype = data_wrapper.num_hiddens_genotype
        self.gene_id_mapping = data_wrapper.gene_id_mapping

        # dictionary from terms to genes directly annotated with the term
        self.term_direct_gene_map = data_wrapper.term_direct_gene_map

        # Dropout Params
        self.min_dropout_layer = data_wrapper.min_dropout_layer
        self.dropout_fraction = data_wrapper.dropout_fraction

        try:
            # calculate the number of values in a state (term): term_size_map is the number of all genes annotated with
            # the term
            self.cal_term_dim(data_wrapper.term_size_map)

            # ngenes, gene_dim are the number of all genes
            self.gene_dim = len(self.gene_id_mapping)

            # No of input features per gene
            self.feature_dim = len(data_wrapper.cell_features[0, 0, :])

            # add modules for neural networks to process genotypes
            self.construct_direct_gene_layer()
            self.construct_nn_graph(copy.deepcopy(data_wrapper.digraph))

            # add module for final layer
            self.add_module('final_aux_linear_layer', nn.Linear(data_wrapper.num_hiddens_genotype, 1))
            self.add_module('final_linear_layer_output', nn.Linear(1, 1))
        except Exception as e:
            raise CellmapsvnnError(f"Error in VNN initialization: {e}")

    def cal_term_dim(self, term_size_map):
        """
        Calculates the dimensionality of each term based on the term sizes.

        This method updates the `term_dim_map` attribute, which maps each term to its dimensionality.
        The dimensionality for each term is set to the number of hidden genotype variables.

        :param term_size_map: A mapping of terms to their sizes.
        :type term_size_map: dict
        """
        self.term_dim_map = {term: int(self.num_hiddens_genotype) for term in term_size_map}

    def construct_direct_gene_layer(self):
        """
        Constructs layers for genes directly annotated with each term.

        This method iterates through each gene and term to create specific layers in the neural network. For each gene,
        it adds a feature layer and a batch normalization layer. For each term, if there are genes directly annotated
        with it, it adds a linear layer that takes all genes as input and outputs only those genes directly annotated
        with the term. If a term has no directly associated genes, the method will raise exception.
        """
        for gene, _ in self.gene_id_mapping.items():
            self.add_module(gene + '_feature_layer', nn.Linear(self.feature_dim, 1))
            self.add_module(gene + '_batchnorm_layer', nn.BatchNorm1d(1))

        for term, gene_set in self.term_direct_gene_map.items():
            if len(gene_set) == 0:
                raise CellmapsvnnError(f'There are no directly associated genes for term: {term}')
            self.add_module(term + '_direct_gene_layer', nn.Linear(self.gene_dim, len(gene_set)))

    def construct_nn_graph(self, digraph):
        """
        Constructs a neural network graph based on given hierarchy.

        This method builds the neural network by starting from the bottom (leaves) of the given directed graph (digraph)
        and iteratively adding modules for each term in the hierarchy. The method stores the built neural network layers
        in `term_layer_list` and maintains a map (`term_neighbor_map`) of each term to its children.

        For each term, the method calculates the input size, which is the sum of the dimensions of its children and the
        number of genes directly annotated by the term. It then adds a series of layers (dropout, linear,
        batch normalization, and auxiliary linear layers) for each term.

        The process continues until all nodes (terms) in the digraph have been processed and added to the network.

        :param digraph: A directed graph representing the ontology, where nodes are terms and edges
                                        indicate term relationships.
        :type digraph: networkx.DiGraph
        """
        self.term_layer_list = []
        self.term_neighbor_map = {}

        for term in digraph.nodes():
            self.term_neighbor_map[term] = []
            for child in digraph.neighbors(term):
                self.term_neighbor_map[term].append(child)

        i = 0
        while True:
            leaves = [n for n in digraph.nodes() if digraph.out_degree(n) == 0]
            if len(leaves) == 0:
                break
            self.term_layer_list.append(leaves)

            for term in leaves:
                input_size = 0
                for child in self.term_neighbor_map[term]:
                    input_size += self.term_dim_map[child]
                if term in self.term_direct_gene_map:
                    input_size += len(self.term_direct_gene_map[term])

                term_hidden = self.term_dim_map[term]

                if i >= self.min_dropout_layer:
                    self.add_module(term + '_dropout_layer', nn.Dropout(p=self.dropout_fraction))
                self.add_module(term + '_linear_layer', nn.Linear(input_size, term_hidden))
                self.add_module(term + '_batchnorm_layer', nn.BatchNorm1d(term_hidden))
                self.add_module(term + '_aux_linear_layer1', nn.Linear(term_hidden, 1))
                self.add_module(term + '_aux_linear_layer2', nn.Linear(1, 1))

            i += 1
            digraph.remove_nodes_from(leaves)

    def forward(self, x):
        """
        Defines the forward function of the VNN model.

        This method processes the input through the neural network constructed in the VNN class. It applies
        a series of transformations to the input data, including feature layer operations, batch normalization,
        and tanh activations. The method aggregates outputs from different terms in the network and finally
        produces two dictionaries: one for hidden embeddings and one for auxiliary outputs.

        :param x: Input tensor representing gene data. Each row corresponds to a gene, and columns are features.
        :type x: torch.Tensor

        :returns: A tuple containing two dictionaries:
                  - hidden_embeddings_map: A mapping from terms to their hidden embeddings.
                  - aux_out_map: A mapping from terms to their auxiliary output.
        :rtype: (dict, dict)
        """
        hidden_embeddings_map = {}
        aux_out_map = {}

        feat_out_list = []
        for gene, i in self.gene_id_mapping.items():
            feat_out = torch.tanh(self._modules[gene + '_feature_layer'](x[:, i, :]))
            hidden_embeddings_map[gene] = self._modules[gene + '_batchnorm_layer'](feat_out)
            feat_out_list.append(hidden_embeddings_map[gene])

        gene_input = torch.cat(feat_out_list, dim=1)
        term_gene_out_map = {}
        for term, _ in self.term_direct_gene_map.items():
            term_gene_out_map[term] = self._modules[term + '_direct_gene_layer'](gene_input)

        for i, layer in enumerate(self.term_layer_list):

            for term in layer:

                child_input_list = []
                for child in self.term_neighbor_map[term]:
                    child_input_list.append(hidden_embeddings_map[child])

                if term in self.term_direct_gene_map:
                    child_input_list.append(term_gene_out_map[term])

                child_input = torch.cat(child_input_list, 1)
                if i >= self.min_dropout_layer:
                    dropout_out = self._modules[term + '_dropout_layer'](child_input)
                    term_nn_out = self._modules[term + '_linear_layer'](dropout_out)
                else:
                    term_nn_out = self._modules[term + '_linear_layer'](child_input)
                tanh_out = torch.tanh(term_nn_out)
                hidden_embeddings_map[term] = self._modules[term + '_batchnorm_layer'](tanh_out)
                aux_layer1_out = torch.tanh(self._modules[term + '_aux_linear_layer1'](hidden_embeddings_map[term]))
                aux_out_map[term] = self._modules[term + '_aux_linear_layer2'](aux_layer1_out)

        final_input = hidden_embeddings_map[self.root]
        aux_layer_out = torch.tanh(self._modules['final_aux_linear_layer'](final_input))
        aux_out_map['final'] = self._modules['final_linear_layer_output'](aux_layer_out)

        return aux_out_map, hidden_embeddings_map
