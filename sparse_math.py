import numpy as array_engine
import scipy.sparse as sparse_library
import torch as tensor_engine

from records import convert_sparse_layout


def normalize_adjacency(adjacency_object):
    if isinstance(adjacency_object, sparse_library.coo_matrix):
        adjacency_with_loops = adjacency_object + sparse_library.eye(adjacency_object.shape[0])
        row_totals = array_engine.array(adjacency_with_loops.sum(1))
        inverse_root_degrees = sparse_library.diags(array_engine.power(row_totals, -0.5).flatten())
        normalized_adjacency = adjacency_with_loops.dot(inverse_root_degrees).transpose().dot(inverse_root_degrees).tocoo()
        return convert_sparse_layout(normalized_adjacency)
    if isinstance(adjacency_object, tensor_engine.Tensor):
        execution_device = tensor_engine.device("cuda" if adjacency_object.is_cuda else "cpu")
        adjacency_with_loops = adjacency_object + tensor_engine.eye(adjacency_object.shape[0]).to(execution_device)
        row_totals = adjacency_with_loops.sum(1)
        inverse_root_degrees = row_totals.pow(-1 / 2).flatten()
        inverse_root_degrees[tensor_engine.isinf(inverse_root_degrees)] = 0.0
        inverse_degree_matrix = tensor_engine.diag(inverse_root_degrees)
        normalized_adjacency = tensor_engine.mm(inverse_degree_matrix, adjacency_with_loops)
        normalized_adjacency = tensor_engine.mm(normalized_adjacency, inverse_degree_matrix)
        return normalized_adjacency
    raise TypeError


def reconstruction_accuracy(reconstructed_adjacency, target_adjacency):
    integer_targets = target_adjacency.long()
    integer_predictions = tensor_engine.gt(reconstructed_adjacency, 0.5).long()
    return tensor_engine.eq(integer_targets, integer_predictions).float().mean()
