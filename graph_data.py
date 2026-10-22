import scipy.sparse as sparse_library
import torch as tensor_engine


def aggregate_group_graph(soft_assignments: tensor_engine.Tensor, sparse_adjacency: sparse_library.coo_matrix):
    hard_group_ids = tensor_engine.argmax(soft_assignments, dim=1)
    hard_assignment_matrix = tensor_engine.stack([tensor_engine.where(hard_group_ids == group_index, 1.0, 0.0) for group_index in range(soft_assignments.size(1))]).T
    weighted_assignments = hard_assignment_matrix * soft_assignments
    if not tensor_engine.is_tensor(sparse_adjacency):
        sparse_adjacency = tensor_engine.tensor(sparse_adjacency.todense(), dtype=tensor_engine.float, device="cuda:0")
    grouped_adjacency = tensor_engine.mm(tensor_engine.mm(weighted_assignments.T, sparse_adjacency), weighted_assignments).fill_diagonal_(0)
    grouped_adjacency = grouped_adjacency / sparse_adjacency.sum()
    return grouped_adjacency
