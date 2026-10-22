import torch as tensor_engine
import torch.nn.functional as tensor_functions


def scheduled_scale(round_index: int, opening_round: int, closing_round: int, ceiling: float) -> float:
    if round_index <= opening_round:
        return 0.0
    if round_index >= closing_round:
        return ceiling
    return (round_index - opening_round) / float(closing_round - opening_round) * ceiling


def beta_sequence(step_total: int, lower_beta=1e-4, upper_beta=2e-2, execution_device=None):
    return tensor_engine.linspace(lower_beta, upper_beta, step_total, device=execution_device)


def diffusion_weights(step_total: int, execution_device):
    beta_values = beta_sequence(step_total, execution_device=execution_device)
    alpha_values = 1.0 - beta_values
    cumulative_alpha_values = tensor_engine.cumprod(alpha_values, dim=0)
    complement_roots = tensor_engine.sqrt(1.0 - cumulative_alpha_values)
    reversed_cumulative_values = tensor_engine.flip(complement_roots, dims=[0]).cumsum(0)
    forward_normalizers = tensor_engine.flip(reversed_cumulative_values, dims=[0])
    return complement_roots, forward_normalizers


def reconstruct_propagated_graph(embedding_matrix, normalized_adjacency, step_total=8, opening_step=1):
    execution_device = embedding_matrix.device
    complement_roots, forward_normalizers = diffusion_weights(step_total, execution_device=execution_device)
    normalization_value = forward_normalizers[opening_step - 1]
    accumulated_similarity = None
    propagated_embeddings = tensor_functions.normalize(embedding_matrix, p=2, dim=1)
    for step_index in range(opening_step, step_total + 1):
        if step_index > opening_step:
            if normalized_adjacency.is_sparse:
                propagated_embeddings = tensor_engine.sparse.mm(normalized_adjacency, propagated_embeddings)
            else:
                propagated_embeddings = normalized_adjacency @ propagated_embeddings
            propagated_embeddings = tensor_functions.normalize(propagated_embeddings, p=2, dim=1)
        similarity_matrix = propagated_embeddings @ propagated_embeddings.t()
        step_weight = complement_roots[step_index - 1]
        accumulated_similarity = similarity_matrix * step_weight if accumulated_similarity is None else accumulated_similarity + similarity_matrix * step_weight
    return tensor_engine.clamp(accumulated_similarity / normalization_value, 0, 1)
