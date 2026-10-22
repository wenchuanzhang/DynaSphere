import time as clock_engine
import torch as tensor_engine
import torch.nn.functional as tensor_functions

from diffusion_math import reconstruct_propagated_graph, scheduled_scale
from records import convert_sparse_layout
from sparse_math import reconstruction_accuracy
from sphere_math import uniform_direction_divergence
from temporal_objective import TemporalPenalty


def optimize_snapshot(network_model, sparse_features, sparse_adjacency, run_profile, temporal_penalty, snapshot_key):
    execution_device = "cuda"
    adjacency_coordinates, adjacency_values, adjacency_shape = convert_sparse_layout(sparse_adjacency)
    adjacency_indices = tensor_engine.LongTensor(adjacency_coordinates.T).to(execution_device)
    adjacency_weights = tensor_engine.FloatTensor(adjacency_values).to(execution_device)
    target_adjacency = tensor_engine.sparse_coo_tensor(adjacency_indices, adjacency_weights, tensor_engine.Size(adjacency_shape), device=execution_device, dtype=tensor_engine.float32).coalesce()
    positive_weight = float(sparse_adjacency.shape[0] * sparse_adjacency.shape[0] - sparse_adjacency.sum()) / sparse_adjacency.sum()
    reconstruction_normalizer = sparse_adjacency.shape[0] * sparse_adjacency.shape[0] / float((sparse_adjacency.shape[0] * sparse_adjacency.shape[0] - sparse_adjacency.sum()) * 2)
    positive_locations = target_adjacency.to_dense().view(-1) == 1
    element_weights = tensor_engine.ones(positive_locations.size(0), device=execution_device)
    element_weights[positive_locations] = positive_weight
    update_rule = tensor_engine.optim.AdamW(network_model.parameters(), lr=run_profile.step_size, weight_decay=1e-4)
    network_model.train()
    for round_index in range(run_profile.training_rounds):
        round_start_time = clock_engine.time()
        partition_switch = round_index > run_profile.delayed_stage
        if temporal_penalty:
            partition_switch = True
        base_reconstruction, embedded_features, soft_assignments = network_model(sparse_features, partition_switch)
        propagated_reconstruction = reconstruct_propagated_graph(embedded_features, network_model.embedding_projector.normalized_adjacency, step_total=run_profile.propagation_steps)
        base_reconstruction_loss = reconstruction_normalizer * tensor_functions.binary_cross_entropy(base_reconstruction.view(-1), target_adjacency.to_dense().view(-1), weight=element_weights)
        propagated_reconstruction_loss = reconstruction_normalizer * tensor_functions.binary_cross_entropy(propagated_reconstruction.view(-1), target_adjacency.to_dense().view(-1), weight=element_weights)
        propagation_weight = scheduled_scale(round_index, opening_round=run_profile.delayed_stage, closing_round=run_profile.delayed_stage + 100, ceiling=run_profile.propagation_scale)
        total_reconstruction_loss = base_reconstruction_loss + propagation_weight * propagated_reconstruction_loss
        normalized_embeddings = tensor_functions.normalize(embedded_features, p=2, dim=1)
        concentration_value = network_model.concentration()
        direction_divergence = uniform_direction_divergence(concentration_value, ambient_dimension=normalized_embeddings.size(1))
        divergence_weight = scheduled_scale(round_index, opening_round=run_profile.delayed_stage, closing_round=run_profile.delayed_stage + 100, ceiling=0.01)
        divergence_loss = divergence_weight * direction_divergence
        partition_weight = scheduled_scale(round_index, opening_round=run_profile.delayed_stage, closing_round=run_profile.delayed_stage + 100, ceiling=run_profile.partition_scale)
        if soft_assignments is not None and partition_weight > 0.0:
            normalized_prototypes = tensor_functions.normalize(network_model.prototype_matrix, p=2, dim=1)
            assignment_probabilities = soft_assignments
            within_group_cost = (assignment_probabilities[:, :, None] * (normalized_embeddings[:, None, :] - normalized_prototypes[None, :, :]).pow(2)).sum() / normalized_embeddings.size(0)
            between_group_distance = tensor_engine.pdist(normalized_prototypes, p=2).mean()
            raw_partition_loss = within_group_cost / (between_group_distance + 1e-9)
            partition_loss = partition_weight * raw_partition_loss
        else:
            raw_partition_loss = tensor_engine.tensor(0.0, device=execution_device)
            partition_loss = raw_partition_loss
        if temporal_penalty:
            safe_assignments = soft_assignments if soft_assignments is not None else network_model.row_partition(tensor_engine.mm(embedded_features, tensor_engine.linalg.pinv(network_model.prototype_matrix)))
            total_loss = temporal_penalty(sparse_adjacency, safe_assignments) + total_reconstruction_loss + divergence_loss + partition_loss
        elif round_index > run_profile.delayed_stage:
            centroid_reconstruction_loss = tensor_functions.mse_loss(embedded_features, tensor_engine.mm(soft_assignments, network_model.prototype_matrix))
            total_loss = 1 * centroid_reconstruction_loss + total_reconstruction_loss + divergence_loss + partition_loss
        else:
            total_loss = total_reconstruction_loss + divergence_loss + partition_loss
        update_rule.zero_grad()
        total_loss.backward()
        update_rule.step()
        with tensor_engine.no_grad():
            base_accuracy = reconstruction_accuracy(base_reconstruction, target_adjacency.to_dense())
            propagated_accuracy = reconstruction_accuracy(propagated_reconstruction, target_adjacency.to_dense())
        if round_index % 100 == 0:
            displayed_accuracy = 0 if int(snapshot_key) >= 9 else base_accuracy
            print("Epoch:", "%04d" % (round_index + 1), "re_base=", "{:.5f}".format(base_reconstruction_loss.item()), "re_diff=", "{:.5f}".format(propagated_reconstruction_loss.item()), "kl=", "{:.5f}".format(direction_divergence.item() if tensor_engine.is_tensor(direction_divergence) else float(direction_divergence)), "clu=", "{:.5f}".format(raw_partition_loss.item() if tensor_engine.is_tensor(raw_partition_loss) else float(raw_partition_loss)), "Acc(diff)=", "{:.5f}".format(propagated_accuracy), "elapsed=", "{:.5f}".format(clock_engine.time() - round_start_time))


def run_initial_optimization(architecture_key, network_model, sparse_features, sparse_adjacency, run_profile, snapshot_key):
    if architecture_key == "DynaSphere":
        optimize_snapshot(network_model, sparse_features, sparse_adjacency, run_profile, None, snapshot_key)
        return
    raise ValueError(f"Unknown network type: {architecture_key}")


def run_temporal_optimization(architecture_key, network_model, neighboring_diagrams, sparse_adjacency, sparse_features, run_profile, snapshot_key):
    temporal_penalty = TemporalPenalty(neighboring_diagrams=neighboring_diagrams, run_profile=run_profile)
    if architecture_key == "DynaSphere":
        optimize_snapshot(network_model, sparse_features, sparse_adjacency, run_profile, temporal_penalty, snapshot_key)
        return
    raise ValueError(f"Unknown network type: {architecture_key}")
