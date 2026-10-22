import torch as tensor_engine
import torch.nn as neural_networks

from neural_parts import SparseGraphProjector, create_glorot_parameter


class DynaSphere(neural_networks.Module):
    def __init__(instance, normalized_adjacency, feature_width, run_profile):
        super(DynaSphere, instance).__init__()
        instance.delayed_stage = run_profile.delayed_stage
        instance.embedding_projector = SparseGraphProjector(feature_width, run_profile.latent_width, normalized_adjacency)
        instance.prototype_matrix = create_glorot_parameter(run_profile.group_total, run_profile.latent_width)
        instance.raw_concentration = neural_networks.Parameter(tensor_engine.tensor(10.0))

    def concentration(instance):
        return tensor_engine.nn.functional.softplus(instance.raw_concentration)

    def reset_prototypes(instance):
        tensor_engine.nn.init.xavier_normal_(instance.prototype_matrix.data)

    def embed(instance, source_features):
        return instance.embedding_projector(source_features)

    @staticmethod
    def row_partition(raw_scores):
        range_scaled_scores = (raw_scores - raw_scores.min(dim=1).values[:, None]) / (raw_scores.max(dim=1).values - raw_scores.min(dim=1).values)[:, None]
        return range_scaled_scores / tensor_engine.sum(range_scaled_scores, dim=1)[:, None]

    def forward(instance, source_features, partition_switch):
        embedded_features = instance.embed(source_features)
        reconstructed_adjacency = tensor_engine.sigmoid(tensor_engine.matmul(embedded_features, embedded_features.t()))
        if type(partition_switch) != bool or partition_switch is True:
            prototype_inverse = tensor_engine.linalg.pinv(instance.prototype_matrix)
            soft_assignments = instance.row_partition(tensor_engine.mm(embedded_features, prototype_inverse))
            return reconstructed_adjacency, embedded_features, soft_assignments
        return reconstructed_adjacency, embedded_features, None
