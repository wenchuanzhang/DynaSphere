import numpy as array_engine
import torch as tensor_engine
import torch.nn as neural_networks
import torch.nn.functional as tensor_functions


def create_glorot_parameter(source_width, target_width):
    sampling_radius = array_engine.sqrt(6.0 / (source_width + target_width))
    sampled_parameter = tensor_engine.rand(source_width, target_width) * 2 * sampling_radius - sampling_radius
    return neural_networks.Parameter(sampled_parameter, requires_grad=True)


class SparseGraphProjector(neural_networks.Module):
    def __init__(instance, source_width, target_width, normalized_adjacency, nonlinear_operator=tensor_functions.leaky_relu, **module_options):
        super(SparseGraphProjector, instance).__init__(**module_options)
        instance.projection_matrix = create_glorot_parameter(source_width, target_width)
        instance.normalized_adjacency = normalized_adjacency
        instance.nonlinear_operator = nonlinear_operator

    def forward(instance, source_features):
        projected_features = tensor_engine.mm(source_features, instance.projection_matrix)
        propagated_features = tensor_engine.mm(instance.normalized_adjacency, projected_features)
        return instance.nonlinear_operator(propagated_features)
