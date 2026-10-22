import torch as tensor_engine
from gudhi.wasserstein import wasserstein_distance as transport_distance

from graph_data import aggregate_group_graph
from persistence_math import PersistenceEncoder


class TemporalPenalty(tensor_engine.nn.Module):
    def __init__(instance, neighboring_diagrams, run_profile) -> None:
        super().__init__()
        instance.zero_dimensional_encoder = PersistenceEncoder(homology_dimension=0, diagram_capacity=run_profile.diagram_capacity)
        instance.one_dimensional_encoder = PersistenceEncoder(homology_dimension=1, diagram_capacity=run_profile.diagram_capacity)
        instance.neighboring_diagrams = neighboring_diagrams
        instance.temporal_scale = run_profile.temporal_scale

    def forward(instance, sparse_adjacency, soft_assignments):
        grouped_adjacency = aggregate_group_graph(soft_assignments, sparse_adjacency)
        current_zero_diagram = instance.zero_dimensional_encoder(grouped_adjacency)
        current_one_diagram = instance.one_dimensional_encoder(grouped_adjacency)
        if instance.neighboring_diagrams[0]:
            previous_zero_distance = transport_distance(current_zero_diagram, instance.neighboring_diagrams[0][0], order=1, enable_autodiff=True, keep_essential_parts=False)
            previous_one_distance = transport_distance(current_one_diagram, instance.neighboring_diagrams[0][1], order=1, enable_autodiff=True, keep_essential_parts=False)
        if instance.neighboring_diagrams[1]:
            next_zero_distance = transport_distance(current_zero_diagram, instance.neighboring_diagrams[1][0], order=1, enable_autodiff=True, keep_essential_parts=False)
            next_one_distance = transport_distance(current_one_diagram, instance.neighboring_diagrams[1][1], order=1, enable_autodiff=True, keep_essential_parts=False)
        if not instance.neighboring_diagrams[0]:
            temporal_distance = next_zero_distance + next_one_distance
        elif not instance.neighboring_diagrams[1]:
            temporal_distance = previous_zero_distance + previous_one_distance
        else:
            temporal_distance = previous_zero_distance + previous_one_distance + next_zero_distance + next_one_distance
        return instance.temporal_scale * temporal_distance
