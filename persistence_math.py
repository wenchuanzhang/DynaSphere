import numpy as array_engine
import networkx as graph_library
import torch as tensor_engine
import gudhi as persistence_engine


def build_weight_filtration(weighted_graph: graph_library.Graph):
    simplex_structure = persistence_engine.SimplexTree()
    for vertex_id in weighted_graph.nodes():
        simplex_structure.insert([vertex_id], filtration=0)
    descending_weights = array_engine.unique([edge_record[2] for edge_record in weighted_graph.edges.data("weight")])[::-1]
    for level_index, threshold_weight in enumerate(descending_weights):
        threshold_graph = weighted_graph.edge_subgraph([(left_vertex, right_vertex) for left_vertex, right_vertex, edge_weight in weighted_graph.edges.data("weight") if edge_weight >= threshold_weight])
        for maximal_clique in graph_library.find_cliques(threshold_graph):
            simplex_structure.insert(maximal_clique, filtration=1 / threshold_weight)
    return simplex_structure


def extract_persistence_indices(weight_matrix, homology_dimension, diagram_capacity):
    simplex_structure = build_weight_filtration(graph_library.from_numpy_array(weight_matrix))
    simplex_structure.persistence()
    persistence_pairs = simplex_structure.persistence_pairs()
    endpoint_indices = []
    persistence_lengths = []
    for birth_simplex, death_simplex in persistence_pairs:
        if len(birth_simplex) == homology_dimension + 1 and len(death_simplex) > 0:
            birth_vertices = array_engine.array(birth_simplex)
            death_vertices = array_engine.array(death_simplex)
            birth_edge = [birth_simplex[position] for position in array_engine.unravel_index(array_engine.argmax(weight_matrix[birth_vertices, :][:, birth_vertices]), [len(birth_simplex), len(birth_simplex)])]
            death_edge = [death_simplex[position] for position in array_engine.unravel_index(array_engine.argmax(weight_matrix[death_vertices, :][:, death_vertices]), [len(death_simplex), len(death_simplex)])]
            endpoint_indices += birth_edge
            endpoint_indices += death_edge
            persistence_lengths.append(simplex_structure.filtration(death_simplex) - simplex_structure.filtration(birth_simplex))
    descending_order = array_engine.argsort(persistence_lengths)
    endpoint_indices = list(array_engine.reshape(endpoint_indices, [-1, 4])[descending_order][::-1, :].flatten())
    endpoint_indices = endpoint_indices[: 4 * diagram_capacity] + [0 for padding_index in range(0, max(0, 4 * diagram_capacity - len(endpoint_indices)))]
    return array_engine.array(endpoint_indices, dtype=array_engine.int32)


class PersistenceEncoder(tensor_engine.nn.Module):
    def __init__(instance, homology_dimension=1, diagram_capacity=50):
        super(PersistenceEncoder, instance).__init__()
        instance.homology_dimension = homology_dimension
        instance.diagram_capacity = diagram_capacity

    def forward(instance, grouped_adjacency: tensor_engine.Tensor):
        homology_dimension = instance.homology_dimension
        diagram_capacity = instance.diagram_capacity
        cpu_grouped_adjacency = grouped_adjacency.cpu()
        with tensor_engine.no_grad():
            flattened_indices = tensor_engine.from_numpy(extract_persistence_indices(cpu_grouped_adjacency.numpy(), homology_dimension, diagram_capacity))
        if homology_dimension > 0:
            paired_indices = flattened_indices.view([2 * diagram_capacity, 2]).long()
            persistence_diagram = cpu_grouped_adjacency[paired_indices[:, 0], paired_indices[:, 1]].view(diagram_capacity, 2)
        else:
            paired_indices = flattened_indices.view([2 * diagram_capacity, 2])[1::2, :].long()
            persistence_diagram = tensor_engine.cat([tensor_engine.zeros(diagram_capacity, 1), cpu_grouped_adjacency[paired_indices[:, 0], paired_indices[:, 1]].view(diagram_capacity, 1).float()], dim=1)
        return persistence_diagram.cuda()
