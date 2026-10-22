import pickle as serialization_engine
import numpy as array_engine
import scipy.sparse as sparse_library
import networkx as graph_library
import torch as tensor_engine

_SUPPORTED_COLLECTIONS = ["enron", "highschool", "DBLP", "Cora", "DBLPdyn"]
_GROUP_COUNTS = {"Cora": 10, "enron": 7, "highschool": 9, "DBLP": 15, "DBLPdyn": 14}
_EXPAND_VERTEX_SET = False


def read_snapshots(dataset_key, architecture_key):
    if dataset_key in _SUPPORTED_COLLECTIONS:
        return read_pickled_snapshots("Data/" + dataset_key, architecture_key)
    raise NameError(f"Unknown dataset {dataset_key}")


def read_pickled_snapshots(dataset_path, architecture_key, expand_vertex_set=_EXPAND_VERTEX_SET):
    with open(dataset_path + ".pkl", "rb") as graph_stream:
        try:
            evolving_graphs = serialization_engine.load(graph_stream, encoding="bytes", fix_imports=True)
        except ValueError:
            graph_stream.seek(0)
            evolving_graphs = serialization_engine.load(graph_stream, encoding="bytes", fix_imports=True, protocol=2)
    with open(dataset_path + "_label.pkl", "rb") as label_stream:
        try:
            membership_records = serialization_engine.load(label_stream, encoding="bytes", fix_imports=True)
        except ValueError:
            label_stream.seek(0)
            membership_records = serialization_engine.load(label_stream, encoding="bytes", fix_imports=True, protocol=2)
    print("Lengths of snapshots:", len(evolving_graphs))
    print("Types of labels:", _GROUP_COUNTS[dataset_path.split("/")[-1]])
    if dataset_path == "DBLP":
        evolving_graphs = evolving_graphs[:8]
    if expand_vertex_set:
        evolving_graphs = create_empty_full_vertex_graphs(evolving_graphs)
    return assemble_snapshot_records(evolving_graphs, membership_records, architecture_key, dataset_path), _GROUP_COUNTS[dataset_path.split("/")[-1]]


def assemble_snapshot_records(evolving_graphs, membership_records, architecture_key, dataset_path):
    packaged_snapshots = []
    if dataset_path != "Data/DBLPdyn":
        observed_memberships = set(membership_records.values())
        membership_lookup = {membership_value: membership_index for membership_index, membership_value in enumerate(observed_memberships)}
    else:
        membership_lookup = {str(membership_index): membership_index - 1 for membership_index in range(1, 15)}
        assert len(membership_lookup) == 14
    for snapshot_index, snapshot_graph in enumerate(evolving_graphs):
        sparse_adjacency = sparse_library.coo_matrix(graph_library.adjacency_matrix(snapshot_graph))
        sparse_features = sparse_library.coo_matrix(array_engine.eye(sparse_adjacency.shape[0]), dtype=array_engine.int64)
        feature_coordinates, feature_values, feature_shape = convert_sparse_layout(sparse_features.tocoo())
        feature_indices = tensor_engine.LongTensor(feature_coordinates.T).to("cuda:0")
        unit_feature_values = tensor_engine.ones(feature_indices.size(1), device="cuda:0", dtype=tensor_engine.float32)
        sparse_features = tensor_engine.sparse_coo_tensor(feature_indices, unit_feature_values, tensor_engine.Size(feature_shape), device="cuda:0", dtype=tensor_engine.float32).coalesce()
        if dataset_path == "Data/DBLPdyn":
            snapshot_memberships = [membership_lookup[membership_records[snapshot_index][vertex_id]] for vertex_position, vertex_id in enumerate(snapshot_graph.nodes())]
        else:
            snapshot_memberships = [membership_lookup[membership_records[vertex_id]] for vertex_position, vertex_id in enumerate(snapshot_graph.nodes())]
        packaged_snapshots.append([sparse_adjacency, sparse_features, snapshot_memberships])
    return packaged_snapshots


def convert_sparse_layout(sparse_object):
    if not sparse_library.isspmatrix_coo(sparse_object):
        sparse_object = sparse_object.tocoo()
    coordinate_pairs = array_engine.vstack((sparse_object.row, sparse_object.col)).transpose()
    stored_values = sparse_object.data
    matrix_shape = sparse_object.shape
    return coordinate_pairs, stored_values, matrix_shape


def create_empty_full_vertex_graphs(evolving_graphs):
    complete_vertex_ids = set()
    for snapshot_graph in evolving_graphs:
        complete_vertex_ids.update(snapshot_graph.nodes)
    expanded_graphs = []
    for snapshot_graph in evolving_graphs:
        expanded_graph = graph_library.Graph()
        expanded_graph.add_nodes_from(complete_vertex_ids)
        expanded_graphs.append(expanded_graph)
    return expanded_graphs
