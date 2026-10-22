import os as system_engine
system_engine.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pickle as serialization_engine
import torch as tensor_engine
from tensorflow.keras.utils import to_categorical as category_expander

from dynasphere_model import DynaSphere
from graph_data import aggregate_group_graph
from optimization_flow import run_initial_optimization, run_temporal_optimization
from persistence_math import PersistenceEncoder
from records import read_snapshots
from runtime import RunProfile
from sparse_math import normalize_adjacency


def launch_training(dataset_key, architecture_key):
    execution_device = "cuda"
    snapshot_records, group_total = read_snapshots(dataset_key=dataset_key, architecture_key=architecture_key)
    print(len(snapshot_records))
    run_profile = RunProfile(group_total, dataset_key, architecture_key)
    zero_dimensional_encoder = PersistenceEncoder(homology_dimension=0, diagram_capacity=run_profile.diagram_capacity)
    one_dimensional_encoder = PersistenceEncoder(homology_dimension=1, diagram_capacity=run_profile.diagram_capacity)
    initial_outputs = []
    temporal_outputs = []
    snapshot_models = {}
    for snapshot_index, snapshot_record in enumerate(snapshot_records):
        sparse_adjacency, sparse_features, snapshot_memberships = snapshot_record
        normalized_layout = normalize_adjacency(sparse_adjacency)
        normalized_indices = tensor_engine.LongTensor(normalized_layout[0].T).to(execution_device)
        normalized_values = tensor_engine.FloatTensor(normalized_layout[1]).to(execution_device)
        normalized_adjacency = tensor_engine.sparse_coo_tensor(normalized_indices, normalized_values, tensor_engine.Size(normalized_layout[2]), device=execution_device, dtype=tensor_engine.float32).coalesce()
        snapshot_model = DynaSphere(normalized_adjacency, sparse_features.size(1), run_profile).to(execution_device)
        snapshot_models[snapshot_index] = snapshot_model
        run_initial_optimization(architecture_key, snapshot_model, sparse_features, sparse_adjacency, run_profile, str(snapshot_index))
        with tensor_engine.no_grad():
            unused_reconstruction, embedded_features, soft_assignments = snapshot_model(sparse_features, True)
            initial_outputs.append([embedded_features.cpu().detach().numpy(), soft_assignments.cpu().detach().numpy(), sparse_adjacency, snapshot_memberships])
            grouped_adjacency = aggregate_group_graph(soft_assignments, sparse_adjacency)
            zero_diagram = zero_dimensional_encoder(grouped_adjacency)
            one_diagram = one_dimensional_encoder(grouped_adjacency)
            snapshot_models[snapshot_index].cached_diagrams = [zero_diagram, one_diagram]
            temporal_outputs.append(None)
    for snapshot_index in range(len(snapshot_records)):
        snapshot_model = snapshot_models[snapshot_index]
        sparse_adjacency, sparse_features, snapshot_memberships = snapshot_records[snapshot_index]
        if snapshot_index == 0:
            neighboring_diagrams = [None, snapshot_models[snapshot_index + 1].cached_diagrams]
        elif snapshot_index == len(snapshot_records) - 1:
            neighboring_diagrams = [snapshot_models[snapshot_index - 1].cached_diagrams, None]
        else:
            neighboring_diagrams = [snapshot_models[snapshot_index - 1].cached_diagrams, snapshot_models[snapshot_index + 1].cached_diagrams]
        run_temporal_optimization(architecture_key, snapshot_model, neighboring_diagrams, sparse_adjacency, sparse_features, run_profile, str(snapshot_index))
        with tensor_engine.no_grad():
            unused_reconstruction, embedded_features, soft_assignments = snapshot_model(sparse_features, True)
            grouped_adjacency = aggregate_group_graph(soft_assignments, sparse_adjacency)
            zero_diagram = zero_dimensional_encoder(grouped_adjacency)
            one_diagram = one_dimensional_encoder(grouped_adjacency)
            snapshot_models[snapshot_index].cached_diagrams = [zero_diagram, one_diagram]
            temporal_outputs[snapshot_index] = [embedded_features.cpu().detach().numpy(), soft_assignments.cpu().detach().numpy(), sparse_adjacency, snapshot_memberships]
    output_directory = system_engine.path.join("Data", dataset_key)
    system_engine.makedirs(output_directory, exist_ok=True)
    with open(system_engine.path.join(output_directory, "results_raw.pkl"), "wb") as raw_output_stream:
        serialization_engine.dump(initial_outputs, raw_output_stream, protocol=serialization_engine.HIGHEST_PROTOCOL)
    with open(system_engine.path.join(output_directory, "results_topo.pkl"), "wb") as temporal_output_stream:
        serialization_engine.dump(temporal_outputs, temporal_output_stream, protocol=serialization_engine.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    tensor_engine.manual_seed(42)
    selected_architecture = "DynaSphere"
    selected_dataset = "DBLPdyn"
    print(selected_dataset)
    launch_training(selected_dataset, architecture_key=selected_architecture)
