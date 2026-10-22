class RunProfile(dict):
    def __init__(instance, group_total, dataset_key, architecture_key) -> None:
        instance.latent_width = 50
        instance.group_total = group_total
        instance.training_rounds = 701
        instance.step_size = 0.001
        instance.temporal_scale = 1
        instance.diagram_capacity = 20
        instance.dataset_key = dataset_key
        instance.architecture_key = architecture_key
        instance.delayed_stage = 500
        instance.partition_scale = 0.05
        instance.propagation_steps = 8
        instance.propagation_scale = 0.10
