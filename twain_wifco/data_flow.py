from typing import List
from twain_wifco.interface import Ambient, Control, ModelOutput, Aggregate
from twain_wifco.statistics import AmbientStatistics
from twain_wifco.control_policy import DiscreteControlPolicy
from twain_wifco.plant_model import PlantModel
from twain_wifco.aggregation import Aggregation
from twain_wifco.metrics_accumulation import MetricsAccumulation

def validate_data_graph(ambient_statistics: AmbientStatistics,
                        control_policy: DiscreteControlPolicy,
                        plant_model: PlantModel,
                        aggregation: Aggregation,
                        metrics_accumulation: MetricsAccumulation):
    # Input for control policy
    control_policy.input_interface.validate_shapes(
        external_shapes={
            Ambient: ambient_statistics.output_interface.shapes[Ambient]},
        component_name=control_policy.component_name)
    
    # Input for plant model
    plant_model.input_interface.validate_shapes(
        external_shapes={
            Ambient: ambient_statistics.output_interface.shapes[Ambient],
            Control: control_policy.output_interface.shapes[Control]},
        component_name=plant_model.component_name)

    # Input for aggregation
    aggregation.input_interface.validate_shapes(
        external_shapes={
            ModelOutput: plant_model.output_interface.shapes[ModelOutput],
            Ambient: ambient_statistics.output_interface.shapes[Ambient],
            Control: control_policy.output_interface.shapes[Control]},
        component_name=aggregation.component_name)

    # Input for metrics accumulation
    metrics_accumulation.input_interface.validate_shapes(
        external_shapes={
            Aggregate: aggregation.output_interface.shapes[Aggregate]
        },
        component_name=metrics_accumulation.component_name)
    