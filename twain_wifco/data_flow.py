from typing import List
from twain_wifco.interface import Ambient, Control, ModelOutput, Aggregated
from twain_wifco.statistics import Statistics
from twain_wifco.control_policy import DiscreteControlPolicy
from twain_wifco.plant_model import PlantModel
from twain_wifco.aggregation import Aggregation
from twain_wifco.metrics_accumulation import MetricsAccumulation

def validate_data_graph(ambient_statistics: Statistics,
                        control_policy: DiscreteControlPolicy,
                        plant_model: PlantModel,
                        aggregation: Aggregation,
                        metrics_accumulation: MetricsAccumulation):
    # Input for control policy
    control_policy._input_interface.validate_shapes(
        external_shapes={
            Ambient: ambient_statistics._output_interface.shapes(Ambient)},
        component_name=control_policy._component_name)
    
    # Input for plant model
    plant_model._input_interface.validate_shapes(
        external_shapes={
            Ambient: ambient_statistics._output_interface.shapes(Ambient),
            Control: control_policy._output_interface.shapes(Control)},
        component_name=plant_model._component_name)

    # Input for aggregation
    aggregation._input_interface.validate_shapes(
        external_shapes={
            ModelOutput: plant_model._output_interface.shapes(ModelOutput),
            Ambient: ambient_statistics._output_interface.shapes(Ambient),
            Control: control_policy._output_interface.shapes(Control)},
        component_name=aggregation._component_name)

    # Input for metrics accumulation
    metrics_accumulation._input_interface.validate_shapes(
        external_shapes={
            Aggregated: aggregation._output_interface.shapes(Aggregated)
        },
        component_name=metrics_accumulation._component_name)
    