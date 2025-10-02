from typing import Dict, List, Union, Set
from abc import ABC, abstractmethod
from enum import Enum
from twain_wifco.interface import Component
from twain_wifco.statistics import Statistics
from twain_wifco.control_input import ControlPolicy
from twain_wifco.plant_model import PlantModel
from twain_wifco.aggregation import Aggregation
from twain_wifco.metrics_accumulation import MetricsAccumulation


def validate_inputs(sources: Set[Component],
                    target: Component):
    available_inputs_list = [out_var for source in sources for out_var in source.output_variables]
    available_inputs = set(available_inputs_list)
    if not len(available_inputs_list) == len(available_inputs):
        msg = (f"Ambiguous outputs detected in source components"
               f" {[source.component_name for source in sources]}.")
        raise ValueError(msg)
    
    if not (target.input_variables <= available_inputs):
        msg = (f"Insufficient inputs for target component {target.component_name} from source "
                f"components '{[source.component_name for source in sources]}'.")
        raise ValueError(msg)

def validate_data_graph(statistics: Statistics,
                        control_policy: ControlPolicy,
                        plant_model: PlantModel,
                        aggregation: Aggregation,
                        metrics_accumulation: MetricsAccumulation):
    # Input for control policy
    validate_inputs(sources=set([statistics]),
                    target=control_policy)
    # Input for plant model
    validate_inputs(sources=set([statistics, control_policy]),
                    target=plant_model)
    # Input for aggregation
    validate_inputs(sources=set([statistics, control_policy, plant_model]),
                    target=aggregation)
    # Input for metrics accumulation
    validate_inputs(sources=set([aggregation]),
                    target=metrics_accumulation)
    