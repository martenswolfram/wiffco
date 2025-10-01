from typing import Dict, Any, Set
from abc import abstractmethod
import numpy as np
from enum import Enum
from twain_wifco.interface import (
    Component,
    ComponentParams,
    Ambient,
    ModelOutput,
    Control,
    AggregatedOutput)

class Aggregation(Component):
    def __init__(self,
                 aggregation_name: str,
                 aggregation_params: ComponentParams):
        super().__init__(component_name=aggregation_name,
                         component_params=aggregation_params)

    def compute_aggregate(self,
                          model_output: Dict[ModelOutput, float],
                          ambient_condition: Dict[Ambient, float],
                          control_setpoints: Dict[Control, float]):
        
        self._validate_inputs(inputs=(model_output |
                                      ambient_condition |
                                      control_setpoints))

        return self._compute_aggregate(output_variables=model_output,
                                       ambient_condition=ambient_condition,
                                       control_setpoints=control_setpoints)
            
    @abstractmethod
    def _compute_aggregate(self,
                           output_variables: Dict[ModelOutput, float],
                           ambient_condition: Dict[Ambient, float],
                           control_setpoints: Dict[Control, float]):
        pass

class AggregationType(Enum):
    SIMPLE_PRODUCTS = "simple_products"

class ProductAggregateMapping:
    def __init__(self,
                 from_model: Set[ModelOutput],
                 from_ambient: Set[Ambient],
                 from_control: Set[Control]):
        self.from_model = from_model
        self.from_ambient = from_ambient
        self.from_control = from_control

class SimpleProductParams(ComponentParams):
    def __init__(self,
                 aggregate_mappings: Dict[AggregatedOutput, ProductAggregateMapping]):
        self.aggregate_mappings = aggregate_mappings
        
    def input_variables(self):
        required_model_output = set().union(*[mapping.from_model for mapping in self.aggregate_mappings.values()])
        required_ambient = set().union(*[mapping.from_ambient for mapping in self.aggregate_mappings.values()])
        required_control = set().union(*[mapping.from_control for mapping in self.aggregate_mappings.values()])
        return set().union(required_model_output,
                           required_ambient,
                           required_control)
    
    def output_variables(self):
        return set(self.aggregate_mappings.keys())

def simple_product_params_from_dict(param_dict: Dict[str, Dict | Any]):
    aggregate_mappings = {}
    for out_var, product_mapping in param_dict["aggregate_mappings"].items():
         aggregate_mappings[AggregatedOutput(out_var)] = ProductAggregateMapping(
              from_model=set([ModelOutput(out_var) for out_var in product_mapping["from_model"]]),
              from_ambient=set([Ambient(ambient_var) for ambient_var in product_mapping["from_ambient"]]),
              from_control=set([Control(ctrl_var) for ctrl_var in product_mapping["from_control"]])
         )
    return SimpleProductParams(aggregate_mappings=aggregate_mappings)

class SimpleProduct(Aggregation):
    def __init__(self,
                 aggregation_name: str,
                 aggregation_params: SimpleProductParams):
        super().__init__(aggregation_name=aggregation_name,
                         aggregation_params=aggregation_params)
        self.aggregate_mappings = aggregation_params.aggregate_mappings

    def _compute_aggregate(self,
                           output_variables: Dict[ModelOutput, float],
                           ambient_condition: Dict[Ambient, float],
                           control_setpoints: Dict[Control, float]):
    
        aggregated_output = {}
        for out_var, mapping in self.aggregate_mappings.items():
            aggregated_output[out_var] = \
                np.prod([output_variables[out_var] for out_var in mapping.from_model]) * \
                np.prod([ambient_condition[ambient_var] for ambient_var in mapping.from_ambient]) * \
                np.prod([control_setpoints[ctrl_var] for ctrl_var in mapping.from_control])
        return aggregated_output