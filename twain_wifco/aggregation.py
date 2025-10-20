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
    Aggregated,
    DataPoint,
    Interface)

class Aggregation(Component):
    def __init__(self,
                 aggregation_name: str,
                 aggregation_params: ComponentParams):
        super().__init__(component_name=aggregation_name,
                         component_params=aggregation_params)

    def compute_aggregate(self,
                          model_output:      DataPoint[ModelOutput],
                          ambient_condition: DataPoint[Ambient],
                          control_setpoints: DataPoint[Control]) -> DataPoint[Aggregated]:
        
        self.validate_inputs(input_data={
            ModelOutput: model_output,
            Ambient: ambient_condition,
            Control: control_setpoints}
            )

        return self._compute_aggregate(model_output=model_output,
                                       ambient_condition=ambient_condition,
                                       control_setpoints=control_setpoints)
            
    @abstractmethod
    def _compute_aggregate(self,
                           model_output:      DataPoint[ModelOutput],
                           ambient_condition: DataPoint[Ambient],
                           control_setpoints: DataPoint[Control]):
        pass

class AggregationType(Enum):
    SIMPLE_PRODUCT = "simple_product"

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
                 aggregate_mappings: Dict[Aggregated, ProductAggregateMapping]):
        self.aggregate_mappings = aggregate_mappings
        
    def input_interface(self) -> Interface:
        required_model_output_shapes = {}
        required_ambient_shapes = {}
        required_control_shapes = {}
        for mapping in self.aggregate_mappings.values():
            required_model_output_shapes |= {in_var: None for in_var in mapping.from_model}
            required_ambient_shapes |= {in_var: None for in_var in mapping.from_ambient}
            required_control_shapes |= {in_var: None for in_var in mapping.from_control}
        
        return Interface(all_data_type_shapes={
            ModelOutput: required_model_output_shapes,
            Ambient: required_ambient_shapes,
            Control: required_control_shapes}
            )

    def output_interface(self) -> Interface:
        return Interface(all_data_type_shapes={
            Aggregated: {
                aggr_var: (1,) for aggr_var in self.aggregate_mappings.keys()
                }})

def simple_product_params_from_dict(param_dict: Dict[str, Dict | Any]):
    aggregate_mappings = {}
    for out_var, product_mapping in param_dict["aggregate_mappings"].items():
         aggregate_mappings[Aggregated(out_var)] = ProductAggregateMapping(
              from_model=  set([ModelOutput(out_var) for \
                out_var     in product_mapping["from_model"]]),
              from_ambient=set([Ambient(ambient_var) for \
                ambient_var in product_mapping["from_ambient"]]),
              from_control=set([Control(ctrl_var)    for \
                ctrl_var    in product_mapping["from_control"]])
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
                           model_output:      DataPoint[ModelOutput],
                           ambient_condition: DataPoint[Ambient],
                           control_setpoints: DataPoint[Control]):
    
        aggregated_output = {}
        for out_var, mapping in self.aggregate_mappings.items():
            res = np.prod([np.prod(model_output[out_var]) for out_var in mapping.from_model]) * \
                np.prod([np.prod(ambient_condition[ambient_var]) for ambient_var in mapping.from_ambient]) * \
                np.prod([np.prod(control_setpoints[ctrl_var]) for ctrl_var in mapping.from_control])
            aggregated_output[out_var] = np.array([res])
        return DataPoint(data=aggregated_output)
