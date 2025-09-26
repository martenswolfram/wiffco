from typing import Dict, Any, Set
from abc import ABC, abstractmethod
import numpy as np
from enum import Enum
from twain_wifco.interface import (
    InterfaceVariables,
    AmbientVariable,
    OutputVariable,
    ComponentType,
    ComponentParams)

class OutputAggregation(ABC):
    def __init__(self,
                 params: ComponentParams):
        self.interface = params.interface()

    def compute_aggregate(self,
                          output_variables: Dict[OutputVariable, float],
                          ambient_condition: Dict[AmbientVariable, float]):
        
        self.interface.validate_inputs(output_variables=output_variables.keys(),
                                       ambient_condition=ambient_condition.keys())

        return self._compute_aggregate(output_variables=output_variables,
                                       ambient_condition=ambient_condition)
        
    @abstractmethod
    def _compute_aggregate(self,
                          output_variables: Dict[OutputVariable, float],
                          ambient_condition: Dict[AmbientVariable, float]):
        pass

class AggregationType(Enum):
    SIMPLE_PRODUCT = "simple_product"

class ProductAggregateMapping:
    def __init__(self,
                 from_model: Set[OutputVariable],
                 from_ambient: Set[AmbientVariable]):
        self.from_model = from_model
        self.from_ambient = from_ambient

class SimpleProductParams(ComponentParams):
    def __init__(self,
                 component_name: str,
                 aggregate_mappings: Dict[OutputVariable, ProductAggregateMapping]):
        super().__init__(component_type=ComponentType.OUTPUT_AGGREGATOR,
                         component_name=component_name)
        self.aggregate_mappings = aggregate_mappings

    def _interface(self):
        # Inputs
        output_variables = set().union(*[mapping.from_model for mapping in self.aggregate_mappings.values()])
        ambient_variables = set().union(*[mapping.from_ambient for mapping in self.aggregate_mappings.values()])
        inputs = InterfaceVariables(output_variables=output_variables,
                                 ambient_variables=ambient_variables)
        outputs = InterfaceVariables(output_variables=set(self.aggregate_mappings.keys()))
        return inputs, outputs

def simple_product_params_from_dict(name: str,
                                    param_dict: Dict[str, Dict | Any]):
    aggregate_mappings = {}
    for out_var, product_mapping in param_dict["aggregate_mappings"].items():
         aggregate_mappings[OutputVariable(out_var)] = ProductAggregateMapping(
              from_model=set([OutputVariable(out_var) for out_var in product_mapping["from_model"]]),
              from_ambient=set([AmbientVariable(ambient_var) for ambient_var in product_mapping["from_ambient"]])
         )
    return SimpleProductParams(component_name=name,
                               aggregate_mappings=aggregate_mappings)

class SimpleProduct(OutputAggregation):
    def __init__(self,
                 params: SimpleProductParams):
        super().__init__(params=params)
        self.aggregate_mappings = params.aggregate_mappings
        
    def _compute_aggregate(self,
                           output_variables: Dict[OutputVariable, float],
                           ambient_condition: Dict[AmbientVariable, float]):    
        aggregated_output = {}
        for out_var, mapping in self.aggregate_mappings.items():
            aggregated_output[out_var] = \
                np.prod([output_variables[out_var] for out_var in mapping.from_model]) * \
                np.prod([ambient_condition[ambient_var] for ambient_var in mapping.from_ambient])
        return aggregated_output