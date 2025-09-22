from typing import Dict, Any
from abc import ABC, abstractmethod
from enum import Enum
from twain_wifco.interface import (
    Interface,
    AmbientVariable,
    OutputVariable,
    Component,
    ComponentParams)

class OutputAggregation(ABC):
    def __init__(self,
                 params: ComponentParams):
        self.interface = params.interface()

    def compute_aggregate(self,
                          output_variables: Dict[OutputVariable, float],
                          ambient_condition: Dict[AmbientVariable, float]):
        
        self.interface.validate_inputs(output_variables=output_variables,
                                       ambient_condition=ambient_condition)

        return self._compute_aggregate(output_variables=output_variables,
                                       ambient_condition=ambient_condition)
        
    @abstractmethod
    def _compute_aggregate(self,
                          output_variables: Dict[OutputVariable, float],
                          ambient_condition: Dict[AmbientVariable, float]):
        pass

class AggregationType(Enum):
    SIMPLE_PRODUCT = "simple_product"

class SimpleProductParams(ComponentParams):
    def __init__(self,
                 name: str,
                 param_dict: Dict[str, Any]):
        super().__init__(name=name)
        from_model_list = [OutputVariable(out_var) for out_var in param_dict["from_model"]]
        self.from_model = set(from_model_list)
        from_context_list = [AmbientVariable(ambient_var) for ambient_var in param_dict["from_context"]]
        self.from_context = set(from_context_list)
        self.single_output = OutputVariable(param_dict["single_output"])

    def interface(self):
        return Interface(component=Component.OUTPUT_AGGREGATOR,
                         name=self.name,
                         output_variables=self.from_model,
                         ambient_variables=self.from_context)

class SimpleProduct(OutputAggregation):
    def __init__(self,
                 params: SimpleProductParams):
        super().__init__(params=params)
        self.from_model = params.from_model
        self.from_context = params.from_context
        self.single_output = params.single_output
        
    def _compute_aggregate(self,
                           output_variables: Dict[OutputVariable, float],
                           ambient_condition: Dict[AmbientVariable, float]):
        aggregate = 1
        for model_output in self.from_model:
            aggregate *= output_variables[model_output]
        for ambient_variable in self.from_context:
            aggregate *= ambient_condition[ambient_variable]
            
        return {self.single_output: aggregate}