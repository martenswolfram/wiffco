from typing import Dict, List, Any
from abc import ABC, abstractmethod
from enum import Enum
from twain_wiffco.model_output import OutputVariable
from twain_wiffco.ambient_conditions import AmbientVariable

class AggregationType(Enum):
    SIMPLE_PRODUCT = "simple_product"

class OutputAggregation(ABC):
    
    def __init__(self,
                 name: str):
        self.name = name

    def compute_aggregate(self,
                          model_outputs: Dict[OutputVariable, float],
                          ambient_condition: Dict[AmbientVariable, float]):
        
        self._validate_input(model_outputs=model_outputs,
                              ambient_condition=ambient_condition)

        return self._compute_aggregate(model_outputs=model_outputs,
                                       ambient_conditions=ambient_condition)
        

    @abstractmethod
    def _validate_input(self,
                        model_outputs: Dict[OutputVariable, float],
                        ambient_condition: Dict[AmbientVariable, float]):
        pass        

    @abstractmethod
    def _compute_aggregate(self,
                          model_outputs: Dict[OutputVariable, float],
                          ambient_conditions: Dict[AmbientVariable, float]):
        pass

class SimpleProductParams:
    def __init__(self,
                 param_dict: Dict[str, Any]):
        from_model_list = [OutputVariable(out_var) for out_var in param_dict["from_model"]]
        self.from_model = set(from_model_list)
        from_context_list = [AmbientVariable(ambient_var) for ambient_var in param_dict["from_context"]]
        self.from_context = set(from_context_list)
        self.single_output = OutputVariable(param_dict["single_output"])

class SimpleProduct(OutputAggregation):
    def __init__(self,
                 name: str,
                 params: SimpleProductParams):
        super().__init__(name=name)
        self.params = params

    def _validate_input(self,
                        model_outputs: Dict[OutputVariable, float],
                        ambient_condition: Dict[AmbientVariable, float]):
        if not (self.params.from_model <= model_outputs.keys()):
            raise ValueError("Insufficient model outputs provided for output aggregation '{}'.".format(self.name))
        if not (self.params.from_context <= ambient_condition.keys()):
            raise ValueError("Insufficient ambient conditions provided for output aggregation '{}'.".format(self.name))

    def _compute_aggregate(self,
                           model_outputs: Dict[OutputVariable, float],
                           ambient_conditions: Dict[AmbientVariable, float]):
        aggregate = 1
        for model_output in self.params.from_model:
            aggregate *= model_outputs[model_output]
        for ambient_variable in self.params.from_context:
            aggregate *= ambient_conditions[ambient_variable]
            
        return {self.params.single_output: aggregate}