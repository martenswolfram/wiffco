from typing import Dict, Any, Set
from abc import ABC, abstractmethod
from enum import Enum
from twain_wifco.interface import (
    InterfaceInputs,
    InterfaceOutputs,
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
                 component_name: str,
                 from_model: Set[OutputVariable],
                 from_context: Set[AmbientVariable],
                 single_output: OutputVariable):
        super().__init__(component_type=ComponentType.OUTPUT_AGGREGATOR,
                         component_name=component_name)
        self.from_model = from_model
        self.from_context = from_context
        self.single_output = single_output

    def _interface(self):
        inputs = InterfaceInputs(output_variables=self.from_model,
                                 ambient_variables=self.from_context)
        outputs = InterfaceOutputs(output_variables=set([self.single_output]))
        return inputs, outputs

def simple_product_params_from_dict(name: str,
                                    param_dict: Dict[str, Any]):
    from_model = set([OutputVariable(out_var) for out_var in param_dict["from_model"]])
    from_context = set([AmbientVariable(ambient_var) for ambient_var in param_dict["from_context"]])
    single_output = OutputVariable(param_dict["single_output"])
    return SimpleProductParams(component_name=name,
                               from_model=from_model,
                               from_context=from_context,
                               single_output=single_output)

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