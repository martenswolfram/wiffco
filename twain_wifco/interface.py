from typing import Dict, List, Union
from abc import ABC, abstractmethod
from enum import Enum

class Ambient(Enum):
    WIND_SPEED = "wind_speed"
    WIND_DIRECTION = "wind_direction"
    ELECTRICITY_PRICE = "electricity_price"

class Control(Enum):
    POWER_REGULATION = "power_regulation"
    YAW_STEERING = "yaw_steering"

class ModelOutput(Enum):
    ELECTRICAL_POWER = "electrical_power"
    DAMAGE_RATE = "damage_rate"

class AggregatedOutput(Enum):
    REVENUE_RATE = "revenue_rate"
    
class AccumulatedMetric(Enum):
    REVENUE = "revenue"
    ACCRUED_DAMAGE = "accrued_damage"

DataVariable = Union[Ambient,
                      Control,
                      ModelOutput,
                      AggregatedOutput,
                      AccumulatedMetric]

class ComponentParams(ABC):

    def __init__(self):
        pass
        
    @abstractmethod
    def input_variables(self):
        pass
    
    @abstractmethod
    def output_variables(self):
        pass

class Component(ABC):

    def __init__(self,
                 component_name: str,
                 component_params: ComponentParams):
        self.component_name = component_name
        self.input_variables = component_params.input_variables()
        self.output_variables = component_params.output_variables()

    def _validate_inputs(self, inputs: Dict[DataVariable, float]):
        if not (self.input_variables <= inputs.keys()):
            msg = (f"Insufficient input variables for component '{self.component_name}'. "
                   f"Missing variable(s): {[var.value for var in (self.input_variables - inputs.keys())]}")
            raise ValueError(msg)

def validate_component_disambiguation(components: List[Component]):
    # This one is not necessarily a problem, but let's keep it clean:
    component_names = [component.component_name for component in components]
    if len(component_names) != len(set(component_names)):
        raise ValueError("Ambiguous component names detected for multiple components.")
    # Avoid that two components provide a result for the same output variable
    all_outputs = [out_var for model in components for out_var in model.output_variables]
    if len(all_outputs) != len(set(all_outputs)):
        raise ValueError("Ambiguous output variables detected for multiple components.")
