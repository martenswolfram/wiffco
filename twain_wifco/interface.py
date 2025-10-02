from typing import Dict, List, Union, Set, TypeVar
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

class Aggregated(Enum):
    REVENUE_RATE = "revenue_rate"
    DAMAGE_RATE = "damage_rate"
    
class AccumulatedMetric(Enum):
    REVENUE = "revenue"
    ACCRUED_DAMAGE = "accrued_damage"

DataVariable = Union[Ambient,
                      Control,
                      ModelOutput,
                      Aggregated,
                      AccumulatedMetric]

# Allow statistical distributions only over Ambient condition and Aggregated variables
Statistical = TypeVar('Statistical', Ambient, Aggregated)

class ComponentParams(ABC):

    def __init__(self):
        pass
        
    @abstractmethod
    def input_variables(self) -> Set[DataVariable]:
        pass
    
    @abstractmethod
    def output_variables(self) -> Set[DataVariable]:
        pass

class Component(ABC):

    def __init__(self,
                 component_name: str,
                 component_params: ComponentParams):
        self.component_name = component_name
        self.input_variables = component_params.input_variables()
        self.output_variables = component_params.output_variables()

    def _validate_inputs(self, inputs: Set[DataVariable]):
        if not (self.input_variables <= inputs):
            msg = (f"Insufficient input variables for component '{self.component_name}'. "
                   f"Missing variable(s): {[var.value for var in (self.input_variables - inputs)]}")
            raise ValueError(msg)

def validate_data_flow(components: List[Component]):
    current_out = set()
    while len(components):
        current_component = components.pop(0)
        current_in = current_component.input_variables
        if not (current_in <= current_out):
            msg = (f"Insufficient input variables for component "
                   f"'{current_component.component_name}'. "
                   f"Missing variable(s): {current_in - current_out}")
            raise ValueError(msg)
        current_out = current_component.output_variables

    