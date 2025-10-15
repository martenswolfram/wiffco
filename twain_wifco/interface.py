from typing import List, Union, Set, TypeVar, Dict, Any
import numpy as np
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

def get_default_value(data_var: DataVariable):
    # TODO: implement in DataVariable class 
    if data_var == Control.POWER_REGULATION:
        return 1.0
    elif data_var == Control.YAW_STEERING:
        return 0.0
    else:
        raise ValueError(f"No default value for data variable '{data_var}'.") 

def get_abs_tol(data_var: DataVariable):
    # TODO: implement in DataVariable class 
    if data_var == Ambient.WIND_DIRECTION:
        return 0.001
    elif data_var == Ambient.WIND_SPEED:
        return 0.001
    elif data_var == Ambient.ELECTRICITY_PRICE:
        return 0.001
    elif data_var == AccumulatedMetric.ACCRUED_DAMAGE:
        return 0.1
    else:
        raise ValueError(f"No abs tol value for data variable '{data_var}'.") 

# Allow statistical distributions only over Ambient condition and Aggregated variables
StatisticalVar = TypeVar('Statistical', Ambient, Aggregated)

# Allow constraint variables to be only Control, Aggregated or AccumulatedMetric
ConstraintVar = TypeVar('ConstraintVar',
                        Control,
                        Aggregated,
                        AccumulatedMetric)

DataType = TypeVar("DataType",
                   Ambient,
                   Control,
                   ModelOutput,
                   Aggregated,
                   AccumulatedMetric)

class ComponentParams(ABC):

    def __init__(self):
        pass
        
    @abstractmethod
    def input_variables(self) -> Dict[DataVariable, int]:
        pass
    
    @abstractmethod
    def output_variables(self) -> Dict[DataVariable, int]:
        pass

class Component(ABC):

    def __init__(self,
                 component_name: str,
                 component_params: ComponentParams):
        self.component_name = component_name
        self.input_variables = component_params.input_variables()
        self.output_variables = component_params.output_variables()

    def input_of_type(self, t: DataType):
        return set([var for var in self.input_variables if isinstance(var, t)])

    def _validate_inputs(self, inputs: Dict[DataVariable, np.ndarray]):
        if not (self.input_variables.keys() <= inputs.keys()):
            msg = (f"Insufficient input variables for component '{self.component_name}'. "
                   f"Missing variable(s): {[var.value for var in (self.input_variables.keys() - inputs.keys())]}")
            raise ValueError(msg)
        for in_var, in_dim in self.input_variables.items():
            if in_dim is None:
                continue
            if not (in_dim == len(inputs[in_var])):
                msg = (f"Input variable dimensions mismatch for component '{self.component_name}'. "
                    f"Mismatch variable: {in_var}")
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

def retrieve_single_key_str(input_dict: Dict[str, Any], key_strs: Set[str]):
    single_key = key_strs.intersection(input_dict.keys())
    if not len(single_key) == 1:
        raise ValueError(f"Exactly one element of {key_strs} must appear as key in {input_dict}.")
    return list(single_key)[0]
