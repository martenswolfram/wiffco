from typing import Dict, Any, Set
from abc import ABC, abstractmethod
from enum import Enum

class Component(Enum):
    WIND_FARM_MODEL = "wind_farm_model"
    OUTPUT_AGGREGATOR = "output_aggregator"
    CONTROL_POLICY = "control_policy"
    
class AmbientVariable(Enum):
    WIND_SPEED = "wind_speed"
    WIND_DIRECTION = "wind_direction"
    ELECTRICITY_PRICE = "electricity_price"

class ControlVariable(Enum):
    POWER_REGULATION = "power_regulation"
    YAW_STEERING = "yaw_steering"

class OutputVariable(Enum):
    ELECTRICAL_POWER = "electrical_power"
    REVENUE_RATE = "revenue_rate"
    DAMAGE_RATE = "damage_rate"

class InterfaceInputs:
    def __init__(self,
                 ambient_variables: Set[AmbientVariable] = set([]),
                 control_inputs: Set[ControlVariable] = set([]),
                 output_variables: Set[OutputVariable] = set([])):
        self.ambient_variables = ambient_variables
        self.control_inputs = control_inputs
        self.output_variables = output_variables

class InterfaceOutputs:
    def __init__(self,
                 ambient_variables: Set[AmbientVariable] = set([]),
                 control_inputs: Set[ControlVariable] = set([]),
                 output_variables: Set[OutputVariable] = set([])):
        self.ambient_variables = ambient_variables
        self.control_inputs = control_inputs
        self.output_variables = output_variables

class Interface:
    def __init__(self,
                 component: Component,
                 name: str,
                 inputs: InterfaceInputs = InterfaceInputs(),
                 outputs: InterfaceOutputs = InterfaceOutputs()):
        match component:
            case Component.WIND_FARM_MODEL:
                self.name = "Model '{}'".format(name)
            case Component.OUTPUT_AGGREGATOR:
                self.name = "Output Aggregator '{}'".format(name)
            case Component.CONTROL_POLICY:
                self.name = "Control Policy '{}'".format(name)
        self.inputs = inputs
        self.outputs = outputs

    def validate_inputs(self,
                        ambient_condition: Dict[AmbientVariable, Any] = {},
                        control_input: Dict[ControlVariable, Any] = {},
                        output_variables: Dict[OutputVariable, Any] = {}):
        if not (self.inputs.ambient_variables <= ambient_condition.keys()):
            raise ValueError("Insufficient Ambient Condition as input for {}.".format(self.name))
        if not (self.inputs.control_inputs <= control_input.keys()):
            raise ValueError("Insufficient Control Input for {}.".format(self.name))
        if not (self.inputs.output_variables <= output_variables.keys()):
            raise ValueError("Insufficient Output Variable as input for {}.".format(self.name))
        
class ComponentParams(ABC):
    def __init__(self,
                 name: str):
        self.name = name

    @abstractmethod
    def interface(self) -> Interface:
        pass

