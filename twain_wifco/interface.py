from typing import Dict, Any, Set, Tuple
from abc import ABC, abstractmethod
from enum import Enum

class ComponentType(Enum):
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
                 component_type: ComponentType,
                 component_name: str,
                 inputs: InterfaceInputs = InterfaceInputs(),
                 outputs: InterfaceOutputs = InterfaceOutputs()):
        match component_type:
            case ComponentType.WIND_FARM_MODEL:
                self.name = "Model '{}'".format(component_name)
            case ComponentType.OUTPUT_AGGREGATOR:
                self.name = "Output Aggregator '{}'".format(component_name)
            case ComponentType.CONTROL_POLICY:
                self.name = "Control Policy '{}'".format(component_name)
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
                 component_type: ComponentType,
                 component_name: str):
        self.component_type = component_type
        self.component_name = component_name

    def interface(self) -> Interface:
        inputs, outputs = self._interface()
        return Interface(component_type=self.component_type,
                         component_name=self.component_name,
                         inputs=inputs,
                         outputs=outputs)

    @abstractmethod
    def _interface(self) -> Tuple[InterfaceInputs, InterfaceOutputs]:
        pass

