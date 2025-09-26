from typing import Dict, Any, Set, Tuple
from abc import ABC, abstractmethod
from enum import Enum

class ComponentType(Enum):
    STATISTICS = "statistics"
    WIND_FARM_MODEL = "wind_farm_model"
    OUTPUT_AGGREGATOR = "output_aggregator"
    CONTROL_POLICY = "control_policy"
    OUTPUT_ACCUMULATOR = "output_accumulator"

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

class AccumulatedMetric(Enum):
    REVENUE = "revenue"
    ACCRUED_DAMAGE = "accrued_damage"

class InterfaceVariables:
    def __init__(self,
                 ambient_variables: Set[AmbientVariable] = set([]),
                 control_inputs: Set[ControlVariable] = set([]),
                 output_variables: Set[OutputVariable] = set([]),
                 accumulated_metrics: Set[AccumulatedMetric] = set([])):
        self.ambient_variables = ambient_variables
        self.control_inputs = control_inputs
        self.output_variables = output_variables
        self.accumulated_metrics = accumulated_metrics

class Interface:
    def __init__(self,
                 component_type: ComponentType,
                 component_name: str,
                 inputs: InterfaceVariables = InterfaceVariables(),
                 outputs: InterfaceVariables = InterfaceVariables()):
        match component_type:
            case ComponentType.STATISTICS:
                self.name = "Statistics '{}'".format(component_name)
            case ComponentType.WIND_FARM_MODEL:
                self.name = "Model '{}'".format(component_name)
            case ComponentType.CONTROL_POLICY:
                self.name = "Control Policy '{}'".format(component_name)
            case ComponentType.OUTPUT_AGGREGATOR:
                self.name = "Output Aggregator '{}'".format(component_name)
            case ComponentType.OUTPUT_ACCUMULATOR:
                self.name = "Output Accumulator '{}'".format(component_name)
        self.inputs = inputs
        self.outputs = outputs

    def validate_inputs(self,
                        ambient_condition: Set[AmbientVariable] = set([]),
                        control_input: Set[ControlVariable] = set([]),
                        output_variables: Set[OutputVariable] = set([])):
        if not (self.inputs.ambient_variables <= ambient_condition):
            raise ValueError("Insufficient Ambient Condition as input for {}.".format(self.name))
        if not (self.inputs.control_inputs <= control_input):
            raise ValueError("Insufficient Control Input for {}.".format(self.name))
        if not (self.inputs.output_variables <= output_variables):
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
    def _interface(self) -> Tuple[InterfaceVariables, InterfaceVariables]:
        pass

