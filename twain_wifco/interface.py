from typing import (
    List,
    Set,
    Tuple,
    TypeVar,
    Dict,
    Any,
    Generic)
import numpy as np
from dataclasses import dataclass
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

# Union of data types
DataVariable = Ambient | Control | ModelOutput | Aggregated | AccumulatedMetric

# Data-type TypeVar
DataType = TypeVar("DataType",
                   Ambient,
                   Control,
                   ModelOutput,
                   Aggregated,
                   AccumulatedMetric)

# Allow statistical distributions only over Ambient condition and Aggregated variables
StatisticalType = TypeVar("StatisticalType",
                          Ambient,
                          Aggregated)

# Allow constraint variables to be only Control, Aggregated or AccumulatedMetric
ConstraintType = TypeVar("ConstraintType",
                         Control,
                         Aggregated,
                         AccumulatedMetric)

def get_default_value(data_var: DataVariable,
                      shape: Tuple[int]) -> np.ndarray:
    # TODO: implement in DataVariable class 
    if data_var == Control.POWER_REGULATION:
        return np.ones(shape=shape)
    elif data_var == Control.YAW_STEERING:
        return np.zeros(shape=shape)
    else:
        raise ValueError(f"No default value for data variable '{data_var}'.") 

def get_abs_tol(data_var: DataVariable) -> float:
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


@dataclass
class DataCollection(Generic[DataType]):
    data: dict[DataType, np.ndarray]
    order: list[DataType] | None = None
    
    def keys(self):
        return self.data.keys()
    
    def __post_init__(self):
        if self.order is None:
            self.order = list(self.data.keys())

    def __getitem__(self, key: DataType) -> np.ndarray:
        return self.data[key]
    
    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.data[k].ravel() for k in self.order])

    def from_vector(self, vector: np.ndarray):
        i = 0
        for k in self.order:
            n = self.data[k].size
            self.data[k] = vector[i:i+n].reshape(self.data[k].shape)
            i += n

    def shapes(self):
        return {}
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataCollection):
            return False
        if set(self.data.keys()) != set(other.data.keys()):
            return False
        for k in self.data.keys():
            if not np.array_equal(self.data[k], other.data[k]):
                return False
        return True

@dataclass
class DataPoint(DataCollection[DataType]):
    def shapes(self):
        return {k: v.shape for k, v in self.data.items()}

@dataclass
class DataTable(DataCollection[DataType]):
    def __len__(self):
        first_key = self.order[0]
        return self.data[first_key].shape[0]

    def shapes(self):
        return {k: v.shape[1:] for k, v in self.data.items()}
    
    def to_matrix(self) -> np.ndarray:
        n_points = len(self)
        flattened = [self.data[k].reshape(n_points, -1) for k in self.order]
        return np.concatenate(flattened, axis=1)
    
    def get_point(self, idx: int) -> DataPoint[DataType]:
        return DataPoint({
            k: self.data[k][idx] for k in self.order
        }, self.order)

class Interface:
    def __init__(self,
                 ambient_shapes: Dict[Ambient, Tuple[int, ...]] = {},
                 control_shapes: Dict[Control, Tuple[int, ...]] = {},
                 model_output_shapes: Dict[Ambient, Tuple[int, ...]] = {},
                 aggregated_shapes: Dict[Ambient, Tuple[int, ...]] = {},
                 accumulated_metric_shapes: Dict[Ambient, Tuple[int, ...]] = {}):
        self.ambient_shapes = ambient_shapes
        self.control_shapes = control_shapes
        self.model_output_shapes = model_output_shapes
        self.aggregated_shapes = aggregated_shapes
        self.accumulated_metric_shapes = accumulated_metric_shapes

class ComponentParams(ABC):

    def __init__(self):
        pass
        
    @abstractmethod
    def input_interface(self) -> Interface:
        pass
    
    @abstractmethod
    def output_interface(self) -> Interface:
        pass

class Component(ABC):

    def __init__(self,
                 component_name: str,
                 component_params: ComponentParams):
        self.component_name = component_name
        self.input_interface = component_params.input_interface()
        self.output_interface = component_params.output_interface()

    def validate_single_input(
            self,
            input: Dict[DataType, Tuple[int, ...]],
            interface_shapes: Dict[DataType, Tuple[int, ...]]):
        missing_vars = interface_shapes.keys() - input.keys()
            
        if len(missing_vars):
            msg = (f"Insufficient input variables for component '{self.component_name}'. "
                f"Missing variable(s): {missing_vars}")
            raise ValueError(msg)
        for in_var, in_shape in interface_shapes.items():
            if in_shape is None:
                continue
            if not (in_shape == input[in_var]):
                msg = (f"Input variable dimensions mismatch for component '{self.component_name}'. "
                    f"Mismatch variable: {in_var}")
                raise ValueError(msg)
        
    def validate_inputs(self,
                        ambient: DataCollection[Ambient] = DataCollection({}),
                        control: DataCollection[Control] = DataCollection({}),
                        model_output: DataCollection[ModelOutput] = DataCollection({}),
                        aggregated: DataCollection[Aggregated] = DataCollection({}),
                        accumulated_metric: DataCollection[AccumulatedMetric] = DataCollection({})):
        
        self.validate_single_input(ambient.shapes(), self.input_interface.ambient_shapes)
        self.validate_single_input(control.shapes(), self.input_interface.control_shapes)
        self.validate_single_input(model_output.shapes(), self.input_interface.model_output_shapes)
        self.validate_single_input(aggregated.shapes(), self.input_interface.aggregated_shapes)
        self.validate_single_input(accumulated_metric.shapes(), self.input_interface.accumulated_metric_shapes)
            
def validate_data_flow(components: List[Component]):
    current_out = {}
    while len(components):
        current_component = components.pop(0)
        current_in = current_component.input_shape
        if not (current_in <= current_out):
            msg = (f"Insufficient input variables for component "
                   f"'{current_component.component_name}'. "
                   f"Missing variable(s): {current_in - current_out}")
            raise ValueError(msg)
        current_out = current_component.output_shape

def retrieve_single_key_str(input_dict: Dict[str, Any], key_strs: Set[str]):
    single_key = key_strs.intersection(input_dict.keys())
    if not len(single_key) == 1:
        raise ValueError(f"Exactly one element of {key_strs} must appear as key in {input_dict}.")
    return list(single_key)[0]
