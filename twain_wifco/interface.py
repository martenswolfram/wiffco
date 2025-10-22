from typing import (
    List,
    Set,
    Type,
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
    data_type: Type[DataType] | None = None # TODO: Fix data type handling
    order: list[DataType] | None = None
    
    def __post_init__(self):
        if self.order is None:
            self.order = list(self.data.keys())
        if len(self.order) > 0:
            self.data_type = type(self.order[0])
    
    def keys(self):
        return self.data.keys()
    
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
        return {k: v.shape for k, v in self.data.items()}
     
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataCollection):
            return False
        if set(self.data.keys()) != set(other.data.keys()):
            return False
        for k in self.data.keys():
            if not np.array_equal(self.data[k], other.data[k]):
                return False
        return True

@dataclass(eq=False)
class DataPoint(DataCollection[DataType]):
    def shapes(self):
        return {k: v.shape for k, v in self.data.items()}
    
    def __sub__(self, other: "DataPoint[DataType]") -> "DataPoint[DataType]":
        if set(self.data.keys()) != set(other.data.keys()):
            raise KeyError("DataPoint instances must have identical keys for subtraction.")

        new_data = {}
        for k in self.data.keys():
            if self.data[k].shape != other.data[k].shape:
                raise ValueError(f"Shape mismatch for key {k}: "
                                 f"{self.data[k].shape} vs {other.data[k].shape}")
            new_data[k] = np.subtract(self.data[k], other.data[k])

        return DataPoint(new_data, self.order)

@dataclass(eq=False)
class DataTable(DataCollection[DataType]):
    def __len__(self):
        first_key = self.order[0]
        return self.data[first_key].shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self.get_point(i)

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
        
    def get_points(self, ids: np.ndarray) -> DataPoint[DataType]:
        return DataTable({
            k: self.data[k][ids] for k in self.order
        }, self.order)
    
    @classmethod
    def from_matrix(cls,
                    data_matrix: np.ndarray,
                    order: List[DataVariable],
                    shapes_dict: Dict[DataType, Tuple[int, ...]]) -> "DataTable[DataType]":
        data_dict = {}
        col_index = 0
        num_points = data_matrix.shape[0]
        for var in order:
            numel = np.prod(shapes_dict[var])
            data_dict[var] = np.reshape(data_matrix[:, col_index:(col_index + numel)], shape=(num_points, *(shapes_dict[var])))
            
        return cls(data_dict, order)

    @classmethod
    def from_data_points(cls,
                         data_points: List[DataPoint]):
        if len(data_points) == 0:
            return cls({})
        # data_point = data_points[0]
        data_dict = {var: np.concatenate(list(data_point[var][np.newaxis, :] for \
                                         data_point in data_points)) for \
                                            var in data_points[0].order}
        # for data_point in data_points[1:]:
        #     for var in data_point.order:
        #         data_dict[var]

        #     numel = np.prod(shapes_dict[var])
        #     data_dict[var] = np.reshape(data_matrix[:, col_index:(col_index + numel)], shape=(num_points, *(shapes_dict[var])))
            
        return cls(data_dict)

class Interface:
    def __init__(self,
                 all_shapes: Dict[Type[DataVariable], Dict[DataVariable, Tuple[int, ...]]]):
        self.all_shapes = all_shapes

    def shapes(self, data_type: Type[DataVariable]):
        return self.all_shapes[data_type]
        
    def validate_shapes(self,
                        external_shapes: Dict[Type[DataVariable],
                                              Dict[DataVariable, Tuple[int, ...]]],
                        component_name: str):
        for data_type, data_type_shapes in self.all_shapes.items():
            missing_vars = data_type_shapes.keys() - external_shapes[data_type].keys()
            if len(missing_vars):
                msg = (f"Insufficient input variables for component '{component_name}'. "
                    f"Missing variable(s): {missing_vars}")
                raise ValueError(msg)
            for var, shape in data_type_shapes.items():
                if shape is None:
                    continue
                if not (shape == external_shapes[data_type][var]):
                    msg = (f"Input variable dimensions mismatch for component '{component_name}'. "
                        f"Mismatch variable: {var}")
                    raise ValueError(msg)

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
        
    def validate_inputs(self,
                        input_data: Dict[Type[DataVariable], DataCollection[DataVariable]]):
        self.input_interface.validate_shapes(
            external_shapes={data_type: data_collection.shapes() for \
                             data_type, data_collection in input_data.items()},
                             component_name=self.component_name)
            
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
