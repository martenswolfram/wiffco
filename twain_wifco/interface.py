from typing import (
    List,
    Set,
    Type,
    Tuple,
    TypeVar,
    Dict,
    Any,
    Generic
)
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum


# ======================================================================
# ENUMERATIONS
# ======================================================================

class Ambient(Enum):
    """Enumeration of ambient (environmental and contextual) variables."""
    WIND_SPEED = "wind_speed"
    WIND_DIRECTION = "wind_direction"
    ELECTRICITY_PRICE = "electricity_price"


class Control(Enum):
    """Enumeration of control variables."""
    POWER_REGULATION = "power_regulation"
    YAW_STEERING = "yaw_steering"


class ModelOutput(Enum):
    """Enumeration of model output variables."""
    ELECTRICAL_POWER = "electrical_power"
    DAMAGE_RATE = "damage_rate"


class Aggregated(Enum):
    """Enumeration of aggregated variables (derived from outputs and ambient conditions)."""
    REVENUE_RATE = "revenue_rate"
    DAMAGE_RATE = "damage_rate"


class AccumulatedMetric(Enum):
    """Enumeration of accumulated (time-integrated) metrics."""
    REVENUE = "revenue"
    ACCRUED_DAMAGE = "accrued_damage"


# ======================================================================
# TYPE DEFINITIONS
# ======================================================================

# Union of all possible data variable types
DataVariable = Ambient | Control | ModelOutput | Aggregated | AccumulatedMetric

# Generic type variables for class parameterization
DataType = TypeVar("DataType",
                   Ambient,
                   Control,
                   ModelOutput,
                   Aggregated,
                   AccumulatedMetric)

StatisticalType = TypeVar("StatisticalType",
                          Ambient,
                          Aggregated)

ConstraintType = TypeVar("ConstraintType",
                         Control,
                         Aggregated,
                         AccumulatedMetric)


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def get_default_value(data_var: DataVariable,
                      shape: Tuple[int]) -> np.ndarray:
    """Return a default numpy array for a given data variable."""
    if data_var == Control.POWER_REGULATION:
        return np.ones(shape=shape)
    elif data_var == Control.YAW_STEERING:
        return np.zeros(shape=shape)
    raise ValueError(f"No default value defined for data variable '{data_var}'.")


def get_abs_tol(data_var: DataVariable) -> float:
    """Return the absolute tolerance for a specific data variable."""
    mapping = {
        Ambient.WIND_DIRECTION: 0.001,
        Ambient.WIND_SPEED: 0.001,
        Ambient.ELECTRICITY_PRICE: 0.001,
        Control.POWER_REGULATION: 0.1,
        AccumulatedMetric.ACCRUED_DAMAGE: 0.1
    }
    return mapping.get(data_var, 0.0)


# ======================================================================
# DATA COLLECTION CLASSES
# ======================================================================

@dataclass
class DataCollection(Generic[DataType]):
    """Base container class for mapping variables to numpy arrays.

    This class handles shared functionality between `DataPoint` and `DataTable`.
    """

    data: dict[DataType, np.ndarray]
    data_type: Type[DataType] | None = None
    order: list[DataType] | None = None
    abs_tols: dict[DataType, float] | None = None

    def __post_init__(self):
        """Validate and initialize derived attributes."""
        if self.order is None:
            self.order = list(self.data.keys())
        if len(self.order) > 0:
            self.data_type = type(self.order[0])
            if self.abs_tols is None:
                self.abs_tols = {dv: get_abs_tol(dv) for dv in self.order}

    def shapes(self):
        """Must be implemented by subclasses to return variable shapes."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement `shapes()`."
        )

    def keys(self):
        """Return the variable keys of this collection."""
        return self.data.keys()

    def items(self):
        """Return key-value pairs of data variables and arrays."""
        return self.data.items()

    def __getitem__(self, key: DataType) -> np.ndarray:
        """Access the array corresponding to a given variable."""
        return self.data[key]

    def abs_tols_vec(self) -> np.ndarray:
        """Concatenate absolute tolerances into a flat vector."""
        return np.concatenate([
            self.abs_tols[dv] * np.ones(np.prod(self.shapes()[dv]))
            for dv in self.order
        ])

    def abs_tol(self, key: DataType) -> float:
        """Return the absolute tolerance for a single variable."""
        return self.abs_tols[key]

    def to_vector(self, keys: List[DataType] | None = None) -> np.ndarray:
        """Flatten and concatenate all variable arrays into a single vector."""
        if keys is None:
            keys = self.order
        return np.concatenate([self.data[k].ravel() for k in self.order if k in keys])

    def update_from_vector(self, vector: np.ndarray):
        """Update variable data from a flattened vector."""
        offset = 0
        for key in self.order:
            size = self.data[key].size
            self.data[key] = vector[offset:offset+size].reshape(self.data[key].shape)
            offset += size

    def __eq__(self, other: object) -> bool:
        """Compare two DataCollections elementwise within tolerance."""
        if not isinstance(other, DataCollection):
            return False
        if set(self.data.keys()) != set(other.data.keys()):
            return False
        return all(
            np.allclose(self.data[k], other.data[k], atol=self.abs_tol(k))
            for k in self.data.keys()
        )


@dataclass(eq=False)
class DataPoint(DataCollection[DataType]):
    """Represents a single data point (non-tabular)."""

    def shapes(self):
        """Return the shape of each variable array."""
        return {k: v.shape for k, v in self.data.items()}

    def __sub__(self, other: "DataPoint[DataType]") -> "DataPoint[DataType]":
        """Subtract two DataPoint objects elementwise."""
        if set(self.data.keys()) != set(other.data.keys()):
            raise KeyError("DataPoint instances must have identical keys for subtraction.")

        new_data = {}
        for k in self.data.keys():
            if self.data[k].shape != other.data[k].shape:
                raise ValueError(f"Shape mismatch for key {k}: "
                                 f"{self.data[k].shape} vs {other.data[k].shape}")
            new_data[k] = np.subtract(self.data[k], other.data[k])

        return DataPoint(new_data, self.order)

    @classmethod
    def from_vector(cls,
                    data_vector: np.ndarray,
                    order: List[DataVariable],
                    shapes_dict: Dict[DataType, Tuple[int, ...]]) -> "DataPoint[DataType]":
        """Construct a DataPoint instance from a flat vector."""
        data_dict = {}
        offset = 0
        for var in order:
            numel = np.prod(shapes_dict[var])
            data_dict[var] = np.reshape(data_vector[offset:(offset + numel)], shape=shapes_dict[var])
            offset += numel
        return cls(data_dict, order)


@dataclass(eq=False)
class DataTable(DataCollection[DataType]):
    """Represents a table of multiple data points (2D structure)."""

    def __len__(self) -> int:
        """Return the number of rows (data points) in the table."""
        return self.data[self.order[0]].shape[0]

    def __iter__(self):
        """Iterate over individual DataPoint instances."""
        for i in range(len(self)):
            yield self.get_point(i)

    def shapes(self) -> Dict[DataType, Tuple[int, ...]]:
        """Return shapes of the per-variable data arrays (excluding leading row dimension)."""
        return {k: v.shape[1:] for k, v in self.data.items()}

    def to_matrix(self) -> np.ndarray:
        """Flatten all variable arrays and combine into a single 2D matrix."""
        n_points = len(self)
        flattened = [self.data[k].reshape(n_points, -1) for k in self.order]
        return np.concatenate(flattened, axis=1)

    def get_point(self, idx: int) -> DataPoint[DataType]:
        """Extract a single DataPoint (row) by index."""
        return DataPoint({k: self.data[k][idx] for k in self.order}, self.order)

    def get_points(self, ids: np.ndarray) -> "DataTable[DataType]":
        """Extract multiple rows by index array."""
        return DataTable({k: self.data[k][ids] for k in self.order}, self.order)

    def find_matching_point(self, data_point: DataPoint[DataType]) -> int:
        """Find the row index corresponding to a given DataPoint."""
        mask = np.all(np.isclose(
            self.to_matrix(),
            data_point.to_vector(),
            atol=self.abs_tols_vec()),
            axis=1)
        row_index = np.where(mask)[0]
        if not len(row_index):
            raise ValueError(f"Data point not found in DataTable '{self.__class__.__name__}'.")
        return row_index[0]

    def update_point_from_vector(self, ind: int, vector: np.ndarray):
        """Overwrite a specific row of the table with new flattened data."""
        i = 0
        for k in self.order:
            n = self.data[k][ind].size
            self.data[k][ind] = vector[i:i+n].reshape(self.data[k].shape[1:])
            i += n

    @classmethod
    def from_matrix(cls,
                    data_matrix: np.ndarray,
                    order: List[DataVariable],
                    shapes_dict: Dict[DataType, Tuple[int, ...]]) -> "DataTable[DataType]":
        """Construct a DataTable from a flat 2D numpy matrix."""
        data_dict = {}
        offset = 0
        n_rows = data_matrix.shape[0]
        for var in order:
            numel = np.prod(shapes_dict[var])
            data_dict[var] = np.reshape(
                data_matrix[:, offset:(offset + numel)],
                shape=(n_rows, *shapes_dict[var])
            )
            offset += numel
        return cls(data_dict, order)

    @classmethod
    def from_data_points(cls, data_points: List[DataPoint]):
        """Combine a list of DataPoint objects into a single DataTable."""
        if not data_points:
            return cls({})
        order = data_points[0].order
        data_dict = {
            var: np.stack([dp[var] for dp in data_points])
            for var in order
        }
        return cls(data_dict, order)


class Interface:
    def __init__(self,
                 all_shapes: Dict[Type[DataVariable],
                                  Dict[DataVariable, Tuple[int, ...]]]):
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
