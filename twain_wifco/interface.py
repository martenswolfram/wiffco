from typing import (
    List,
    Type,
    Tuple,
    TypeVar,
    Dict,
    Generic,
    Callable,
    Any
)
import numpy as np
from scipy.sparse import coo_matrix
from dataclasses import dataclass
from abc import ABC
from enum import Enum
from twain_wifco.utils import print_table

# ======================================================================
# ENUMERATIONS
# ======================================================================

class DataEnum(Enum):
    def __repr__(self):
        return self.value

class Ambient(DataEnum):
    """Enumeration of ambient (environmental and contextual) variables."""
    WIND_SPEED = "wind_speed"
    WIND_DIRECTION = "wind_direction"
    TURBULENCE_INTENSITY = "turbulence_intensity"
    ELECTRICITY_PRICE = "electricity_price"

class Control(DataEnum):
    """Enumeration of control variables."""
    POWER_REGULATION = "power_regulation"
    YAW_ANGLE = "yaw_angle"


class ModelOutput(DataEnum):
    """Enumeration of model output variables."""
    ELECTRICAL_POWER = "electrical_power"
    DAMAGE_RATE = "damage_rate"


class Aggregate(DataEnum):
    """Enumeration of aggregated variables (derived from outputs and ambient conditions)."""
    REVENUE_RATE = "revenue_rate"
    DAMAGE_RATE = "damage_rate"


class AccumulatedMetric(DataEnum):
    """Enumeration of accumulated (time-integrated) metrics."""
    REVENUE = "revenue"
    ACCRUED_DAMAGE = "accrued_damage"

MAP_STR_TO_ENUM: Dict[str, Type[DataEnum]] = {
    "ambient": Ambient,
    "control": Control,
    "model_output": ModelOutput,
    "aggregate": Aggregate,
    "accumulated_metric": AccumulatedMetric}

MAP_ENUM_TO_STR: Dict[Type[DataEnum], str] = {
    Ambient: "ambient",
    Control: "control",
    ModelOutput: "model_output",
    Aggregate: "aggregate",
    AccumulatedMetric: "accumulated_metric"}


# ======================================================================
# TYPE DEFINITIONS
# ======================================================================

# Union of all possible data variable types
DataVariable = Ambient | Control | ModelOutput | Aggregate | AccumulatedMetric

# Generic type variables for class parameterization
DataType = TypeVar("DataType", bound=DataEnum)

# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def get_default_value(data_var: DataVariable,
                      shape: Tuple[int] | None) -> np.ndarray:
    """Return a default numpy array for a given data variable."""
    if shape is None:
        # Cannot provide default for variable shape
        return None
    if data_var == Ambient.ELECTRICITY_PRICE:
        return np.zeros(shape=shape)
    elif data_var == Control.POWER_REGULATION:
        return np.ones(shape=shape)
    elif data_var == Control.YAW_ANGLE:
        return np.zeros(shape=shape)
    elif data_var == Ambient.WIND_SPEED:
        return np.zeros(shape=shape)
    raise ValueError(f"No default value defined for data variable '{data_var}'.")


def get_abs_tol(data_var: DataVariable) -> float:
    """Return the absolute tolerance for a specific data variable."""
    mapping = {
        Ambient.WIND_DIRECTION: 0.001,
        Ambient.WIND_SPEED: 0.001,
        Ambient.ELECTRICITY_PRICE: 0.001,
        Control.POWER_REGULATION: 0.1,
        Aggregate.DAMAGE_RATE: 0.1,
        AccumulatedMetric.ACCRUED_DAMAGE: 0.1
    }
    return mapping.get(data_var, 0.0)


# ======================================================================
# DATA COLLECTION CLASSES
# ======================================================================

@dataclass(eq=False, repr=False)
class DataTable(Generic[DataType]):
    """Base container class for mapping variables to numpy arrays.

    This class handles shared functionality between `DataTable` and `DataTable`.
    """

    data: dict[DataType, np.ndarray]
    data_type: Type[DataType] | None = None
    order: list[DataType] | None = None
    abs_tols: dict[DataType, float] | None = None
    size: int = 0

    def __post_init__(self):
        """Validate and initialize derived attributes."""
        if self.order is None:
            self.order = list(self.data.keys())
        if len(self.order) > 0:
            self.data_type = type(self.order[0])
            if self.abs_tols is None:
                self.abs_tols = {dv: get_abs_tol(dv) for dv in self.order}
            if not all([key_data.ndim > 0 for key_data in self.data.values()]):
                raise ValueError("Input data must have ndim > 0.")
            num_points = [key_data.shape[0] for key_data in self.data.values()]
            if len(set(num_points)) != 1:
                raise ValueError("Inconsistent number of data entries.")
            self.size = num_points[0]

    def __len__(self) -> int:
        """Return the number of rows (data points) in the table."""
        return self.size

    def __iter__(self):
        """Iterate over individual data entries instances."""
        for i in range(len(self)):
            yield {key: data[i] for key, data in self.items()}
    
    def __getitem__(self, key: DataType) -> np.ndarray:
        """Access the array corresponding to a given variable."""
        return self.data[key]

    # def __setitem__(self, key: DataType, data: np.array) -> np.ndarray:
    #     """Access the array corresponding to a given variable."""
    #     self.data[key] = data
    
    def keys(self):
        """Return the variable keys of this collection."""
        return self.data.keys()

    def items(self):
        """Return key-value pairs of data variables and arrays."""
        return self.data.items()

    def shapes(self) -> Dict[DataType, Tuple[int, ...]]:
        """Return shapes of the per-variable data arrays (excluding leading row dimension)."""
        return {k: v.shape[1:] for k, v in self.data.items()}

    def abs_tols_vec(self) -> np.ndarray:
        """Concatenate absolute tolerances into a flat vector."""
        return np.concatenate([
            self.abs_tols[dv] * np.ones(np.prod(self.shapes()[dv], dtype=int))
            for dv in self.order
        ])

    def abs_tol(self, key: DataType) -> float:
        """Return the absolute tolerance for a single variable."""
        return self.abs_tols[key]

    def to_vector(self,
                  order: List[DataType] | None = None,
                  fill_shapes: Dict[DataType, Tuple[int, ...]] | None = None,
                  fill_value: float | None = None,
                  batch_multiply: int | None = None) -> np.ndarray:
        """
        Flatten and concatenate all variable arrays into a single vector.
        Note that the order can be specified different from own order.
        Elements for missing variables are filled with fill_value.
        """
        def fill_vec(shape: Tuple[int, ...]):
            return np.full(shape=np.prod(shape, dtype=int), fill_value=fill_value)

        if order is None:
            order = self.order
        if batch_multiply is None:
            return np.concatenate([self.data[k].ravel() \
                                   if k in self.keys() else fill_vec(fill_shapes[k]) \
                                    for k in order])
        else:
            # For batch evaluation, tile the data for each variable
            return np.concatenate([np.tile(self.data[k].ravel(), batch_multiply) for k in order])

    def to_matrix(self,
                  order: List[DataType] | None = None) -> np.ndarray:
        """Flatten all variable arrays and combine into a single 2D matrix.
        Each row corresponds to one data point.
        Note that the order can be different from self.order."""
        if order is None:
            order = self.order
        flattened = [self.data[k].reshape(self.size, -1) for k in order]
        return np.concatenate(flattened, axis=1)
    
    def update_from_vector(self, vector: np.ndarray):
        """
        Update variable data from a flattened vector.
        Note that this assumes that the vector was created with matching order.
        """
        offset = 0
        for key in self.order:
            size = self.data[key].size
            self.data[key] = vector[offset:offset+size].reshape(self.data[key].shape)
            offset += size

    def __eq__(self, other: object) -> bool:
        """Compare two DataCollections elementwise within tolerance."""
        if not isinstance(other, DataTable):
            return False
        if set(self.data.keys()) != set(other.data.keys()):
            return False
        return all(
            np.allclose(self.data[k], other.data[k], atol=self.abs_tol(k))
            for k in self.data.keys()
        )
    
    def __repr__(self):
        out = ""
        for var, data in self.data.items():
            out += f"{var}:\n{data}\n"
        return out

    def get_points(self, ids: np.ndarray) -> "DataTable[DataType]":
        """Extract multiple rows by index array."""
        return DataTable({k: self.data[k][ids] for k in self.order}, self.order)

    def find_matching_points(self, data_table: "DataTable[DataType]") -> int:
        """Find the row indices corresponding to a given DataTable."""
        # Find matching row indices
        row_indices = [next((i for i, ref_row in enumerate(self.to_matrix()) if np.allclose(
            ref_row, query_row, atol=self.abs_tols_vec())), -1) for query_row in data_table.to_matrix()]
        return row_indices

    def update_point_from_vector(self, ind: int, vector: np.ndarray):
        """Overwrite a specific row of the table with new flattened data."""
        i = 0
        for k in self.order:
            n = self.data[k][ind].size
            self.data[k][ind] = vector[i:i+n].reshape(self.data[k].shape[1:])
            i += n

    @classmethod
    def from_vector(cls,
                    data_vector: np.ndarray,
                    order: List[DataVariable],
                    shapes_dict: Dict[DataType, Tuple[int, ...]],
                    num_points: int) -> "DataTable[DataType]":
        """Construct a DataTable instance from a flat vector."""
        data_dict = {}
        offset = 0
        for var in order:
            numel = np.prod(shapes_dict[var], dtype=int) * num_points
            data_dict[var] = np.reshape(data_vector[offset:(offset + numel)],
                                        shape=(num_points,) + shapes_dict[var])
            offset += numel
        return cls(data_dict, order)
    
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
            numel = np.prod(shapes_dict[var], dtype=int)
            data_dict[var] = np.reshape(
                data_matrix[:, offset:(offset + numel)],
                shape=(n_rows, *shapes_dict[var])
            )
            offset += numel
        return cls(data_dict, order)

    def __repr__(self):
        
        table = [list(k.value for k in self.order)]
        for point_data in self:
            table.append(
                list(np.array2string(point_data[k],
                                     precision=3,
                                     separator=", ") \
                                        for k in self.order))
        return print_table(table)
    
    def expected_value(self, probabilities: coo_matrix):
        # probabilities: shape [N, M]
        # self.data[.]: [N, ...]
        if probabilities.shape[0] != self.size:
            raise ValueError(
                f"Expected probabilities of shape ({self.size}, N), got {probabilities.shape}."
            )
        # Normalize probabilities in case they don't sum exactly to 1
        p = probabilities / np.sum(probabilities, axis=0)

        expected_data = {}
        for key, data in self.data.items():
            # Compute weighted sum along axis 0
            data_reshaped = data.reshape(self.size, -1)
            expected_reshaped = p.T @ data_reshaped
            expected_data[key] = expected_reshaped.reshape(
                probabilities.shape[1], *data.shape[1:])

        return DataTable(expected_data, self.order)
    
    def tile(self, reps: int):
        tiled_data = {key: np.tile(
            key_data,
            reps=(reps,) + (1,)*(key_data.ndim - 1)) for \
                key, key_data in self.data.items()}
        return DataTable(data=tiled_data)

    def repeat(self, reps: int):
        repeated_data = {key: np.repeat(self.data[key],
                                        repeats=reps,
                                        axis=0) for key in self.order}
        return DataTable(data=repeated_data)
    
    def extract(self, indices: np.ndarray):
        if len(indices) == 0:
            raise ValueError("Need at least one index to extract from DataTable.")
        
        extracted_data = {key: self.data[key][indices] for key in self.order}
        return DataTable(data=extracted_data)

@dataclass(eq=False, repr=False)
class DataPoint(DataTable[DataType]):
    """A specialization of DataTable that contains exactly one data point."""
    
    def __post_init__(self):
        super().__post_init__()
        if self.size != 1:
            raise ValueError("DataPoint must contain exactly one row (size == 1).")
    
    @classmethod
    def from_table(cls, table: "DataTable[DataType]", index: int) -> "DataPoint[DataType]":
        """Create a DataPoint from one row of a DataTable."""
        if index < 0 or index >= len(table):
            raise IndexError("Index out of range for DataTable.")
        return cls({k: table.data[k][index:index+1] for k in table.order}, table.order)

# ======================================================================
# INTERFACE AND COMPONENT CLASSES
# ======================================================================

class Interface:
    """Defines input and output variable structures for a component.

    Each interface specifies the required variables and their expected shapes,
    grouped by their data type (e.g. Ambient, Control, ModelOutput, etc.).

    Attributes:
        all_shapes: Dictionary mapping data types to variable-shape dictionaries.
            Example:
                {
                    Ambient: {Ambient.WIND_SPEED: (1,), Ambient.WIND_DIRECTION: (1,)},
                    Control: {Control.POWER_REGULATION: (1,)}
                }
    """

    def __init__(self, all_shapes: Dict[Type[DataVariable],
                                   Dict[DataVariable, Tuple[int, ...]]] = {}):
        self.shapes = all_shapes

    def validate_shapes(self,
                        external_shapes: Dict[Type[DataVariable],
                                              Dict[DataVariable, Tuple[int, ...]]],
                        component_name: str):
        """Validate that all required variables exist and have matching shapes.

        Args:
            external_shapes: Mapping of variable shapes provided by external data.
            component_name: Name of the component being validated.

        Raises:
            ValueError: If required variables are missing or shape mismatches occur.
        """
        for data_type, required_shapes in self.shapes.items():
            provided_shapes = external_shapes.get(data_type, {})

            # Check for missing variables
            missing_vars = required_shapes.keys() - provided_shapes.keys()
            if missing_vars:
                raise ValueError(
                    f"Component '{component_name}' missing required variables of type "
                    f"{data_type.__name__}: {missing_vars}"
                )

            # Check for shape mismatches
            for var, req_shape in required_shapes.items():
                if req_shape is None:
                    continue  # shape is not constrained
                if provided_shapes[var] != req_shape:
                    raise ValueError(
                        f"Component '{component_name}' shape mismatch for variable '{var}'. "
                        f"Expected {req_shape}, got {provided_shapes[var]}."
                    )

T = TypeVar("T", bound="Component")

class Component(ABC):
    """Abstract base class for all components in the toolbox.

    A `Component` represents any computational unit (control policy, model, aggregator)
    that transforms a set of input variables into output variables.

    Attributes:
        component_name: Descriptive name of the component.
        input_interface: Interface object defining required input variables.
        output_interface: Interface object defining produced output variables.
    """
    component_name: str = "<unnamed component>"
    input_interface: Interface | None = None
    output_interface: Interface | None = None

    def validate_shapes(self,
                        input_shapes: Dict[Type[DataVariable],
                                           Dict[DataVariable, Tuple[int, ...]]]):
        """Validate that the input data shapes satisfy this component's interface.

        Args:
            input_shapes: Mapping from data type (e.g. Ambient, Control)
                to shapes for each data variable.

        Raises:
            ValueError: If input variables are missing or mismatched in shape.
        """
        if self.input_interface is not None:
            # Validate shape consistency with the declared input interface
            self.input_interface.validate_shapes(
                external_shapes=input_shapes,
                component_name=self.component_name
            )

    def validate_input(self, input_tables: List[DataTable]):
        
        # Check consistent number of data points
        if not len({len(table) for table in input_tables}) == 1:
            raise ValueError(
                f"Component '{self.component_name}' called with inconsistent number of data points."
                )

        # Collect the external data shapes for validation
        external_shapes = {
            data_table.data_type: data_table.shapes()
            for data_table in input_tables
        }
        
        # Validate consistency with the declared input interface
        self.validate_shapes(
            input_shapes=external_shapes
        )

    # @staticmethod
    # def with_validation(func: Callable[..., Any]) -> Callable[..., Any]:
    #     def wrapper(self: T, **kwargs: Any) -> Any:
    #         self.validate_input(*[value for value in kwargs.values() if isinstance(value, DataTable)])
    #         return func(self, **kwargs)
    #     return wrapper
    
    def __repr__(self):
        out = f"{self.__class__.__name__} '{self.component_name}'"
        repr_details = self.repr_details()
        if repr_details is None:
            return out
        else:
            return out + ":\n" + repr_details
    
    def repr_details(self):
        return None
         