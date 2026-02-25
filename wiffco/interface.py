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
from wiffco.utils import print_table

# ======================================================================
# ENUMERATIONS
# ======================================================================

class DataEnum(Enum):
    def __repr__(self):
        return self.value

class Ambient(DataEnum):
    """Enumeration of ambient (environmental and contextual) variables."""
    WIND_SPEED_MPS = "wind_speed_mps"
    WIND_DIRECTION_DEG = "wind_direction_deg"
    TURBULENCE_INTENSITY = "turbulence_intensity"
    ELECTRICITY_PRICE_EURPKWH = "electricity_price_eurpkwh"

class Control(DataEnum):
    """Enumeration of control variables."""
    POWER_REGULATION = "power_regulation"
    YAW_ANGLE_DEG = "yaw_angle_deg"

class ModelOutput(DataEnum):
    """Enumeration of model output variables."""
    ELECTRICAL_POWER_KW = "electrical_power_kw"
    DAMAGE_RATE = "damage_rate"
    DAMAGE_RATE_2 = "damage_rate_2"
    DEL = "del"

class Aggregate(DataEnum):
    """Enumeration of aggregated variables (derived from outputs and ambient conditions)."""
    ELECTRICAL_POWER_KW = "electrical_power_kw"
    REVENUE_RATE = "revenue_rate"
    DAMAGE_RATE = "damage_rate"
    DAMAGE_RATE_2 = "damage_rate_2"
    DEL = "del"

class AccumulatedMetric(DataEnum):
    """Enumeration of accumulated (time-integrated) metrics."""
    REVENUE_EUR = "revenue_eur"
    ENERGY_PRODUCED = "energy_produced"
    ACCRUED_DAMAGE = "accrued_damage"
    ACCRUED_DAMAGE_2 = "accrued_damage_2"
    ACCRUED_DEL = "accrued_del"

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
    if data_var == Ambient.ELECTRICITY_PRICE_EURPKWH:
        return np.zeros(shape=shape)
    elif data_var == Control.POWER_REGULATION:
        return np.ones(shape=shape)
    elif data_var == Control.YAW_ANGLE_DEG:
        return np.zeros(shape=shape)
    elif data_var == Ambient.WIND_SPEED_MPS:
        return np.zeros(shape=shape)
    raise ValueError(f"No default value defined for data variable '{data_var}'.")


def get_abs_tol(data_var: DataVariable) -> float:
    """Return the absolute tolerance for a specific data variable."""
    mapping = {
        Ambient.WIND_DIRECTION_DEG: 0.001,
        Ambient.WIND_SPEED_MPS: 0.001,
        Ambient.ELECTRICITY_PRICE_EURPKWH: 0.001,
        Control.POWER_REGULATION: 0.001,
        Aggregate.DAMAGE_RATE: 0.001,
        Aggregate.DAMAGE_RATE_2: 0.001,
        AccumulatedMetric.ACCRUED_DAMAGE: 0.001,
        AccumulatedMetric.ACCRUED_DAMAGE_2: 0.001
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
                  point_index: int | None = None,
                  order: List[DataType] | None = None) -> np.ndarray:
        """
        Flatten and concatenate all variable arrays into a single vector.
        Note that the order can be specified different from own order.
        """
        if order is None:
            order = self.order
        if len(order) == 0:
            # Empty vector
            return np.array([])
        if point_index is None:
            return np.concatenate([self.data[k].ravel() for k in order])
        else:
            if point_index >= self.size:
                raise IndexError("Index out of bounds.")
            return np.concatenate([self.data[k][point_index].ravel() for k in order])            
        
    def to_matrix(self,
                  order: List[DataType] | None = None) -> np.ndarray:
        """Flatten all variable arrays and combine into a single 2D matrix.
        Each row corresponds to one data point.
        Note that the order can be different from self.order."""
        if order is None:
            order = self.order
        flattened = [self.data[k].reshape(self.size, -1) for k in order]
        return np.concatenate(flattened, axis=1)
    
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

    def find_matching_points(self, other_data_table: "DataTable[DataType]") -> int:
        """Find the row indices corresponding to rows in other DataTable."""
        # Find matching row indices
        row_indices = [next((i for i, ref_row in enumerate(self.to_matrix()) if np.allclose(
            ref_row, query_row, atol=self.abs_tols_vec())), -1) for query_row in other_data_table.to_matrix()]
        return row_indices # size = len(other_data_table)

    def find_nearest_points(self, other_data_table: "DataTable[DataType]") -> int:
        """Find the nearest-row indices corresponding to rows in other DataTable."""
        # Find nearest row indices
        diff = self.to_matrix()[np.newaxis, :, :] - other_data_table.to_matrix(order=self.order)[:, np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        return np.argmin(dists, axis=1) # size = len(other_data_table)

    def update_point_from_vector(self,
                                 order: List[DataVariable],
                                 point_index: int,
                                 vector: np.ndarray):
        """Overwrite a specific row of the table with new flattened data."""
        i = 0
        for k in order:
            n = self.data[k][point_index].size
            self.data[k][point_index] = vector[i:i+n].reshape(self.data[k].shape[1:])
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
    
    def expected_value(self, probabilities: np.ndarray | coo_matrix):
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
    
    def extract(self, indices: np.ndarray | int):
        if isinstance(indices, int):
            if indices >= self.size:
                raise IndexError("Index out of bounds.")
            extracted_data = {key: self.data[key][indices][np.newaxis, ...] for \
                              key in self.order}
        else:
            if len(indices) == 0:
                raise ValueError("Need at least one index to extract from DataTable.")
            if len(indices) == 1:
                # return a view rather than a copy
                extracted_data = {key: self.data[key][indices[0]][np.newaxis, ...] for \
                    key in self.order}
            else:
                # fancy indexing -> copy to new array
                extracted_data = {key: self.data[key][indices] for key in self.order}
        
        return DataTable(data=extracted_data)
    
    def copy(self):
        return DataTable(data={key: key_data.copy() for key, key_data in self.data.items()})

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
    
    def __repr__(self):
        out = f"{self.__class__.__name__} '{self.component_name}'"
        repr_details = self.repr_details()
        if repr_details is None:
            return out
        else:
            return out + ":\n" + repr_details
    
    def repr_details(self):
        return None
         