from typing import Dict, Any, Generic, TypeVar, Type
from enum import Enum
import numpy as np
from scipy.interpolate import RBFInterpolator, LinearNDInterpolator, make_interp_spline
from wiffco.interface import (
    DataTable,
    DataTable,
    DataType
)


# ----------------------------
# Types for Generics
# ----------------------------
InDataType = TypeVar("InDataType", bound=DataType)
OutDataType = TypeVar("OutDataType", bound=DataType)


# ----------------------------
# Interpolator Type Enum
# ----------------------------
class ScatteredInterpolatorType(Enum):
    RBF = "rbf"
    LINEAR = "linear"


# ----------------------------
# Interpolator Parameters
# ----------------------------
class ScatteredInterpolatorParams(Generic[InDataType]):
    """Parameters for creating a scattered data interpolator.

    Args:
        support_data (DataTable[InDataType]): Input coordinates for interpolation.
        out_data (DataTable[OutDataType]): Output values corresponding to support data.
        scattered_interp_type (ScatteredInterpolatorType, optional): Type of interpolator to use.
    """
    def __init__(self,
                 support_data: DataTable[InDataType],
                 out_data: DataTable[OutDataType],
                 scattered_interp_type: ScatteredInterpolatorType = ScatteredInterpolatorType.LINEAR):
        self.support_data = support_data
        self.out_data = out_data
        self.scattered_interp_type = scattered_interp_type


def scattered_interpolator_params_from_dict(param_dict: Dict[str, Any],
                                            support_data_type: Type[InDataType],
                                            out_data_type: Type[OutDataType]) -> ScatteredInterpolatorParams:
    """Create ScatteredInterpolatorParams from a dictionary.

    Args:
        param_dict (Dict[str, Any]): Dictionary containing 'support_data', 'out_data', and 'scattered_interp_type'.
        support_data_type (Type[InDataType]): Data type for input variables (Control or Ambient).
        out_data_type (Type[OutDataType]): Data type for output variables (subclass of ModelOutput).

    Returns:
        ScatteredInterpolatorParams: Constructed parameters object.
    """
    scattered_interp_type = ScatteredInterpolatorType(param_dict["scattered_interp_type"])
    support_data = DataTable({support_data_type(var): np.array(vals)
                              for var, vals in param_dict["ambient_support"].items()})
    out_data = DataTable({out_data_type(var): np.array(vals)
                          for var, vals in param_dict["out_data"].items()})
    return ScatteredInterpolatorParams(
        scattered_interp_type=scattered_interp_type,
        support_data=support_data,
        out_data=out_data
    )


# ----------------------------
# Scattered Interpolator
# ----------------------------
class ScatteredInterpolator(Generic[InDataType, OutDataType]):
    """Interpolator for scattered data points using RBF or linear methods.

    Args:
        interpolator_params (ScatteredInterpolatorParams): Parameters defining support and output data, and interpolation type.
    """
    def __init__(self, interpolator_params: ScatteredInterpolatorParams[InDataType]):
        self.scattered_interp_type = interpolator_params.scattered_interp_type
        self.support_order = interpolator_params.support_data.order
        self.support_shapes = interpolator_params.support_data.shapes()
        self.out_order = interpolator_params.out_data.order
        self.out_shapes = interpolator_params.out_data.shapes()
        support_matrix = interpolator_params.support_data.to_matrix()
        out_matrix = interpolator_params.out_data.to_matrix()
        self.scalar_interp = (support_matrix.shape[1] == 1)

        if self.scattered_interp_type == ScatteredInterpolatorType.RBF:
            self.interpolator = RBFInterpolator(y=support_matrix, d=out_matrix)
        elif self.scattered_interp_type == ScatteredInterpolatorType.LINEAR:
            if self.scalar_interp:
                sort_index = np.argsort(support_matrix.T[0])
                self.interpolator = make_interp_spline(
                    x=support_matrix[sort_index, :].flatten(),
                    y=out_matrix[sort_index, :],
                    k=1)
            else:
                self.interpolator = LinearNDInterpolator(points=support_matrix, values=out_matrix)
        else:
            raise NotImplementedError(
                "ScatteredInterpolator: Only RBF and LINEAR types are implemented."
            )

    def evaluate(self, query: DataTable[InDataType]) -> DataTable[OutDataType]:
        """Evaluate the interpolator at the given query points.

        Args:
            query (DataTable[InDataType]): Input data points where interpolation is requested.

        Returns:
            DataTable[OutDataType]: Interpolated output values.
        """
        x = query.to_matrix(order=self.support_order)
        if self.scalar_interp:
            x = x.flatten()
        interp_result = self.interpolator(x)
        out_data = DataTable.from_matrix(data_matrix=interp_result,
                                         order=self.out_order,
                                         shapes_dict=self.out_shapes)
        return out_data
