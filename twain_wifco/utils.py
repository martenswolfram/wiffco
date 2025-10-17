from typing import Dict, Any, List, Generic, TypeVar
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from scipy.interpolate import (
    RBFInterpolator,
    LinearNDInterpolator,
    interp1d)
from twain_wifco.interface import (
    Ambient
    DataTable,
    DataVariable)

def hstacked_from_dict(data_dict: Dict[DataVariable, np.ndarray],
                       variables: List[DataVariable]):
    return np.hstack(list(data_dict[var] for \
                          var in variables))

def partition_into_dict(stacked_array: np.ndarray,
                        variables: List[DataVariable],
                        partition_indices: np.ndarray):
    partitioned_data = np.split(stacked_array, partition_indices)
    return {var: part for \
            var, part in zip(variables, partitioned_data)}


class ScatteredInterpolatorType(Enum):
    RBF = "rbf"
    LINEAR = "linear"

InDataType = TypeVar("InDataType",
                     Ambient, Control)

class ScatteredInterpolatorParams(Generic[DataType]):
    def __init__(self,
                 scattered_interp_type: ScatteredInterpolatorType,
                 support_points: DataTable[DataType],
                 out_values: DataTable):
        self.scattered_interp_type = scattered_interp_type
        self.support_points = support_points
        self.in_variables = list(self.support_points.keys())
        self.out_values = out_values
        self.out_variables = list(self.out_values.keys())
        
def scattered_interpolator_params_from_dict(param_dict: Dict[str, Dict | Any],
                                            in_data_type: DataType,
                                            out_data_type: DataType):
    scattered_interp_type = ScatteredInterpolatorType(param_dict["scattered_interp_type"])
    support_points = {in_data_type(in_var): np.array(supp) for \
                      in_var, supp in param_dict["support_points"].items()}
    out_values = {out_data_type(out_var): np.array(out_vals) for \
                  out_var, out_vals in param_dict["out_values"].items()}

    return ScatteredInterpolatorParams(
        scattered_interp_type=scattered_interp_type,
        support_points=support_points,
        out_values=out_values
        )
        
class ScatteredInterpolator:
    def __init__(self,                 
                 interpolator_params: ScatteredInterpolatorParams):
        self.scattered_interp_type = interpolator_params.scattered_interp_type
        out_dims = np.array(list(interpolator_params.out_values[out_var].shape[1] for \
                                 out_var in interpolator_params.out_variables))
        self.out_partition_indices = np.cumsum(out_dims[:-1])
        self.in_variables = interpolator_params.in_variables
        self.out_variables = interpolator_params.out_variables
        support_points = hstacked_from_dict(data_dict=interpolator_params.support_points,
                                          variables=interpolator_params.in_variables)
        out_values = hstacked_from_dict(data_dict=interpolator_params.out_values,
                                      variables=interpolator_params.out_variables)
        if self.scattered_interp_type == ScatteredInterpolatorType.RBF:
            self.interpolator = RBFInterpolator(
                y=support_points,
                d=out_values
            )
        elif self.scattered_interp_type == ScatteredInterpolatorType.LINEAR:
            if support_points.shape[1] == 1:
                sort_index = np.argsort(support_points.T[0])
                self.interpolator = interp1d(x=support_points[sort_index, :].flatten(),
                                             y=out_values[sort_index, :].T,
                                             fill_value="extrapolate")
            else:
                self.interpolator = LinearNDInterpolator(
                    points=support_points,
                    values=out_values
                )
        else:
            raise NotImplementedError("ScatteredDataInterpolator: Only RBF type implemented.")
        pass

    def evaluate(self,
                 query: Dict[DataType, np.ndarray]) -> Dict[DataType, np.ndarray]:
        x = hstacked_from_dict(data_dict=query,
                             variables=self.in_variables)[np.newaxis, :]
        result = self.interpolator(x=x).flatten()
        return partition_into_dict(stacked_array=result,
                                   variables=self.out_variables,
                                   partition_indices=self.out_partition_indices)
