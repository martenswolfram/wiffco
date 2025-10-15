from typing import Dict, Any
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from scipy.interpolate import (
    RBFInterpolator,
    LinearNDInterpolator,
    interp1d)
from twain_wifco.interface import (
    DataType,
    DataVariable)

class ScatteredInterpolatorType(Enum):
    RBF = "rbf"
    LINEAR = "linear"

class ScatteredInterpolatorParams:
    def __init__(self,
                 scattered_interp_type: ScatteredInterpolatorType,
                 support_points: Dict[DataType, np.ndarray],
                 out_values: Dict[DataType, np.ndarray]):
        self.scattered_interp_type = scattered_interp_type
        self.support_points = support_points
        self.in_variables = list(self.support_points.keys())
        self.in_dims = list(self.support_points[in_var].shape[1] for \
                            in_var in self.in_variables)        
        self.out_values = out_values
        self.out_variables = list(self.out_values.keys())
        self.out_dims = list(self.out_values[out_var].shape[1] for \
                             out_var in self.out_variables)

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
        self.params = interpolator_params
        support_points = np.hstack(list(interpolator_params.support_points[in_var] for \
                                        in_var in self.params.in_variables))
        out_values = np.hstack(list(interpolator_params.out_values[out_var] for \
                                    out_var in self.params.out_variables))
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

    def _partition_result(self, result_vector):
        return np.split(result_vector, np.cumsum(self.params.out_dims[:-1]))

    def evaluate(self,
                 query: Dict[DataType, np.ndarray]) -> Dict[DataType, np.ndarray]:
        x = np.hstack(list(query[in_var] for in_var in self.params.in_variables))[np.newaxis, :]
        result = self.interpolator(x=x).flatten()
        return {out_var: res_part for \
                out_var, res_part in zip(self.params.out_variables, self._partition_result(result))}
