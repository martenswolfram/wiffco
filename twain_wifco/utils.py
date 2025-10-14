from typing import Dict, Any
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from scipy.interpolate import RBFInterpolator
from twain_wifco.interface import (
    DataType,
    DataVariable)

class DataInterpolator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self,
                 query: Dict[DataType, np.ndarray]) -> Dict[DataType, np.ndarray]:
        pass

class InterpolatorType(Enum):
    RBF_INTERPOLATOR = "rbf_interpolator"

class RadialBFInterpolatorParams:
    def __init__(self,
                 support_points: Dict[DataType, np.ndarray],
                 out_values: Dict[DataType, np.ndarray]):
        self.support_points = support_points
        self.in_variables = list(self.support_points.keys())
        self.in_dims = list(self.support_points[in_var].shape[1] for \
                            in_var in self.in_variables)        
        self.out_values = out_values
        self.out_variables = list(self.out_values.keys())
        self.out_dims = list(self.out_values[out_var].shape[1] for \
                             out_var in self.out_variables)

def rbf_interpolator_params_from_dict(param_dict: Dict[str, Dict | Any],
                                      in_data_type: DataType,
                                      out_data_type: DataType):
    support_points = {in_data_type(in_var): np.array(supp) for \
                      in_var, supp in param_dict["support_points"].items()}
    out_values = {out_data_type(out_var): np.array(out_vals) for \
                  out_var, out_vals in param_dict["out_values"].items()}

    return RadialBFInterpolatorParams(
        support_points=support_points,
        out_values=out_values
        )

class RadialBFInterpolator(DataInterpolator):
    def __init__(self,
                 interpolator_params: RadialBFInterpolatorParams):
        self.params = interpolator_params
        self.rbf_interpolator = RBFInterpolator(
            y=np.hstack(list(interpolator_params.support_points[in_var] for \
                        in_var in self.params.in_variables)),
            d=np.hstack(list(interpolator_params.out_values[out_var] for \
                        out_var in self.params.out_variables))
        )

    def _partition_result(self, result_vector):
        return np.split(result_vector, np.cumsum(self.out_dims[:-1]))
        
    def evaluate(self,
                 query: Dict[DataType, np.ndarray]) -> Dict[DataType, np.ndarray]:
        x = np.hstack(query[in_var] for in_var in self.in_variables)
        result = self.rbf_interpolator(x=x)
        return {out_var: res_part for \
                out_var, res_part in zip(self.out_variables, self._partition_result(result))}
