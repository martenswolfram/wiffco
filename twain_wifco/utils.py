from typing import Dict, Any, List, Generic, TypeVar, Type
from enum import Enum
import numpy as np
from scipy.interpolate import (
    RBFInterpolator,
    LinearNDInterpolator,
    interp1d)
from twain_wifco.interface import (
    Ambient,
    Control,
    ModelOutput,
    DataPoint,
    DataTable,
    DataType)

class ScatteredInterpolatorType(Enum):
    RBF = "rbf"
    LINEAR = "linear"

class ScatteredInterpolatorParams(Generic[DataType]):
    def __init__(self,
                 scattered_interp_type: ScatteredInterpolatorType,
                 support_data: DataTable[DataType],
                 out_data: DataTable[DataType]):
        self.scattered_interp_type = scattered_interp_type
        self.support_data = support_data
        self.out_data = out_data
        
InDataType = TypeVar("InDataType", Control, Ambient)
OutDataType = TypeVar("OutDataType", bound=ModelOutput)

def scattered_interpolator_params_from_dict(param_dict: Dict[str, Dict | Any],
                                            support_data_type: Type[InDataType],
                                            out_data_type: Type[OutDataType]) -> ScatteredInterpolatorParams:
    scattered_interp_type = ScatteredInterpolatorType(param_dict["scattered_interp_type"])
    support_data = DataTable({support_data_type(in_var): np.array(supp) for \
                              in_var, supp in param_dict["support_data"].items()})
    out_data = DataTable({out_data_type(out_var): np.array(out_vals) for \
                          out_var, out_vals in param_dict["out_data"].items()})
    
    return ScatteredInterpolatorParams(
        scattered_interp_type=scattered_interp_type,
        support_data=support_data,
        out_data=out_data)
        
class ScatteredInterpolator:
    def __init__(self,                 
                 interpolator_params: ScatteredInterpolatorParams):
        self.scattered_interp_type = interpolator_params.scattered_interp_type
        
        self.support_data = interpolator_params.support_data
        self.out_data = interpolator_params.out_data
        # Set out data point to arbitrary value of the right form
        self.out_data_point = interpolator_params.out_data.get_point(0)
        
        if self.scattered_interp_type == ScatteredInterpolatorType.RBF:
            self.interpolator = RBFInterpolator(
                y=self.support_data.to_matrix(),
                d=self.out_data.to_matrix()
            )
        elif self.scattered_interp_type == ScatteredInterpolatorType.LINEAR:
            if self.support_data.to_matrix().shape[1] == 1:                
                sort_index = np.argsort(self.support_data.to_matrix().T[0])
                self.interpolator = interp1d(x=self.support_data.to_matrix()[sort_index, :].flatten(),
                                             y=self.out_data.to_matrix()[sort_index, :].T,
                                             fill_value="extrapolate")
            else:
                self.interpolator = LinearNDInterpolator(
                    points=self.support_data.to_matrix(),
                    values=self.out_data.to_matrix()
                )
        else:
            raise NotImplementedError("ScatteredDataInterpolator: Only RBF type implemented.")
        pass

    def evaluate_to_vector(self,
                        query: DataPoint[InDataType]) -> DataPoint[OutDataType]:
        x = query.to_vector()
        return self.interpolator(x).flatten()
        
    def evaluate(self,
                 query: DataPoint[InDataType]) -> DataPoint[OutDataType]:
        result = self.evaluate_to_vector(query=query)
        self.out_data_point.from_vector(result)
        return self.out_data_point