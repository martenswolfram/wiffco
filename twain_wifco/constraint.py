from typing import Dict, Any, List, Type, Generic
from abc import abstractmethod
import numpy as np
from scipy.optimize import NonlinearConstraint
from enum import Enum
from twain_wifco.interface import (
    Component,
    ComponentParams,
    Control,
    Aggregated,
    AccumulatedMetric,
    ConstraintType,
    DataVariable,
    DataType,
    DataPoint,
    DataTable,
    Interface)

class UpperBoundConstraint:
    def __init__(self,
                 upper_bound: np.ndarray,
                 constraint_fun):
        self.upper_bound = upper_bound
        self.constraint_fun = constraint_fun

class TwoSidedConstraintEval:
    def __init__(self,
                 lower_diff: np.ndarray,
                 upper_diff: np.ndarray):
        self.lower_diff = lower_diff
        self.upper_diff = upper_diff

    def satisfied(self):
        return all(self.lower_diff >= 0) and \
            all(self.upper_diff <= 0)

class TwoSidedConstraint(Component, Generic[DataType]):
    def __init__(self,
                 constraint_name: str,
                 constraint_params: ComponentParams):
        super().__init__(component_name=constraint_name,
                         component_params=constraint_params)
        
    def evaluate(self,
                 constr_input_data: DataPoint[DataType]) -> TwoSidedConstraintEval:

        if constr_input_data.data_type == Control:
            self.validate_inputs(control=constr_input_data)
        elif constr_input_data.data_type == Aggregated:
            self.validate_inputs(aggregated=constr_input_data)
        else: 
            self.validate_inputs(accumulated_metric=constr_input_data)

        return self._evaluate(constr_input_data=constr_input_data)
    
    @abstractmethod
    def _evaluate(self,
                  constr_input_data: DataPoint[DataVariable]) -> TwoSidedConstraintEval:
        pass

    @abstractmethod
    def scipy_constraint(self, eval_constraint_from_x):
        pass

    @abstractmethod
    def upper_bound_constraints(self) -> List[UpperBoundConstraint]:
        pass


class ConstraintType(Enum):
    SEPARATE_LINEAR_CONSTRAINTS = "separate_linear_constraints"

class TwoSidedSeparateBounds:
    def __init__(self,
                 upper_bound: DataPoint[ConstraintType],
                 lower_bound: DataPoint[ConstraintType]):
        self.data_type = upper_bound.data_type
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

class SeparateLinearConstraintsParams(ComponentParams):
    def __init__(self,
                 bounds: TwoSidedSeparateBounds):
        self.bounds = bounds
        
    def input_interface(self) -> Interface:
        if self.bounds.data_type == Control:
            return Interface(control_shapes=self.bounds.upper_bound.shapes())
        elif self.bounds.data_type == Aggregated:
            return Interface(aggregated_shapes=self.bounds.upper_bound.shapes())
        else:
            return Interface(accumulated_metric_shapes=self.bounds.upper_bound.shapes())
        
    def output_interface(self):
        return Interface()

def separate_linear_constraints_params_from_dict(param_dict: Dict[str, Dict | Any]):
    
    if param_dict["data_type"] == "control":
        data_type = Control
    elif param_dict["data_type"] == "aggregate":
        data_type = Aggregated
    elif param_dict["data_type"] == "accumulated_metric":
        data_type = AccumulatedMetric

    upper_bound = {}
    for constr_var, ub in param_dict["upper_bound"].items():
        upper_bound[data_type(constr_var)] = \
            np.array([u if u is not None else np.inf for u in ub])
    lower_bound = {}
    for constr_var, lb in param_dict["lower_bound"].items():
        lower_bound[data_type(constr_var)] = \
            np.array([l if l is not None else -np.inf for l in lb])
    bounds = TwoSidedSeparateBounds(
        upper_bound=DataPoint(data=upper_bound),
        lower_bound=DataPoint(data=lower_bound)
    )
    return SeparateLinearConstraintsParams(
        bounds=bounds)

class SeparateLinearConstraints(TwoSidedConstraint):
    def __init__(self,
                 constraint_name: str,
                 constraint_params: SeparateLinearConstraintsParams):
        super().__init__(constraint_name=constraint_name,
                         constraint_params=constraint_params)
        self.bounds = constraint_params.bounds
        
    def _evaluate(self,
                  constr_input_data: DataPoint[DataVariable]) -> TwoSidedConstraintEval:
        
        lower_diff = (constr_input_data - self.bounds.lower_bound).to_vector()
        upper_diff = (constr_input_data - self.bounds.upper_bound).to_vector()
        return TwoSidedConstraintEval(lower_diff=lower_diff,
                                      upper_diff=upper_diff)

    # TODO: In the following functions, make sure that the order of the variables is safe!!
    def scipy_constraint(self, eval_constraint_from_x):
        
        def eval_nl_constraints(x):
            # TODO: should eval_constraint_from_x be vector-to-vector? 
            constraints_eval = eval_constraint_from_x(x)
            return np.array(list(constraints_eval[constr_var] for \
                                 constr_var in self.bounds.upper_bound.order))
        
        return NonlinearConstraint(fun=eval_nl_constraints,
                                   lb=self.bounds.upper_bound,
                                   ub=self.bounds.lower_bound)
    
    def upper_bound_constraints(self) -> List[UpperBoundConstraint]:
        ub_constraints = []
        for constr_var in self.bounds.upper_bound.keys():
            # lower bound
            def constraint_fun(constr_var_values, constr_var=constr_var):
                return -constr_var_values[constr_var]
            ub_constraint = UpperBoundConstraint(upper_bound=-self.bounds.lower_bound[constr_var],
                                                 constraint_fun=constraint_fun)
            ub_constraints.append(ub_constraint)
            # upper bound
            def constraint_fun(constr_var_values, constr_var=constr_var):
                return constr_var_values[constr_var]
            ub_constraint = UpperBoundConstraint(upper_bound=self.bounds.lower_bound[constr_var],
                                                 constraint_fun=constraint_fun)
            ub_constraints.append(ub_constraint)
        return ub_constraints
