from typing import Dict, Any, List
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
    ConstraintVar,
    DataVariable)
from twain_wifco.utils import (
    hstacked_from_dict
)

class Bounds:
    def __init__(self,
                 upper_bound: Dict[ConstraintVar, np.ndarray],
                 lower_bound: Dict[ConstraintVar, np.ndarray]):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

class UpperBoundConstraint:
    def __init__(self,
                 upper_bound: np.ndarray,
                 constraint_fun):
        self.upper_bound = upper_bound
        self.constraint_fun = constraint_fun

class ConstraintEval:
    def __init__(self,
                 lower_diff: np.ndarray,
                 upper_diff: np.ndarray):
        self.lower_diff = lower_diff
        self.upper_diff = upper_diff

    def satisfied(self):
        return all(self.lower_diff >= 0) and \
            all(self.upper_diff <= 0)

class Constraint(Component):
    def __init__(self,
                 constraint_name: str,
                 constraint_params: ComponentParams):
        super().__init__(component_name=constraint_name,
                         component_params=constraint_params)

    def evaluate(self,
                 constr_values_dict: Dict[DataVariable, np.ndarray]) -> ConstraintEval:
        
        self.validate_inputs(inputs=constr_values_dict)

        return self._evaluate(constr_values_dict=constr_values_dict)
    
    @abstractmethod
    def _evaluate(self,
                  constr_values_dict: Dict[DataVariable, np.ndarray]) -> ConstraintEval:
        pass

    @abstractmethod
    def scipy_constraint(self, eval_constraint_from_x):
        pass

    @abstractmethod
    def upper_bound_constraints(self) -> List[UpperBoundConstraint]:
        pass


class ConstraintType(Enum):
    SEPARATE_LINEAR_CONSTRAINTS = "separate_linear_constraints"

class SeparateLinearConstraintsParams(ComponentParams):
    def __init__(self,
                 bounds: Bounds):
        self.bounds = bounds
        
    def input_variables(self):
        return {var: len(ub) for var, ub in self.bounds.upper_bound.items()}
    
    def output_variables(self):
        return {}

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
    bounds = Bounds(
        upper_bound=upper_bound,
        lower_bound=lower_bound
    )
    return SeparateLinearConstraintsParams(
        bounds=bounds)

class SeparateLinearConstraints(Constraint):
    def __init__(self,
                 constraint_name: str,
                 constraint_params: SeparateLinearConstraintsParams):
        super().__init__(constraint_name=constraint_name,
                         constraint_params=constraint_params)
        # self.bounds = constraint_params.bounds
        self.constraint_variables = list(constraint_params.bounds.upper_bound.keys())
        self.stacked_lower_bound = hstacked_from_dict(
            data_dict=constraint_params.bounds.lower_bound,
            variables=self.constraint_variables)
        self.stacked_upper_bound = hstacked_from_dict(
            data_dict=constraint_params.bounds.upper_bound,
            variables=self.constraint_variables)
        
        
    def _evaluate(self,
                  constr_values_dict: Dict[DataVariable, np.ndarray]) -> ConstraintEval:
        
        stacked_values = hstacked_from_dict(
            data_dict=constr_values_dict,
            variables=self.constraint_variables)
        lower_diff = stacked_values - self.stacked_lower_bound
        upper_diff = stacked_values - self.stacked_upper_bound
        return ConstraintEval(lower_diff=lower_diff,
                              upper_diff=upper_diff)

    def scipy_constraint(self, eval_constraint_from_x):
        
        def eval_nl_constraints(x):
            constraints_eval = eval_constraint_from_x(x)
            return np.array(list(constraints_eval[constr_var] for \
                                 constr_var in self.constraint_variables))
        
        return NonlinearConstraint(fun=eval_nl_constraints,
                                   lb=self.bounds[:, 0],
                                   ub=self.bounds[:, 1])
    
    def upper_bound_constraints(self) -> List[UpperBoundConstraint]:
        ub_constraints = []
        for constr_var, scalar_bounds in zip(self.constraint_variables, self.bounds):
            # lower bound
            if scalar_bounds[0] > -np.inf:
                def constraint_fun(constr_var_values, constr_var=constr_var):
                    return -constr_var_values[constr_var]
                ub_constraint = UpperBoundConstraint(upper_bound=-scalar_bounds[0],
                                           constraint_fun=constraint_fun)
                ub_constraints.append(ub_constraint)
            # upper bound
            if scalar_bounds[1] < np.inf:
                def constraint_fun(constr_var_values, constr_var=constr_var):
                    return constr_var_values[constr_var]
                ub_constraint = UpperBoundConstraint(upper_bound=scalar_bounds[1],
                                           constraint_fun=constraint_fun)
                ub_constraints.append(ub_constraint)
        return ub_constraints
