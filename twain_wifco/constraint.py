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
    DataVariable,
    retrieve_single_key_str)

def numeric_bounds_array(lower_bounds: List[float], upper_bounds: List[float]):
    lower_bounds_numeric = [lb if lb is not None else -np.inf for lb in lower_bounds]
    upper_bounds_numeric = [ub if ub is not None else np.inf for ub in upper_bounds]
    return np.array([lower_bounds_numeric, upper_bounds_numeric]).T

class UpperBound:
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
        
        self._validate_inputs(inputs=constr_values_dict.keys())

        return self._evaluate(constr_values_dict=constr_values_dict)
    
    @abstractmethod
    def _evaluate(self,
                  constr_values_dict: Dict[DataVariable, np.ndarray]) -> ConstraintEval:
        pass

    @abstractmethod
    def scipy_constraint(self, eval_constraint_from_x):
        pass

    @abstractmethod
    def upper_bound_constraints(self) -> List[UpperBound]:
        pass


class ConstraintType(Enum):
    SEPARATE_LINEAR_CONSTRAINTS = "separate_linear_constraints"

class SeparateLinearConstraintsParams(ComponentParams):
    def __init__(self,
                 constraint_variables: List[ConstraintVar],
                 bounds: np.ndarray):
        self.constraint_variables = constraint_variables
        self.bounds = bounds
        
    def input_variables(self):
        return set(self.constraint_variables)
    
    def output_variables(self):
        return set()

def separate_linear_constraints_params_from_dict(param_dict: Dict[str, Dict | Any]):
    
    key_str = retrieve_single_key_str(param_dict, set(["control_variables",
                                                       "aggregated_variables",
                                                       "accumulated_metrics"]))
    if key_str == "control_variables":
        constraint_variables = \
            [Control(constr_var) for constr_var in param_dict[key_str]]
    elif key_str == "aggregated_variables":
        constraint_variables = \
            [Aggregated(constr_var) for constr_var in param_dict[key_str]]
    else:
        constraint_variables = \
            [AccumulatedMetric(constr_var) for constr_var in param_dict[key_str]]
    
    bounds = numeric_bounds_array(lower_bounds=param_dict["lower_bounds"],
                                  upper_bounds=param_dict["upper_bounds"])
    null_row = np.array([-np.inf, np.inf])
    null_indices = np.where(np.all(bounds == null_row, axis=1))[0]
    
    # Remove null-constraints
    constraint_variables = [constr_var for ind, constr_var in enumerate(constraint_variables) if ind not in null_indices]
    bounds = np.delete(bounds, null_indices, axis=0)
    return SeparateLinearConstraintsParams(
        constraint_variables=constraint_variables,
        bounds=bounds)

class SeparateLinearConstraints(Constraint):
    def __init__(self,
                 constraint_name: str,
                 constraint_params: SeparateLinearConstraintsParams):
        super().__init__(constraint_name=constraint_name,
                         constraint_params=constraint_params)
        self.constraint_variables = constraint_params.constraint_variables
        self.bounds = constraint_params.bounds

    def scalar_bounds_for_var(self, var: ConstraintVar):
        ind = self.constraint_variables.index(var)
        return self.bounds[ind, :]

    def _evaluate(self,
                  constr_values_dict: Dict[DataVariable, np.ndarray]) -> ConstraintEval:
        lower_diff = np.empty(len(self.constraint_variables))
        upper_diff = np.empty(len(self.constraint_variables))
        for i, var in enumerate(self.constraint_variables):
            lower_diff[i] = constr_values_dict[var] - self.bounds[i, 0]
            upper_diff[i] = constr_values_dict[var] - self.bounds[i, 1]
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
    
    def upper_bound_constraints(self) -> List[UpperBound]:
        ub_constraints = []
        for constr_var, scalar_bounds in zip(self.constraint_variables, self.bounds):
            # lower bound
            if scalar_bounds[0] > -np.inf:
                def constraint_fun(constr_var_values, constr_var=constr_var):
                    return -constr_var_values[constr_var]
                ub_constraint = UpperBound(upper_bound=-scalar_bounds[0],
                                           constraint_fun=constraint_fun)
                ub_constraints.append(ub_constraint)
            # upper bound
            if scalar_bounds[1] < np.inf:
                def constraint_fun(constr_var_values, constr_var=constr_var):
                    return constr_var_values[constr_var]
                ub_constraint = UpperBound(upper_bound=scalar_bounds[1],
                                           constraint_fun=constraint_fun)
                ub_constraints.append(ub_constraint)
        return ub_constraints
