from typing import Dict, Any, List
from abc import abstractmethod
import numpy as np
from scipy.optimize import NonlinearConstraint
from enum import Enum
from twain_wifco.interface import (
    Component,
    ComponentParams,
    Control,
    ModelOutput,
    Aggregated,
    AccumulatedMetric,
    DataVariable)

def numeric_bounds(bounds: List[float]):
    bounds_array = [bounds[0] if bounds[0] is not None else -np.inf,
            bounds[1] if bounds[1] is not None else np.inf]
    if bounds_array[0] == -np.inf and bounds_array[1] == np.inf:
        return None
    else:
        return bounds_array

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
                 constr_vars: Dict[DataVariable, float]) -> ConstraintEval:
        
        self._validate_inputs(inputs=constr_vars.keys())

        return self._evaluate(constr_variables=constr_vars)
    
    @abstractmethod
    def _evaluate(self,
                  constr_variables: Dict[DataVariable, float]) -> ConstraintEval:
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
                 constraint_variables: List[DataVariable],
                 bounds: np.ndarray):
        self.constraint_variables = constraint_variables
        self.bounds = bounds
        
    def input_variables(self):
        return set(self.constraint_variables)
    
    def output_variables(self):
        return set()

def separate_linear_constraints_params_from_dict(param_dict: Dict[str, Dict | Any]):
    constraint_mappings = {}
    constraint_mappings = param_dict["constraint_mappings"]
    constraint_variables = []
    bounds = []
    # control constraints
    for ctrl_var, scalar_bounds in constraint_mappings.get("control", {}).items():
        scalar_bounds_array = numeric_bounds(scalar_bounds)
        if scalar_bounds_array is not None:
            constraint_variables.append(Control(ctrl_var))
            bounds.append(scalar_bounds_array)
    # model output bounds
    for model_output_var, scalar_bounds in constraint_mappings.get("model_output", {}).items():
        scalar_bounds_array = numeric_bounds(scalar_bounds)
        if scalar_bounds_array is not None:
            constraint_variables.append(ModelOutput(model_output_var))
            bounds.append(scalar_bounds_array)
    # aggregate bounds
    for aggregate_var, scalar_bounds in constraint_mappings.get("aggregate", {}).items():
        scalar_bounds_array = numeric_bounds(scalar_bounds)
        if scalar_bounds_array is not None:
            constraint_variables.append(Aggregated(aggregate_var))
            bounds.append(scalar_bounds_array)
    # accumulated metric bounds
    for acc_metric_var, scalar_bounds in constraint_mappings.get("accumulated_metric", {}).items():
        scalar_bounds_array = numeric_bounds(scalar_bounds)
        if scalar_bounds_array is not None:
            constraint_variables.append(AccumulatedMetric(acc_metric_var))
            bounds.append(scalar_bounds_array)
    scalar_bounds_array = np.array(bounds)
         
    return SeparateLinearConstraintsParams(
        constraint_variables=constraint_variables,
        bounds=scalar_bounds_array)

class SeparateLinearConstraints(Constraint):
    def __init__(self,
                 constraint_name: str,
                 constraint_params: SeparateLinearConstraintsParams):
        super().__init__(constraint_name=constraint_name,
                         constraint_params=constraint_params)
        self.constraint_variables = constraint_params.constraint_variables
        self.bounds = constraint_params.bounds

    def scalar_bounds_for_var(self, var: DataVariable):
        ind = self.constraint_variables.index(var)
        return self.bounds[ind, :]

    def _evaluate(self,
                  constr_input_values: Dict[DataVariable, float]) -> ConstraintEval:
        lower_diff = np.empty(len(self.constraint_variables))
        upper_diff = np.empty(len(self.constraint_variables))
        for i, var in enumerate(self.constraint_variables):
            lower_diff[i] = constr_input_values[var] - self.bounds[i, 0]
            upper_diff[i] = constr_input_values[var] - self.bounds[i, 1]
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
