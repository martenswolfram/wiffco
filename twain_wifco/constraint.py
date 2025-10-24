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
    Interface)

class UpperBoundConstraint:
    def __init__(self,
                 upper_bound: np.ndarray,
                 constraint_fun):
        self.upper_bound = upper_bound
        self.constraint_fun = constraint_fun

class TwoSidedConstraint(Component, Generic[DataType]):
    def __init__(self,
                 constraint_name: str,
                 constraint_params: ComponentParams):
        super().__init__(component_name=constraint_name,
                         component_params=constraint_params)
        
    def evaluate_satisfied(self,
                 constr_input_data: DataPoint[DataType]) -> bool:

        self.validate_inputs(input_data={constr_input_data.data_type: constr_input_data})

        return self._evaluate_satisfied(constr_input_data=constr_input_data)
    
    @abstractmethod
    def _evaluate_satisfied(self,
                  constr_input_data: DataPoint[DataVariable]) -> bool:
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
        self.variables = upper_bound.keys()
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

class SeparateLinearConstraintsParams(ComponentParams):
    def __init__(self,
                 bounds: TwoSidedSeparateBounds):
        self.bounds = bounds
        
    def input_interface(self) -> Interface:
        return Interface(all_shapes={
            self.bounds.data_type: self.bounds.upper_bound.shapes()
            })
        
    def output_interface(self):
        return Interface(all_shapes={})

def separate_linear_constraints_params_from_dict(param_dict: Dict[str, Dict | Any]):
    
    if param_dict["data_type"] == "control":
        data_type = Control
    elif param_dict["data_type"] == "aggregate":
        data_type = Aggregated
    elif param_dict["data_type"] == "accumulated_metric":
        data_type = AccumulatedMetric

    upper_bound_data = {}
    lower_bound_data = {}
    upper_bound_params = param_dict["upper_bound"]
    lower_bound_params = param_dict["lower_bound"]
    for constr_var in upper_bound_params.keys():
        upper_bound = \
            np.array([u if u is not None else np.inf for u in upper_bound_params[constr_var]])
        lower_bound = \
            np.array([l if l is not None else -np.inf for l in lower_bound_params[constr_var]])
        # ignore null constraints
        if all(np.isposinf(upper_bound)) and all(np.isneginf(lower_bound)):
            continue
        upper_bound_data[data_type(constr_var)] = upper_bound
        lower_bound_data[data_type(constr_var)] = lower_bound
            
    bounds = TwoSidedSeparateBounds(
        upper_bound=DataPoint(data=upper_bound_data),
        lower_bound=DataPoint(data=lower_bound_data)
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
        
    def _evaluate_satisfied(self,
                  constr_input_data: DataPoint[DataVariable]) -> bool:
        relevant_constr_input_data = DataPoint(
            {constr_var: constr_data for \
             constr_var, constr_data in constr_input_data.data.items() \
                if constr_var in self.bounds.variables})

        lower_diff = (relevant_constr_input_data - self.bounds.lower_bound).to_vector()
        upper_diff = (relevant_constr_input_data - self.bounds.upper_bound).to_vector()
        
        return all(lower_diff >= - self.bounds.lower_bound.abs_tols_vec()) and \
            all(upper_diff <= self.bounds.upper_bound.abs_tols_vec())

    # TODO: In the following functions, make sure that the order of the variables is safe!!
    def scipy_constraint(self, eval_constraint_from_x):
        
        def eval_nl_constraints(x):
            # TODO: should eval_constraint_from_x be vector-to-vector? 
            constraints_eval = eval_constraint_from_x(x)
            result = np.concatenate(list(constraints_eval[constr_var] for \
                                    constr_var in self.bounds.upper_bound.order))
            return result
        
        return NonlinearConstraint(fun=eval_nl_constraints,
                                   lb=self.bounds.lower_bound.to_vector(),
                                   ub=self.bounds.upper_bound.to_vector())
    
    def upper_bound_constraints(self) -> List[UpperBoundConstraint]:
        ub_constraints = []
        for constr_var in self.bounds.upper_bound.keys():
            # lower bound
            if self.bounds.lower_bound[constr_var] > -np.inf:
                def constraint_fun(constr_var_values, constr_var=constr_var):
                    return -constr_var_values[constr_var]
                ub_constraint = UpperBoundConstraint(upper_bound=-self.bounds.lower_bound[constr_var],
                                                     constraint_fun=constraint_fun)
                ub_constraints.append(ub_constraint)
            # upper bound
            if self.bounds.upper_bound[constr_var] < np.inf:
                def constraint_fun(constr_var_values, constr_var=constr_var):
                    return constr_var_values[constr_var]
                ub_constraint = UpperBoundConstraint(upper_bound=self.bounds.upper_bound[constr_var],
                                                    constraint_fun=constraint_fun)
                ub_constraints.append(ub_constraint)
        return ub_constraints
