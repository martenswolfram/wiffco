from typing import Dict, Any, Set
from abc import abstractmethod
import numpy as np
from enum import Enum
from twain_wifco.interface import (
    Component,
    ComponentParams,
    AccumulatedMetric)

class ConstraintEval:
    def __init__(self,
                 lower_diff: float | None,
                 upper_diff: float | None):
        self.lower_diff = lower_diff
        self.upper_diff = upper_diff

class AccumulatedConstraint(Component):
    def __init__(self,
                 acc_constraint_name: str,
                 acc_constraint_params: ComponentParams):
        super().__init__(component_name=acc_constraint_name,
                         component_params=acc_constraint_params)

    def evaluate(self,
                 acc_metrics: Dict[AccumulatedMetric, float]):
        
        self._validate_inputs(inputs=acc_metrics.keys())

        return self._evaluate(acc_metrics=acc_metrics)
            
    @abstractmethod
    def _evaluate(self,
                  acc_metrics: Dict[AccumulatedMetric, float]):
        pass

class AccumulatedConstraintType(Enum):
    SEPARATE_LINEAR_CONSTRAINTS = "separate_linear_constraints"

class SeparateLinearConstraintMapping:
    def __init__(self,
                 upper_bound: float | None,
                 lower_bound: float | None):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

class SeparateLinearConstraintsParams(ComponentParams):
    def __init__(self,
                 constraint_mappings: Dict[AccumulatedMetric, SeparateLinearConstraintMapping]):
        self.constraint_mappings = constraint_mappings
        
    def input_variables(self):
        return self.constraint_mappings.keys()
    
    def output_variables(self):
        return set()

def separate_linear_constraints_params_from_dict(param_dict: Dict[str, Dict | Any]):
    constraint_mappings = {}
    for acc_metric, constr_mapping in param_dict["constraint_mappings"].items():
        constraint_mapping = SeparateLinearConstraintMapping(
             upper_bound=constr_mapping["upper_bound"],
             lower_bound=constr_mapping["lower_bound"])
        if (constraint_mapping.upper_bound is not None) and \
            (constraint_mapping.lower_bound is not None) and \
            (constraint_mapping.upper_bound < constraint_mapping.lower_bound):
                raise ValueError("SeparateLinearConstraintsParams config: Upper bound must be greater than or equal to lower bound.")
        if (constraint_mapping.upper_bound is None) and (constraint_mapping.lower_bound is None):
            # Can be ignored
            continue
        constraint_mappings[AccumulatedMetric(acc_metric)] = constraint_mapping
         
    return SeparateLinearConstraintsParams(constraint_mappings=constraint_mappings)

class SeparateLinearConstraints(AccumulatedConstraint):
    def __init__(self,
                 acc_constraint_name: str,
                 acc_constraint_params: SeparateLinearConstraintsParams):
        super().__init__(acc_constraint_name=acc_constraint_name,
                         acc_constraint_params=acc_constraint_params)
        self.constraint_mappings = acc_constraint_params.constraint_mappings

    def _evaluate(self,
                  acc_metrics: Dict[AccumulatedMetric, float]):
        constraint_evals = {}
        for acc_metric, value in acc_metrics.items():
            if self.constraint_mappings[acc_metric].lower_bound is not None:
                lower_diff = value - self.constraint_mappings[acc_metric].lower_bound
            else:
                lower_diff = None
            if self.constraint_mappings[acc_metric].upper_bound is not None:
                upper_diff = value - self.constraint_mappings[acc_metric].upper_bound
            else:
                upper_diff = None
            constraint_evals[acc_metric] = ConstraintEval(lower_diff=lower_diff,
                                                          upper_diff=upper_diff)
        return constraint_evals
