from typing import Dict, List, Callable, Tuple
import numpy as np
from scipy.optimize import NonlinearConstraint, Bounds
from dataclasses import dataclass
from twain_wifco.interface import (
    Component,
    Control,
    Aggregate,
    AccumulatedMetric,
    get_abs_tol,
    MAP_STR_TO_ENUM,
    DataType,
    DataTable,
    Interface
)

@dataclass
class TwoSidedBound:
    lower: float = None
    upper: float = None
        
    def __post_init__(self):
        if self.lower is None:
            self.lower = -np.inf
        if self.upper is None:
            self.upper = np.inf

class Constraint(Component):
    
    def __init__(self,
                 name: str,
                 two_sided_bounds: Dict[DataType, TwoSidedBound]):
        self.component_name = name
        self._bounds = two_sided_bounds
        self._var_order = list(self._bounds.keys())
        # input_interface is defined in child classes
        self.output_interface = Interface()

    def evaluate_satisfied(self,
                           constraint_input: DataTable[DataType],
                           ) -> bool:
        
        self.validate_input([constraint_input])
        constraints_satisfied = np.full(shape=(len(constraint_input),), fill_value=True)
        for var, bound in self._bounds.items():
            abs_tol = get_abs_tol(data_var=var)
            trailing_axes = tuple(range(1, constraint_input[var].ndim))
            constraints_satisfied &= np.all(bound.lower < constraint_input[var] + abs_tol,
                                            axis=trailing_axes)
            constraints_satisfied &= np.all(constraint_input[var] < bound.upper + abs_tol,
                                            axis=trailing_axes)
            
        return np.where(constraints_satisfied)[0]
    
    def get_flat_bounds(self,
                        var_shapes_dict: Dict[DataType, Tuple[int, ...]],
                        num_points: int,
                        var_order: List[DataType] | None = None):
        if var_order is None:
            var_order = self._var_order
        flat_lower_bounds = [[self._bounds[var].lower] * \
                             np.prod(var_shapes_dict[var]) for \
                             var in var_order]
        flat_upper_bounds = [[self._bounds[var].upper] * \
                             np.prod(var_shapes_dict[var]) for \
                             var in var_order]
        lb = np.concatenate(flat_lower_bounds * num_points)
        ub = np.concatenate(flat_upper_bounds * num_points)
        return lb, ub
    
    def bounds(self) -> Dict[DataType, TwoSidedBound]:
        return self._bounds

class ControlConstraint(Constraint):

    def __init__(self,
                 name: str,
                 two_sided_bounds: Dict[Control, TwoSidedBound]):
        super().__init__(name=name,
                         two_sided_bounds=two_sided_bounds)
        
        self.input_interface = Interface(all_shapes={
            Control: {
                var: None for var in two_sided_bounds
            }
        })

    def scipy_bounds(self,
                     control_order: List[Control],
                     control_shapes_dict: Dict[Control, Tuple[int, ...]],
                     num_points: int):
        
        lb, ub = self.get_flat_bounds(var_shapes_dict=control_shapes_dict,
                                      num_points=num_points,
                                      var_order=control_order)
        return Bounds(lb=lb, ub=ub)

class AggregateConstraint(Constraint):

    def __init__(self,
                 name: str,
                 two_sided_bounds: Dict[Aggregate, TwoSidedBound]):
        super().__init__(name=name,
                         two_sided_bounds=two_sided_bounds)
        
        self.input_interface = Interface(all_shapes={
            Aggregate: {
                var: None for var in two_sided_bounds
            }
        })

    def scipy_constraint(self,
                         aggregate_shapes_dict: Dict[Aggregate, Tuple[int, ...]],
                         aggregate_evaluation: Callable,
                         num_points: int):
        
        def eval_constraints(x):
            aggregate_table: DataTable[Aggregate] = aggregate_evaluation(x)
            return aggregate_table.to_vector(order=self._var_order) 
        lb, ub = self.get_flat_bounds(var_shapes_dict=aggregate_shapes_dict,
                                      num_points=num_points)
        return NonlinearConstraint(fun=eval_constraints, lb=lb, ub=ub)

class AccumulatedConstraint(Constraint):

    def __init__(self,
                 name: str,
                 two_sided_bounds: Dict[AccumulatedMetric, TwoSidedBound]):
        super().__init__(name=name,
                         two_sided_bounds=two_sided_bounds)
        
        self.input_interface = Interface(all_shapes={
            AccumulatedMetric: {
                var: None for var in two_sided_bounds
            }
        })

    def scipy_constraint(self,
                         accumulated_order: List[Aggregate],
                         accumulated_shapes_dict: Dict[Aggregate, Tuple[int, ...]],
                         accumulated_evaluation: Callable):
        
        def eval_constraints(x):
            accumulated_table: DataTable[AccumulatedMetric] = accumulated_evaluation(x)
            return accumulated_table.to_vector(order=self._var_order) 
        lb, ub = self.get_flat_bounds(var_shapes_dict=accumulated_shapes_dict,
                                      num_points=1)
        return NonlinearConstraint(fun=eval_constraints, lb=lb, ub=ub)

def constraint_from_dict(param_dict: Dict[str, str | Dict[str, Dict]]) -> Constraint:
    """Create Constraint from a parameter dictionary.

    """
    name = param_dict["name"]
    two_sided_bounds = {}
    
    data_enum = MAP_STR_TO_ENUM.get(param_dict["constraint_type"], None)
    if data_enum not in {Control, Aggregate, AccumulatedMetric}:
        raise ValueError(f"Invalid constraint type: {param_dict['constraint_type']}.")
    
    for var_str, bounds in param_dict["bounds"].items():
        lb = bounds.get("lower", None)
        ub = bounds.get("upper", None)
        if lb is not None or ub is not None:
            two_sided_bounds[data_enum(var_str)] = TwoSidedBound(lower=lb, upper=ub)
    
    if data_enum is Control:
        return ControlConstraint(name=name,
                                    two_sided_bounds=two_sided_bounds)
    elif data_enum is Aggregate:
        return AggregateConstraint(name=name,
                                   two_sided_bounds=two_sided_bounds)
    else:
        return AccumulatedConstraint(name=name,
                                     two_sided_bounds=two_sided_bounds)