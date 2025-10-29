from typing import Dict, Any, List, Generic, Callable, Tuple
from abc import abstractmethod
import numpy as np
from scipy.optimize import NonlinearConstraint, Bounds
from enum import Enum
from twain_wifco.interface import (
    Component,
    data_type_from_string,
    DataVariable,
    DataType,
    DataPoint,
    DataTable,
    Interface
)


class UpperBoundConstraint:
    """Represents a single upper-bound constraint.

    Attributes:
        upper_bound (np.ndarray): Upper bound values for the constraint.
        constraint_fun (Callable): Function that evaluates the constraint.
    """
    def __init__(self,
                 upper_bound: np.ndarray,
                 constraint_fun: Callable[[Dict[DataVariable, np.ndarray]], np.ndarray]):
        self.upper_bound = upper_bound
        self.constraint_fun = constraint_fun

class TwoSidedBounds:
    """Container for upper and lower bounds for multiple variables of the same type.

    Attributes:
        upper_bound (DataPoint[DataType]): Upper bounds.
        lower_bound (DataPoint[DataType]): Lower bounds.
        data_type (Type[DataType]): Type of data stored.
    """
    def __init__(self,
                 upper_bound: DataPoint[DataType],
                 lower_bound: DataPoint[DataType]):
        self.data_type = upper_bound.data_type
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.order = upper_bound.order

class Constraint(Component, Generic[DataType]):
    """Abstract base class for all constraints.

    """

    @abstractmethod
    @Component.with_validation
    def evaluate_satisfied(self,
                           constr_input_data: DataPoint[DataType]) -> bool:
        """Check whether the constraint is satisfied for the given input.

        Args:
            constr_input_data (DataPoint[DataType]): Input data to evaluate.

        Returns:
            bool: True if constraint is satisfied, False otherwise.
        """
        ...

    @abstractmethod
    def scipy_object(self,
                     x_order: List[DataType],
                     x_shapes_dict: Dict[DataType, Tuple[int, ...]],
                     num_points: int | None = None,
                     x_constraint_evaluation: Callable | None = None):
        """ Creates SciPy objects for constraint evaluation.

        Args:
            x_order (List[DataType]) : 
            x_shapes_dict (Dict[DataType, Tuple[int, ...]]): 
            num_points (int | None): 
            x_constraint_evaluation (Callable | None): Function to be evaluated on input (decision) variables

        """
        pass
    
    @abstractmethod
    def upper_bound_constraints(self) -> List[UpperBoundConstraint]:
        """Convert all constraints into upper-bound constraints.

        Returns:
            List[UpperBoundConstraint]: List of upper-bound constraints.
        """
        pass

class ConstraintType(Enum):
    SEPARATE_CONSTRAINTS = "separate_constraints"

class SeparateConstraints(Component):
    """Parameter container for SeparateConstraints.

    Attributes:
        bounds (TwoSidedBounds): Upper and lower bounds for the constraint.
    """
    def __init__(self,
                 name: str,
                 two_sided_bounds: TwoSidedBounds):
        self._component_name = name
        self._bounds = two_sided_bounds

        self._input_interface = Interface(all_shapes={
            self._bounds.data_type: self._bounds.upper_bound.shapes()
        })
    
        self._output_interface = Interface()

    @Component.with_validation
    def evaluate_satisfied(self,
                           constr_input_data: DataPoint[DataVariable]) -> bool:
        """Check if all constraints are satisfied for the given input.

        Args:
            constr_input_data (DataPoint[DataVariable]): Input data.

        Returns:
            bool: True if all constraints are satisfied.
        """

        lb = self._bounds.lower_bound.to_vector()
        ub = self._bounds.upper_bound.to_vector()
        
        lower_diff = constr_input_data.to_vector(order=self._bounds.order) - lb
        upper_diff = constr_input_data.to_vector(order=self._bounds.order) - ub
        
        return all(lower_diff >= -self._bounds.lower_bound.abs_tols_vec()) and \
               all(upper_diff <=  self._bounds.upper_bound.abs_tols_vec())

    def scipy_object(self,
                     x_order: List[DataType],
                     x_shapes_dict: Dict[DataType, Tuple[int, ...]],
                     num_points: int | None = None,
                     x_constraint_evaluation: Callable | None = None):
        """ Creates SciPy objects for constraint evaluation"""
        if x_constraint_evaluation is None:
            # If x is only passed through, create a SciPy-Bounds object, based on the external variable order
            # Bounds for unconstrained variables are filled with corresponding (pos/neg) infinite bounds 
            lb = self._bounds.lower_bound.to_vector(order=x_order,
                                                    fill_shapes=x_shapes_dict,
                                                    fill_value=-np.inf,
                                                    batch_multiply=num_points)
            ub = self._bounds.upper_bound.to_vector(order=x_order,
                                                    fill_shapes=x_shapes_dict,
                                                    fill_value=np.inf,
                                                    batch_multiply=num_points)
            return Bounds(lb=lb, ub=ub)
        else:
            # Otherwise create a SciPy-NonlinearConstraint object
            if num_points is None:
                # Single-point evaluation
                def eval_constraint(x):
                    constr_evaluation = x_constraint_evaluation(x)                    
                    return np.concatenate([constr_evaluation[var].ravel() for \
                                           var in self._bounds.order])    
                lb = self._bounds.lower_bound.to_vector(order=self._bounds.order)
                ub = self._bounds.upper_bound.to_vector(order=self._bounds.order)
                return NonlinearConstraint(fun=eval_constraint, lb=lb, ub=ub)
            else:
                # Batch-constraint evaluation
                def eval_constraints(x):
                    # Note: The constraint-evaluation function is interpreted such that 
                    # both input data and output data represents a flat concatenation over 
                    # all points.
                    constr_evaluation: DataTable = x_constraint_evaluation(x)
                    return constr_evaluation.to_vector(order=self._bounds.order) 
                # The bounds are defined for each point, hence need to be tiled/extruded.
                lb = self._bounds.lower_bound.to_vector(order=self._bounds.order,
                                                        batch_multiply=num_points)
                ub = self._bounds.upper_bound.to_vector(order=self._bounds.order,
                                                        batch_multiply=num_points)
                return NonlinearConstraint(fun=eval_constraints, lb=lb, ub=ub)
    
    def upper_bound_constraints(self) -> List[UpperBoundConstraint]:
        """Convert to a list of upper-bound constraints for optimization.

        Returns:
            List[UpperBoundConstraint]: List of upper-bound constraints.
        """
        ub_constraints = []

        for var in self._bounds.upper_bound.keys():
            # Lower bound as upper-bound constraint
            if (self._bounds.lower_bound[var] > -np.inf).any():
                def constraint_fun(values, v=var):
                    return -values[v]
                ub_constraints.append(UpperBoundConstraint(
                    upper_bound=-self._bounds.lower_bound[var],
                    constraint_fun=constraint_fun
                ))

            # Upper bound as upper-bound constraint
            if (self._bounds.upper_bound[var] < np.inf).any():
                def constraint_fun(values, v=var):
                    return values[v]
                ub_constraints.append(UpperBoundConstraint(
                    upper_bound=self._bounds.upper_bound[var],
                    constraint_fun=constraint_fun
                ))

        return ub_constraints

def separate_constraints_from_dict(param_dict: Dict[str, Any]) -> SeparateConstraints:
    """Create SeparateConstraints from a parameter dictionary.

    Args:
        param_dict (Dict[str, Any]): Dictionary containing upper/lower bounds and data type.

    Returns:
        SeparateConstraints: Constructed Separate Constraints object.
    """
    name = param_dict["name"]
    data_type = data_type_from_string(param_dict["data_type"])

    upper_bound_data = {}
    lower_bound_data = {}
    upper_bound_params = param_dict["upper_bound"]
    lower_bound_params = param_dict["lower_bound"]

    for constr_var in upper_bound_params.keys():
        upper_bound = np.array(upper_bound_params[constr_var])
        upper_bound = np.where(upper_bound == None, np.inf, upper_bound)
        upper_bound = np.array(upper_bound, dtype=float)
        lower_bound = np.array(lower_bound_params[constr_var])
        lower_bound = np.where(lower_bound == None, -np.inf, lower_bound)
        lower_bound = np.array(lower_bound, dtype=float)

        # ignore null constraints
        if all(np.atleast_1d(np.isposinf(upper_bound))) and \
            all(np.atleast_1d(np.isneginf(lower_bound))):
            continue
        
        upper_bound_data[data_type(constr_var)] = upper_bound
        lower_bound_data[data_type(constr_var)] = lower_bound

    two_sided_bounds = TwoSidedBounds(
        upper_bound=DataPoint(data=upper_bound_data),
        lower_bound=DataPoint(data=lower_bound_data)
    )

    return SeparateConstraints(
        name=name,
        two_sided_bounds=two_sided_bounds)

