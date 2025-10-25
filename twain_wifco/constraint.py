from typing import Dict, Any, List, Generic, Callable, Tuple
from abc import abstractmethod
import numpy as np
from scipy.optimize import NonlinearConstraint, Bounds
from enum import Enum
from twain_wifco.interface import (
    Component,
    ComponentParams,
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

    Args:
        constraint_name (str): Name of the constraint.
        constraint_params (ComponentParams): Parameters defining the constraint.
    """
    def __init__(self,
                 constraint_name: str,
                 constraint_params: ComponentParams):
        super().__init__(component_name=constraint_name,
                         component_params=constraint_params)

    def evaluate_satisfied(self,
                           constr_input_data: DataPoint[DataType]) -> bool:
        """Check whether the constraint is satisfied for the given input.

        Args:
            constr_input_data (DataPoint[DataType]): Input data to evaluate.

        Returns:
            bool: True if constraint is satisfied, False otherwise.
        """
        self.validate_inputs(input_data={constr_input_data.data_type: constr_input_data})
        return self._evaluate_satisfied(constr_input_data=constr_input_data)

    @abstractmethod
    def _evaluate_satisfied(self,
                            constr_input_data: DataPoint[DataVariable]) -> bool:
        """Implementation-specific check if constraint is satisfied.

        Args:
            constr_input_data (DataPoint[DataVariable]): Input data.

        Returns:
            bool: True if satisfied.
        """
        pass

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

class SeparateConstraintsParams(ComponentParams):
    """Parameter container for SeparateConstraints.

    Attributes:
        bounds (TwoSidedBounds): Upper and lower bounds for the constraint.
    """
    def __init__(self, two_sided_bounds: TwoSidedBounds):
        self.bounds = two_sided_bounds

    def input_interface(self) -> Interface:
        """Define input interface for the component.

        Returns:
            Interface: Input interface object.
        """
        return Interface(all_shapes={
            self.bounds.data_type: self.bounds.upper_bound.shapes()
        })

    def output_interface(self) -> Interface:
        """Define output interface for the component.

        Returns:
            Interface: Output interface object (empty for constraints).
        """
        return Interface(all_shapes={})


def separate_constraints_params_from_dict(param_dict: Dict[str, Any]) -> SeparateConstraintsParams:
    """Create SeparateConstraintsParams from a parameter dictionary.

    Args:
        param_dict (Dict[str, Any]): Dictionary containing upper/lower bounds and data type.

    Returns:
        SeparateConstraintsParams: Constructed parameters object.
    """
    data_type = data_type_from_string(param_dict["data_type"])

    upper_bound_data = {}
    lower_bound_data = {}
    upper_bound_params = param_dict["upper_bound"]
    lower_bound_params = param_dict["lower_bound"]

    for constr_var in upper_bound_params.keys():
        upper_bound = np.array([u if u is not None else np.inf for u in upper_bound_params[constr_var]])
        lower_bound = np.array([l if l is not None else -np.inf for l in lower_bound_params[constr_var]])

        # ignore null constraints
        if all(np.isposinf(upper_bound)) and all(np.isneginf(lower_bound)):
            continue

        upper_bound_data[data_type(constr_var)] = upper_bound
        lower_bound_data[data_type(constr_var)] = lower_bound

    two_sided_bounds = TwoSidedBounds(
        upper_bound=DataPoint(data=upper_bound_data),
        lower_bound=DataPoint(data=lower_bound_data)
    )

    return SeparateConstraintsParams(two_sided_bounds=two_sided_bounds)


class SeparateConstraints(Constraint):
    """Concrete constraint class handling two-sided bounds."""

    def __init__(self,
                 constraint_name: str,
                 constraint_params: SeparateConstraintsParams):
        super().__init__(constraint_name=constraint_name,
                         constraint_params=constraint_params)
        self.bounds = constraint_params.bounds

    def _evaluate_satisfied(self,
                            constr_input_data: DataPoint[DataVariable]) -> bool:
        """Check if all constraints are satisfied for the given input.

        Args:
            constr_input_data (DataPoint[DataVariable]): Input data.

        Returns:
            bool: True if all constraints are satisfied.
        """
        # Bounds for unconstrained variables are filled with corresponding (pos/neg) infinite bounds 
        lb = self.bounds.lower_bound.to_vector(order=constr_input_data.order,
                                               fill_shapes=constr_input_data.shapes(),
                                               fill_value=-np.inf)
        ub = self.bounds.upper_bound.to_vector(order=constr_input_data.order,
                                               fill_shapes=constr_input_data.shapes(),
                                               fill_value=np.inf)
        
        lower_diff = constr_input_data.to_vector() - lb
        upper_diff = constr_input_data.to_vector() - ub
        
        return all(lower_diff >= -self.bounds.lower_bound.abs_tols_vec()) and \
               all(upper_diff <= self.bounds.upper_bound.abs_tols_vec())

    def scipy_object(self,
                     x_order: List[DataType],
                     x_shapes_dict: Dict[DataType, Tuple[int, ...]],
                     num_points: int | None = None,
                     x_constraint_evaluation: Callable | None = None):
        """ Creates SciPy objects for constraint evaluation"""
        if x_constraint_evaluation is None:
            # If x is only passed through, create a SciPy-Bounds object, based on the external variable order
            # Bounds for unconstrained variables are filled with corresponding (pos/neg) infinite bounds 
            lb = self.bounds.lower_bound.to_vector(order=x_order,
                                                   fill_shapes=x_shapes_dict,
                                                   fill_value=-np.inf,
                                                   batch_multiply=num_points)
            ub = self.bounds.upper_bound.to_vector(order=x_order,
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
                    return np.concatenate([constr_evaluation[var].ravel() for var in self.bounds.order])    
                lb = self.bounds.lower_bound.to_vector(order=self.bounds.order)
                ub = self.bounds.upper_bound.to_vector(order=self.bounds.order)
                return NonlinearConstraint(fun=eval_constraint, lb=lb, ub=ub)
            else:
                # Batch-constraint evaluation
                def eval_constraints(x):
                    # Note: The constraint-evaluation function is interpreted such that 
                    # both input data and output data represents a flat concatenation over 
                    # all points.
                    constr_evaluation: DataTable = x_constraint_evaluation(x)
                    return constr_evaluation.to_vector(order=self.bounds.order) 
                # The bounds are defined for each point, hence need to be tiled/extruded.
                lb = self.bounds.lower_bound.to_vector(order=self.bounds.order,
                                                       batch_multiply=num_points)
                ub = self.bounds.upper_bound.to_vector(order=self.bounds.order,
                                                       batch_multiply=num_points)
                return NonlinearConstraint(fun=eval_constraints, lb=lb, ub=ub)
    
    def upper_bound_constraints(self) -> List[UpperBoundConstraint]:
        """Convert to a list of upper-bound constraints for optimization.

        Returns:
            List[UpperBoundConstraint]: List of upper-bound constraints.
        """
        ub_constraints = []

        for var in self.bounds.upper_bound.keys():
            # Lower bound as upper-bound constraint
            if self.bounds.lower_bound[var] > -np.inf:
                def constraint_fun(values, v=var):
                    return -values[v]
                ub_constraints.append(UpperBoundConstraint(
                    upper_bound=-self.bounds.lower_bound[var],
                    constraint_fun=constraint_fun
                ))

            # Upper bound as upper-bound constraint
            if self.bounds.upper_bound[var] < np.inf:
                def constraint_fun(values, v=var):
                    return values[v]
                ub_constraints.append(UpperBoundConstraint(
                    upper_bound=self.bounds.upper_bound[var],
                    constraint_fun=constraint_fun
                ))

        return ub_constraints
