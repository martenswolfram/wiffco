from typing import Dict, Any, Tuple, Sequence
from abc import ABC, abstractmethod
import numpy as np
import logging
from scipy.optimize import minimize, Bounds
from functools import lru_cache
import itertools
from enum import Enum
from twain_wifco.interface import (
    Ambient,
    Control,
    Aggregated,
    DataTable,
    DataPoint,
    get_default_value)
from twain_wifco.statistics import (
    Statistics,
    DiscreteStatisticsParams,
    DiscreteStatistics)
from twain_wifco.control_policy import (
    DiscreteControlPolicy,
    DiscreteControlPolicyParams)
from twain_wifco.plant_model import PlantModel
from twain_wifco.aggregation import Aggregation
from twain_wifco.metrics_accumulation import MetricsAccumulation
from twain_wifco.constraint import Constraint
from twain_wifco.multi_metrics_reduction import MultiMetricsReduction

logger = logging.getLogger(__name__)

class ControlEvaluationSystem:
    def __init__(self,
                 name: str, 
                 plant_model: PlantModel,
                 aggregation: Aggregation,
                 control_constraint: Constraint,
                 metrics_accumulation: MetricsAccumulation,
                 accumulated_constraint: Constraint,
                 multi_metrics_reduction: MultiMetricsReduction):
        
        # Parameters
        self.name = name
        self.plant_model = plant_model
        self.aggregation = aggregation
        self.control_constraint = control_constraint
        self.metrics_accumulation = metrics_accumulation
        self.accumulated_constraint = accumulated_constraint
        self.multi_metrics_reduction = multi_metrics_reduction

    def aggregate_from_ambient_cond(self,
                                    ambient_condition: DataPoint[Ambient],
                                    control_setpoints: DataPoint[Control],
                                    eval_constraints: bool = True):
        
        model_output = self.plant_model.evaluate(
             meteorological_condition=ambient_condition,
             control_input=control_setpoints)
        aggregate = self.aggregation.compute_aggregate(model_output=model_output,
                                                       ambient_condition=ambient_condition,
                                                       control_setpoints=control_setpoints)
        
        if not eval_constraints:
            constraint_satisfied = True
        else:
            constraint_satisfied = self.control_constraint.evaluate_satisfied(
                constr_input_data=control_setpoints)
        return aggregate, constraint_satisfied

    def acc_metrics_from_ambient_cond(self,
                                      ambient_condition: DataPoint[Ambient],
                                      control_setpoints: DataPoint[Control],
                                      duration: int,
                                      eval_constraints: bool = True):
        
        aggregate, constraints_satisfied = self.aggregate_from_ambient_cond(
            ambient_condition=ambient_condition,
            control_setpoints=control_setpoints,
            eval_constraints=eval_constraints)

        acc_metrics = self.metrics_accumulation.acc_metrics(aggregate=aggregate,
                                                            duration=duration)
        
        if eval_constraints:
            constraints_satisfied &= self.accumulated_constraint.evaluate_satisfied(
                constr_input_data=acc_metrics)
            
        return acc_metrics, constraints_satisfied
    
    def expected_acc_metrics(self,
                             ambient_condition_statistics: Statistics,
                             control_policy: DiscreteControlPolicy,
                             duration: int,
                             max_num_amb_cond: int | None = None,
                             eval_constraints: bool = True):
        
        ambient_condition_sample = ambient_condition_statistics.systematic_sample(N_max=max_num_amb_cond)
        aggregate_list = []
        constraints_satisfied = True
        for ambient_condition in ambient_condition_sample.support_data:
            control_setpoints = control_policy.get_control_setpoints(ambient_condition=ambient_condition)
            aggregated, constraint_satisfied = self.aggregate_from_ambient_cond(
                ambient_condition=ambient_condition,
                control_setpoints=control_setpoints,
                eval_constraints=eval_constraints)
            aggregate_list.append(aggregated)
            constraints_satisfied &= constraint_satisfied
            pass

        aggregate_support = DataTable.from_data_points(aggregate_list)
        discrete_statistics_params = DiscreteStatisticsParams(support_data=aggregate_support,
                                                              prevalence=ambient_condition_sample.normalized_weights)
        
        new_name = ""
        discrete_aggregate_statistics = DiscreteStatistics(statistics_name=new_name,
                                                           statistics_params=discrete_statistics_params)

        expected_acc_metrics = self.metrics_accumulation.expected_acc_metrics(
            aggregate_statistics=discrete_aggregate_statistics,
            duration=duration)
        if eval_constraints:
            constraints_satisfied &= self.accumulated_constraint.evaluate_satisfied(constr_input_data=expected_acc_metrics)
        
        return expected_acc_metrics, constraints_satisfied
    
class OptimizationMethod(Enum):
    GRID_SEARCH = "grid_search"
    SIMULTANEOUS_OPTIMIZATION = "simultaneous_optimization"
    LAGRANGIAN_RELAXATION = "lagrangian_relaxation"

class ControlPolicyOptimization(ABC):
    def __init__(self,
                 optimization_name: str):
        self.optimization_name = optimization_name

    @abstractmethod
    def optimize_policy(self,
                        control_eval_system: ControlEvaluationSystem,
                        ambient_condition_statistics: Statistics,
                        duration: int):
        pass

class ControlSetpointVectors:
    def __init__(self,
                 shape: Tuple[int, ...],
                 vectors: Sequence[np.ndarray]):
        if np.prod(shape) != len(vectors):
            raise ValueError("ControlSetpointVectors: Input shape and vectors don't match.")
        self.shape = shape
        self.vectors = vectors

class GridSearchParams:
    def __init__(self,
                 max_num_amb_cond: int,
                 control_setpoint_vectors: Dict[Control, ControlSetpointVectors]):
        self.max_num_amb_cond = max_num_amb_cond
        self.control_setpoint_vectors = control_setpoint_vectors
        
def grid_search_params_from_dict(param_dict: Dict[str, Dict | Any]):
    max_num_amb_cond = param_dict["max_num_ambient_conditions"]
    control_setpoint_vectors = {}
    for ctrl_var, setpoint_vectors in param_dict["control_setpoint_vectors"].items():
        control_setpoint_vectors[Control(ctrl_var)] = ControlSetpointVectors(shape=tuple(setpoint_vectors["shape"]),
                                                                             vectors=setpoint_vectors["data"])
    return GridSearchParams(max_num_amb_cond=max_num_amb_cond,
                            control_setpoint_vectors=control_setpoint_vectors)


class GridSearch(ControlPolicyOptimization):
    def __init__(self,
                 optimization_name: str,
                 optimization_params: GridSearchParams):
        super().__init__(optimization_name=optimization_name)
        self.max_num_amb_cond = optimization_params.max_num_amb_cond
        self.control_setpoint_vectors = optimization_params.control_setpoint_vectors

    def optimize_policy(self,
                        control_eval_system: ControlEvaluationSystem,
                        ambient_condition_statistics: Statistics[Ambient],
                        duration: int):
        
        logger.info(f"GridSearch: Find optimal control policy")
        ambient_condition_sample = ambient_condition_statistics.systematic_sample(N_max=self.max_num_amb_cond)
        num_ambient_conditions = ambient_condition_sample.N
        ctrl_variables_order = list(self.control_setpoint_vectors.keys())
        control_setpoint_vectors_list = [setpoints for ctrl_var in ctrl_variables_order for \
                                  setpoints in self.control_setpoint_vectors[ctrl_var].vectors]
        num_ctrl_setpoint_combinations = np.prod([len(ctrl_setpoints) for ctrl_setpoints in control_setpoint_vectors_list])

        ctrl_shapes_dict={var: data.shape for var, data in self.control_setpoint_vectors.items()}

        aggregate_evaluations = {
            aggr_var: np.empty(shape=(num_ambient_conditions, num_ctrl_setpoint_combinations, *aggr_shape)) \
            for aggr_var, aggr_shape in control_eval_system.aggregation.output_interface.shapes(Aggregated).items()
            }
        logger.info(f"Number of ambient conditions: {num_ambient_conditions}.")
        logger.info(f"Number of ctrl settings: {num_ctrl_setpoint_combinations}.")
        num_eval = num_ctrl_setpoint_combinations * num_ambient_conditions
        logger.info(f"Performing {num_eval} system evaluations.")
        
        instantaneous_constr_satisfied_matrix = np.empty(
            shape=(num_ambient_conditions, num_ctrl_setpoint_combinations), dtype=bool)
        
        for n_amb, ambient_condition in enumerate(ambient_condition_sample.support_data):
            ctrl_setpoint_combinations = itertools.product(*control_setpoint_vectors_list)
            for n_ctrl, ctrl_setpoints in enumerate(ctrl_setpoint_combinations):
                control = DataPoint.from_vector(data_vector=ctrl_setpoints,
                                                order=ctrl_variables_order,
                                                shapes_dict=ctrl_shapes_dict)
                aggregate, instantaneous_constr_satisfied = \
                    control_eval_system.aggregate_from_ambient_cond(
                        ambient_condition=ambient_condition,
                        control_setpoints=control)
                instantaneous_constr_satisfied_matrix[n_amb, n_ctrl] = instantaneous_constr_satisfied
                for agg_var in control_eval_system.aggregation.output_interface.shapes(Aggregated).keys():
                    aggregate_evaluations[agg_var][n_amb, n_ctrl] = aggregate[agg_var]
                pass


        num_ctrl_policies = num_ambient_conditions**num_ctrl_setpoint_combinations
        logger.info(f"Evaluating {num_ctrl_policies} control policies.")
                
        multi_metrics_reduced = []
        constraints_satisfied = []
        ctrl_settings_indices_product = itertools.product(range(num_ctrl_setpoint_combinations), repeat=num_ambient_conditions)
        for ctrl_indices in ctrl_settings_indices_product:
            inst_constraints_satisfied = np.all(
                instantaneous_constr_satisfied_matrix[np.arange(num_ambient_conditions),
                                                     ctrl_indices])
            if not inst_constraints_satisfied:
                multi_metrics_reduced.append(None)
                constraints_satisfied.append(False)
            else:
                # Each ctrl_indices corresponds to a discrete control strategy
                aggr_support_data = DataTable(
                    {aggr_var: aggr_eval[np.arange(num_ambient_conditions), ctrl_indices] for \
                     aggr_var, aggr_eval in aggregate_evaluations.items()})
                discrete_stat_params = DiscreteStatisticsParams(
                    support_data=aggr_support_data,
                    prevalence=ambient_condition_sample.normalized_weights)
                discrete_aggr_stat = DiscreteStatistics(
                    statistics_name="",
                    statistics_params=discrete_stat_params)
                expected_acc_metrics = \
                    control_eval_system.metrics_accumulation.expected_acc_metrics(
                    aggregate_statistics=discrete_aggr_stat,
                    duration=duration)
                multi_metrics_reduced.append(control_eval_system.multi_metrics_reduction.evaluate(
                    acc_metrics=expected_acc_metrics))
                acc_metrics_constraint_satisfied = control_eval_system.accumulated_constraint.evaluate_satisfied(
                    constr_input_data=expected_acc_metrics)
                constraints_satisfied.append(acc_metrics_constraint_satisfied)
            
        # Find the optimal control strategy that satisfies the constraints
        if not any(constraints_satisfied):
            raise ValueError("GridSearch.optimize_policy: Failed to find feasible policy due to constraints-violation.")
        if control_eval_system.multi_metrics_reduction.maximize:
            best_index = np.argmax(np.where(constraints_satisfied, np.array(multi_metrics_reduced), -np.inf))
        else:
            best_index = np.argmin(np.where(constraints_satisfied, np.array(multi_metrics_reduced), np.inf))
        
        # Specify discrete control strategy
        # control settings (linear index) for each ambient condition
        amb_cond_ctrl_indices = np.unravel_index([best_index], [num_ctrl_setpoint_combinations] * num_ambient_conditions)
        # Corresponding control setpoints as DataTable
        control_setpoint_list = []
        for multi_index in amb_cond_ctrl_indices:
            control_setpoints = np.array(list(ctrl_vector[i] for i, ctrl_vector in zip(multi_index, control_setpoint_vectors_list)))
            control_setpoint_list.append(DataPoint.from_vector(data_vector=control_setpoints,
                                                                order=ctrl_variables_order,
                                                                shapes_dict=ctrl_shapes_dict))
        control_setpoints_data = DataTable.from_data_points(control_setpoint_list)
        discrete_control_policy_params = DiscreteControlPolicyParams(
            ambient_support_data=ambient_condition_sample.support_data,
            control_out_data=control_setpoints_data
        )
        return DiscreteControlPolicy(name="optimized_discrete_control_policy",
                                     params=discrete_control_policy_params)

class SimultaneousOptimizationParams:
    def __init__(self,
                 max_num_amb_cond: int,
                 scipy_method: str,
                 scipy_options: Dict[str, Any]):
        self.max_num_amb_cond = max_num_amb_cond
        self.scipy_method = scipy_method
        self.scipy_options = scipy_options

def simultaneous_optimization_params_from_dict(param_dict: Dict[str, Any]):
    max_num_amb_cond = param_dict["max_num_ambient_conditions"]
    scipy_method = param_dict["scipy_method"]
    scipy_options = param_dict["scipy_options"]
    return SimultaneousOptimizationParams(
        max_num_amb_cond=max_num_amb_cond,
        scipy_method=scipy_method,
        scipy_options=scipy_options)

class ContinuousOptimizationManager:
    def __init__(self,
                 control_eval_system: ControlEvaluationSystem,
                 ambient_condition_statistics: Statistics[Ambient],
                 duration: int,
                 max_num_amb_cond: int | None = None):
        self.control_eval_system = control_eval_system
        self.ambient_condition_statistics = ambient_condition_statistics
        self.duration = duration
        self.max_num_amb_cond = max_num_amb_cond
        
        self.control_policy = self.default_discrete_policy()

    @lru_cache(maxsize=None)
    def expected_acc_metrics_w_cache(self,
                                     ctrl_as_tuple: Tuple[float, ...]):
        self.control_policy.set_control_data(control_data_vector=np.array(ctrl_as_tuple))
        expected_acc_metrics, _ = self.control_eval_system.expected_acc_metrics(
            ambient_condition_statistics=self.ambient_condition_statistics,
            control_policy=self.control_policy,
            duration=self.duration,
            max_num_amb_cond=self.max_num_amb_cond,
            eval_constraints=False)
        return expected_acc_metrics
        
    def default_discrete_policy(self):
        
        amb_cond_sample = self.ambient_condition_statistics.systematic_sample(N_max=self.max_num_amb_cond)
        ctrl_shapes = self.control_eval_system.plant_model.input_interface.shapes(Control)
        num_samples = amb_cond_sample.N
        control_setpoints = DataTable({ctrl_var: get_default_value(data_var=ctrl_var,
                                                                   shape=(num_samples,) + shape) for \
                                       ctrl_var, shape in ctrl_shapes.items()})
        # Scattered Interpolation out data
        discrete_control_policy_params = DiscreteControlPolicyParams(
            ambient_support_data=amb_cond_sample.support_data,
            control_out_data=control_setpoints)        
        return DiscreteControlPolicy(name="discrete_control_policy",
                                     params=discrete_control_policy_params)
            
class SimultaneousOptimization(ControlPolicyOptimization):
    def __init__(self,
                 optimization_name: str,
                 optimization_params: SimultaneousOptimizationParams):
        super().__init__(optimization_name=optimization_name)
        self.max_num_amb_cond = optimization_params.max_num_amb_cond
        self.scipy_method = optimization_params.scipy_method
        self.scipy_options = optimization_params.scipy_options

    def optimize_policy(self,
                        control_eval_system: ControlEvaluationSystem,
                        ambient_condition_statistics: Statistics[Ambient],
                        duration: int):
        
        # OptimizationManager
        opt_mgr = ContinuousOptimizationManager(
            control_eval_system=control_eval_system,
            ambient_condition_statistics=ambient_condition_statistics,
            duration=duration,
            max_num_amb_cond=self.max_num_amb_cond)
        
        expected_acc_metrics_w_cache = lambda x : opt_mgr.expected_acc_metrics_w_cache(tuple(x))        
        # define cost function based on accumulated metrics
        cost_function = \
            opt_mgr.control_eval_system.multi_metrics_reduction.cost_function(
                eval_acc_metrics_from_x=expected_acc_metrics_w_cache)

        # Constraints
        constraints = []
        # Order and shapes of control variables
        x_order = opt_mgr.control_policy.control_out_data.order
        x_shapes_dict=opt_mgr.control_policy.control_out_data.shapes()
        # define constraint based on accumulated metrics
        acc_metrics_constraint = \
            opt_mgr.control_eval_system.accumulated_constraint.scipy_object(
                x_order=x_order,
                x_shapes_dict=x_shapes_dict,                     
                x_constraint_evaluation=expected_acc_metrics_w_cache)
        constraints.append(acc_metrics_constraint)

        # define control-constraints
        num_ac = len(opt_mgr.control_policy.control_out_data)
        control_constraint_object = opt_mgr.control_eval_system.control_constraint.scipy_object(
            x_order=x_order,
            x_shapes_dict=x_shapes_dict,
            num_points=num_ac,
        )
        if isinstance(control_constraint_object, Bounds):
            bounds = control_constraint_object
        else:
            constraints.append(control_constraint_object)

        x0 = opt_mgr.control_policy.control_out_data.to_vector()
        res = minimize(cost_function,
                       x0,
                       method=self.scipy_method,
                       bounds=bounds,
                       constraints=constraints,
                       options=self.scipy_options)
        opt_mgr.control_policy.set_control_data(control_data_vector=res.x)
        return opt_mgr.control_policy
     

class LagrangianRelaxationParams:
    def __init__(self,
                 max_num_amb_cond: int,
                 alpha_0: float,
                 max_iter: int,
                 subgradient_tol: float,
                 scipy_method: str,
                 scipy_options: Dict[str, Any]):
        self.max_num_amb_cond = max_num_amb_cond
        self.alpha_0 = alpha_0
        self.max_iter = max_iter
        self.subgradient_tol = subgradient_tol
        self.scipy_method = scipy_method
        self.scipy_options = scipy_options

def lagrangian_relaxation_params_from_dict(param_dict: Dict[str, Any]):
    max_num_amb_cond = param_dict["max_num_ambient_conditions"]
    alpha_0 = param_dict["alpha_0"]
    max_iter = param_dict["max_iter"]
    subgradient_tol = param_dict["subgradient_tol"]
    scipy_method = param_dict["scipy_method"]
    scipy_options = param_dict["scipy_options"]
    return LagrangianRelaxationParams(
        max_num_amb_cond=max_num_amb_cond,
        alpha_0=alpha_0,
        max_iter=max_iter,
        subgradient_tol=subgradient_tol,
        scipy_method=scipy_method,
        scipy_options=scipy_options)
 
class LagrangianRelaxation(ControlPolicyOptimization):
    def __init__(self,
                 optimization_name: str,
                 optimization_params: LagrangianRelaxationParams):
        super().__init__(optimization_name=optimization_name)
        self.max_num_amb_cond = optimization_params.max_num_amb_cond
        self.alpha_0 = optimization_params.alpha_0
        self.max_iter = optimization_params.max_iter
        self.subgradient_tol = optimization_params.subgradient_tol
        self.scipy_method = optimization_params.scipy_method
        self.scipy_options = optimization_params.scipy_options
        
    def optimize_policy(self,
                        control_eval_system: ControlEvaluationSystem,
                        ambient_condition_statistics: Statistics[Ambient],
                        duration: int):
        
        # OptimizationManager
        opt_mgr = ContinuousOptimizationManager(
            control_eval_system=control_eval_system,
            ambient_condition_statistics=ambient_condition_statistics,
            duration=duration,
            max_num_amb_cond=self.max_num_amb_cond)
        
        control_setpoints = opt_mgr.control_policy.control_out_data
                
        ambient_condition_sample = ambient_condition_statistics.systematic_sample(N_max=self.max_num_amb_cond)

        upper_bound_constraints = control_eval_system.accumulated_constraint.upper_bound_constraints()
        lagrangian_lambdas = list(np.zeros_like(ub.upper_bound, dtype=float) for ub in upper_bound_constraints)
        # outer-iteration counter
        for t in range(self.max_iter):
            logger.debug(f"Iteration {t}")
            expected_constraint_evaluations = list(np.zeros_like(ub.upper_bound, dtype=float) for ub in upper_bound_constraints)
            for i_ac, (prob_weight, ambient_condition) in enumerate(ambient_condition_sample.weighted_variables_iter()):

                # Separate optimization for each ambient condition
                # Cost function
                def cost_function(ctrl_setpoints_vec, i_ac=i_ac):
                    control_setpoints.update_point_from_vector(ind=i_ac,
                                                               vector=ctrl_setpoints_vec)
                    acc_metrics, _ = opt_mgr.control_eval_system.acc_metrics_from_ambient_cond(
                        ambient_condition=ambient_condition,
                        control_setpoints=control_setpoints.get_point(i_ac),
                        duration=opt_mgr.duration,
                        eval_constraints=False)
                    scalar_objective = opt_mgr.control_eval_system.multi_metrics_reduction.evaluate(
                        acc_metrics=acc_metrics)
                    # First assume that we have an objective function (which we want to maximise),
                    # and upper-bound constraints which must not be exceded.
                    if not opt_mgr.control_eval_system.multi_metrics_reduction.maximize:
                        scalar_objective *= -1                
                    for lagrangian_lambda, ub_constraint in zip(lagrangian_lambdas, upper_bound_constraints):
                        scalar_objective -= lagrangian_lambda.dot(ub_constraint.constraint_fun(acc_metrics))
                    # Scipy will minimize a cost function, hence take the negative value
                    return - scalar_objective
                
                # Order and shapes of control variables
                x_order = opt_mgr.control_policy.control_out_data.order
                x_shapes_dict=opt_mgr.control_policy.control_out_data.shapes()
                
                # control constraints
                control_constraint_object = opt_mgr.control_eval_system.control_constraint.scipy_object(
                    x_order=x_order,
                    x_shapes_dict=x_shapes_dict
                )
                if isinstance(control_constraint_object, Bounds):
                    bounds = control_constraint_object
                    control_constraint = None
                else:
                    bounds = None
                    control_constraint = control_constraint_object

                # Minimize Lagrangian
                # Initial value
                x0 = control_setpoints.get_point(i_ac).to_vector()
                res = minimize(cost_function,
                               x0,
                               method=self.scipy_method,
                               bounds=bounds,
                               constraints=control_constraint,
                               options=self.scipy_options)
                control_setpoints.update_point_from_vector(ind=i_ac,
                                                           vector=res.x)
                
                # Recompute final accumulated metrics
                acc_metrics, _ = opt_mgr.control_eval_system.acc_metrics_from_ambient_cond(
                    ambient_condition=ambient_condition,
                    control_setpoints=control_setpoints.get_point(i_ac),
                    duration=opt_mgr.duration,
                    eval_constraints=False)
                # Constraint evaluation
                for i_constraint, ub_constraint in enumerate(upper_bound_constraints):
                    expected_constraint_evaluations[i_constraint] += prob_weight * ub_constraint.constraint_fun(acc_metrics)
            
            converged = True
            # update lagrangian multipliers
            for i_constr, (ub_constraint, expected_constr_eval) in \
                enumerate(zip(upper_bound_constraints, expected_constraint_evaluations)):
                subgradient = ub_constraint.upper_bound - expected_constr_eval
                step = subgradient * self.alpha_0 / np.sqrt(t + 1)
                lagrangian_lambda = lagrangian_lambdas[i_constr] - step
                lagrangian_lambda[lagrangian_lambda < 0] = 0
                lagrangian_lambdas[i_constr] = lagrangian_lambda
                if np.linalg.norm(subgradient, ord=np.inf) > self.subgradient_tol:
                    converged = False
                logger.debug(f"subgradient: {subgradient}")
                logger.debug(f"lag lambdas: {lagrangian_lambda}")

            t += 1
            
            if converged:
                break

        return opt_mgr.control_policy
        
     