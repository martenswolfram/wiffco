from typing import Dict, Any, Tuple, List
from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize
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
    ControlPolicy,
    ScatteredInterpPolicy,
    ScatteredInterpPolicyParams)
from twain_wifco.utils import ScatteredInterpolatorParams, ScatteredInterpolatorType
from twain_wifco.plant_model import PlantModel
from twain_wifco.aggregation import Aggregation
from twain_wifco.metrics_accumulation import MetricsAccumulation
from twain_wifco.constraint import TwoSidedConstraint
from twain_wifco.multi_metrics_reduction import MultiMetricsReduction

class ControlEvaluationSystem:
    def __init__(self,
                 name: str, 
                 plant_model: PlantModel,
                 aggregation: Aggregation,
                 control_constraint: TwoSidedConstraint,
                 metrics_accumulation: MetricsAccumulation,
                 accumulated_constraint: TwoSidedConstraint,
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
                                    ambient_condition: Dict[Ambient, np.ndarray],
                                    control_setpoints: Dict[Control, np.ndarray]):
        
        model_output = self.plant_model.evaluate(
             meteorological_condition=ambient_condition,
             control_input=control_setpoints)
        aggregate = self.aggregation.compute_aggregate(model_output=model_output,
                                                       ambient_condition=ambient_condition,
                                                       control_setpoints=control_setpoints)
        constraint_eval = self.control_constraint.evaluate(
            constr_input_data=control_setpoints)
        
        return aggregate, constraint_eval.satisfied()

    def acc_metrics_from_ambient_cond(self,
                                      ambient_condition: Dict[Ambient, np.ndarray],
                                      control_setpoints: Dict[Control, np.ndarray],
                                      duration: int):
        
        model_output = self.plant_model.evaluate(
             meteorological_condition=ambient_condition,
             control_input=control_setpoints)    
        aggregate = self.aggregation.compute_aggregate(model_output=model_output,
                                                       ambient_condition=ambient_condition,
                                                       control_setpoints=control_setpoints)
        return self.metrics_accumulation.acc_metrics(aggregate=aggregate,
                                                     duration=duration)

    def expected_acc_metrics(self,
                             ambient_condition_statistics: Statistics,
                             control_policy: ControlPolicy,
                             duration: int,
                             max_num_amb_cond: int | None = None):
        
        ambient_condition_sample = ambient_condition_statistics.systematic_sample(N_max=max_num_amb_cond)
        aggregate_list = []
        for ambient_condition in ambient_condition_sample.support_data:
            control_setpoints = control_policy.get_control_setpoints(ambient_condition=ambient_condition)
            model_output = self.plant_model.evaluate(meteorological_condition=ambient_condition,
                                                     control_input=control_setpoints)
            aggregated = self.aggregation.compute_aggregate(model_output=model_output,
                                                            ambient_condition=ambient_condition,
                                                            control_setpoints=control_setpoints)
            aggregate_list.append(aggregated)

        aggregate_support = DataTable.from_data_points(aggregate_list)
        discrete_statistics_params = DiscreteStatisticsParams(statistical_type=Aggregated,
                                                              support_data=aggregate_support,
                                                              prevalence=ambient_condition_sample.normalized_weights)
        
        new_name = ""
        discrete_aggregate_statistics = DiscreteStatistics(statistics_name=new_name,
                                                           statistics_params=discrete_statistics_params)

        expected_acc_metrics = self.metrics_accumulation.expected_acc_metrics(
            aggregate_statistics=discrete_aggregate_statistics,
            duration=duration)
        
        return expected_acc_metrics
    
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
                        duration: int,
                        initial_policy: ControlPolicy | None = None):
        pass

class ControlSetpointVectors:
    def __init__(self,
                 shape: Tuple[int, ...],
                 vectors: List[np.ndarray]):
        if np.prod(shape) != len(vectors):
            raise ValueError("ControlSetpointVectors: Input shape and vectors don't match.")
        self.shape = shape
        self.vectors = vectors

class GridSearchParams:
    def __init__(self,
                 max_num_amb_cond: int,
                 scattered_interpolation_type: ScatteredInterpolatorType,
                 control_setpoint_vectors: Dict[Control, ControlSetpointVectors]):
        self.max_num_amb_cond = max_num_amb_cond
        self.scattered_interpolation_type = scattered_interpolation_type
        self.control_setpoint_vectors = control_setpoint_vectors
        
def grid_search_params_from_dict(param_dict: Dict[str, Dict | Any]):
    max_num_amb_cond = param_dict["max_num_ambient_conditions"]
    scattered_interpolation_type = ScatteredInterpolatorType(param_dict["scattered_interpolation_type"])
    control_setpoint_vectors = {}
    for ctrl_var, setpoint_vectors in param_dict["control_setpoint_vectors"].items():
        control_setpoint_vectors[Control(ctrl_var)] = ControlSetpointVectors(shape=tuple(setpoint_vectors["shape"]),
                                                                             vectors=setpoint_vectors["data"])
    return GridSearchParams(max_num_amb_cond=max_num_amb_cond,
                            scattered_interpolation_type=scattered_interpolation_type,
                            control_setpoint_vectors=control_setpoint_vectors)


class GridSearch(ControlPolicyOptimization):
    def __init__(self,
                 optimization_name: str,
                 optimization_params: GridSearchParams):
        super().__init__(optimization_name=optimization_name)
        self.max_num_amb_cond = optimization_params.max_num_amb_cond
        self.scattered_interpolation_type = optimization_params.scattered_interpolation_type
        self.control_setpoint_vectors = optimization_params.control_setpoint_vectors

    def optimize_policy(self,
                        control_eval_system: ControlEvaluationSystem,
                        ambient_condition_statistics: Statistics,
                        duration: int,
                        initial_policy: ControlPolicy | None = None):
        
        print(f"GridSearch: Find optimal control policy")
        ambient_condition_sample = ambient_condition_statistics.systematic_sample(N_max=self.max_num_amb_cond)
        num_ambient_conditions = ambient_condition_sample.N
        ctrl_variables_order = list(self.control_setpoint_vectors.keys())
        control_setpoints_list = [setpoints for ctrl_var in ctrl_variables_order for \
                                  setpoints in self.control_setpoint_vectors[ctrl_var].vectors]
        ctrl_setpoint_combinations = np.array(list(itertools.product(*control_setpoints_list)))
        ctrl_setpoints_table = DataTable.from_matrix(
            data_matrix=ctrl_setpoint_combinations,
            order=ctrl_variables_order,
            shapes_dict={var: data.shape for var, data in self.control_setpoint_vectors.items()})
        num_ctrl_settings = len(ctrl_setpoints_table)

        aggregate_evaluations = {
            aggr_var: np.empty(shape=(num_ambient_conditions, num_ctrl_settings, *aggr_shape)) \
            for aggr_var, aggr_shape in control_eval_system.aggregation.output_interface.shapes(Aggregated).items()
            }
        print(f"Number of ambient conditions: {num_ambient_conditions}.")
        print(f"Number of ctrl settings: {num_ctrl_settings}.")
        num_eval = num_ctrl_settings * num_ambient_conditions
        print(f"Performing {num_eval} system evaluations.")
        
        instantaneous_constr_satisfied_matrix = np.empty(
            shape=(num_ambient_conditions, num_ctrl_settings), dtype=bool)
        
        for n_amb, ambient_condition in enumerate(ambient_condition_sample.support_data):
            for n_ctrl, ctrl_setpoints in enumerate(ctrl_setpoints_table):
                aggregate, instantaneous_constr_satisfied = \
                    control_eval_system.aggregate_from_ambient_cond(
                        ambient_condition=ambient_condition,
                        control_setpoints=ctrl_setpoints)
                instantaneous_constr_satisfied_matrix[n_amb, n_ctrl] = instantaneous_constr_satisfied
                for agg_var in control_eval_system.aggregation.output_interface.shapes(Aggregated).keys():
                    aggregate_evaluations[agg_var][n_amb, n_ctrl] = aggregate[agg_var]
        pass


        num_ctrl_policies = num_ambient_conditions**len(ctrl_setpoint_combinations)
        print(f"Evaluating {num_ctrl_policies} control policies.")
                
        multi_metrics_reduced = []
        constraints_satisfied = []
        ctrl_settings_indices_product = itertools.product(range(num_ctrl_settings), repeat=num_ambient_conditions)
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
                    statistical_type=Aggregated,
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
                acc_metrics_constraint_eval = control_eval_system.accumulated_constraint.evaluate(
                    constr_input_data=expected_acc_metrics)
                constraints_satisfied.append(acc_metrics_constraint_eval.satisfied())
            
        # Find the optimal control strategy that satisfies the constraints
        if not any(constraints_satisfied):
            raise ValueError("GridSearch.optimize_policy: Failed to find feasible policy due to constraints-violation.")
        if control_eval_system.multi_metrics_reduction.maximize:
            best_index = np.argmax(np.where(constraints_satisfied, np.array(multi_metrics_reduced), -np.inf))
        else:
            best_index = np.argmin(np.where(constraints_satisfied, np.array(multi_metrics_reduced), np.inf))
        
        # Specify discrete control strategy
        # control settings (linear index) for each ambient condition
        amb_cond_ctrl_indices = np.unravel_index(best_index, [num_ctrl_settings] * num_ambient_conditions)
        # Corresponding control setpoints as DataTable
        control_setpoints_data = ctrl_setpoints_table.get_points(list(amb_cond_ctrl_indices))
        # Scattered Interpolation out data
        interpolation_params = ScatteredInterpolatorParams(
            scattered_interp_type=self.scattered_interpolation_type,
            support_data=ambient_condition_sample.support_data,
            out_data=control_setpoints_data)
        
        scattered_control_policy_params = ScatteredInterpPolicyParams(ambient_interp_params=interpolation_params)
        return ScatteredInterpPolicy(policy_name="optimized_discrete_control_policy",
                                     policy_params=scattered_control_policy_params)

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
                 ambient_condition_statistics: Statistics,
                 duration: int,
                 control_policy: ControlPolicy,
                 max_num_amb_cond: int | None = None):
        self.control_eval_system = control_eval_system
        self.ambient_condition_statistics = ambient_condition_statistics
        self.duration = duration
        self.max_num_amb_cond = max_num_amb_cond
        
        self.control_policy = control_policy
        if self.control_policy is None:
            self.default_initial_discrete_policy()

    @lru_cache(maxsize=None)
    def expected_acc_metrics_w_cache(self,
                                     ctrl_as_tuple: Tuple[float]):
        self.control_policy.set_ctrl_parameters_full(np.array(ctrl_as_tuple))
        return self.control_eval_system.expected_acc_metrics(
            ambient_condition_statistics=self.ambient_condition_statistics,
            control_policy=self.control_policy,
            duration=self.duration,
            max_num_amb_cond=self.max_num_amb_cond)
        
    def default_initial_discrete_policy(self):
        
        amb_cond_sample = self.ambient_condition_statistics.systematic_sample(N_max=self.max_num_amb_cond)
        ctrl_vars = self.control_eval_system.plant_model.inputs_of_type(t=Control)
        control_setpoints = {ctrl_var: get_default_value(data_var=ctrl_var, dim=dim) for \
                             ctrl_var, dim in ctrl_vars.items()}
        discrete_control_policy_params = ScatteredInterpPolicyParams(
            ambient_support_points=amb_cond_sample.support_data,
            control_setpoints=control_setpoints)
        self.control_policy = ScatteredInterpPolicy(
            policy_name="discrete_policy_opt",
            policy_params=discrete_control_policy_params)
            
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
                        ambient_condition_statistics: Statistics,
                        duration: int,
                        initial_policy: ControlPolicy | None = None):
        
        # OptimizationManager
        opt_mgr = ContinuousOptimizationManager(
            control_eval_system=control_eval_system,
            ambient_condition_statistics=ambient_condition_statistics,
            duration=duration,
            control_policy=initial_policy,
            max_num_amb_cond=self.max_num_amb_cond)
        
        expected_acc_metrics_w_cache = lambda x : opt_mgr.expected_acc_metrics_w_cache(tuple(x))        
        # define cost function based on accumulated metrics
        cost_function = \
            opt_mgr.control_eval_system.multi_metrics_reduction.cost_function(
                eval_acc_metrics_from_x=expected_acc_metrics_w_cache)

        # Constraints
        constraints = []

        # define constraint based on accumulated metrics
        acc_metrics_constraint = \
            opt_mgr.control_eval_system.accumulated_constraint.scipy_constraint(
                eval_constraint_from_x=expected_acc_metrics_w_cache)
        constraints.append(acc_metrics_constraint)

        # define constraints based on control
        if not isinstance(opt_mgr.control_policy, ScatteredInterpPolicy):
            raise NotImplementedError("Control constraints only implemented for scattered-interpolation control policy.")
        num_ac, num_ctrl_vars = opt_mgr.control_policy.control_setpoints.shape
        for i_ac in np.arange(num_ac):
            get_ctrl_setpoints_for_ac = lambda x, i_ac=i_ac : {
                ctrl_var: x[i_ac * num_ctrl_vars + i_ctrl] for \
                    i_ctrl, ctrl_var in enumerate(opt_mgr.control_policy.control_variables)}
            constraints.append(opt_mgr.control_eval_system.control_constraint.scipy_constraint(
                eval_constraint_from_x=get_ctrl_setpoints_for_ac))

        x0 = opt_mgr.control_policy.get_ctrl_parameters_full()
        minimize(cost_function,
                 x0,
                 method=self.scipy_method,
                 constraints=constraints,
                 options=self.scipy_options)
        
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
                        ambient_condition_statistics: Statistics,
                        duration: int,
                        initial_policy: ControlPolicy | None = None):
        
        # OptimizationManager
        opt_mgr = ContinuousOptimizationManager(
            control_eval_system=control_eval_system,
            ambient_condition_statistics=ambient_condition_statistics,
            duration=duration,
            control_policy=initial_policy,
            max_num_amb_cond=self.max_num_amb_cond)
        
        ctrl_vars = list(opt_mgr.control_policy.output_shape)
        
        # constrained_acc_metrics = list(control_eval_system.accumulated_constraint.output_variables)

        ambient_condition_sample = ambient_condition_statistics.systematic_sample(N_max=self.max_num_amb_cond)

        upper_bound_constraints = control_eval_system.accumulated_constraint.upper_bound_constraints()
        num_constraints = len(upper_bound_constraints)
        lagrangian_lambdas = np.ones(num_constraints)
        subgradient = np.empty(num_constraints)
        # outer-iteration counter
        for t in range(self.max_iter):
            expected_constraint_eval = np.zeros(num_constraints)
            for prob_weight, ambient_condition in ambient_condition_sample.weighted_variables_iter():
                
                # Separate optimization for each ambient condition
                def cost_function(ctrl_setpoints_vec):
                    ctrl_setpoints = {ctrl_var: ctrl for ctrl_var, ctrl in zip(ctrl_vars, ctrl_setpoints_vec)}
                    acc_metrics = opt_mgr.control_eval_system.acc_metrics_from_ambient_cond(
                        ambient_condition=ambient_condition,
                        control_setpoints=ctrl_setpoints,
                        duration=opt_mgr.duration)
                    scalar_objective = opt_mgr.control_eval_system.multi_metrics_reduction.evaluate(
                        acc_metrics=acc_metrics)
                    # First assume that we have an objective function (which we want to maximise),
                    # and upper-bound constraints which must not be exceded.
                    if not opt_mgr.control_eval_system.multi_metrics_reduction.maximize:
                        scalar_objective *= -1                
                    for lagrangian_lambda, ub_constraint in zip(lagrangian_lambdas, upper_bound_constraints):
                        scalar_objective -= lagrangian_lambda * ub_constraint.constraint_fun(acc_metrics)
                    # Scipy will minimize a cost function, hence take the negative value
                    return - scalar_objective
                
                # control constraints
                get_ctrl_setpoints_for_ac = lambda x : {
                ctrl_var: x[i_ctrl] for i_ctrl, ctrl_var in enumerate(opt_mgr.control_policy.control_variables)}
                constraints = opt_mgr.control_eval_system.control_constraint.scipy_constraint(
                    eval_constraint_from_x=get_ctrl_setpoints_for_ac)

                # Minimize Lagrangian
                x0 = opt_mgr.control_policy.get_ctrl_parameters(ambient_condition=ambient_condition)
                res = minimize(cost_function,
                               x0,
                               method=self.scipy_method,
                               constraints=constraints,
                               options=self.scipy_options)
                ctrl_setpoints_vec = res.x
                opt_mgr.control_policy.set_ctrl_parameters_from_vector(x=ctrl_setpoints_vec,
                                                            ambient_condition=ambient_condition)
                
                # Recompute final accumulated metrics
                acc_metrics = opt_mgr.control_eval_system.acc_metrics_from_ambient_cond(
                    ambient_condition=ambient_condition,
                    control_setpoints={ctrl_var: setpoint for ctrl_var, setpoint in zip(ctrl_vars, ctrl_setpoints_vec)},
                    duration=opt_mgr.duration)
                # Constraint evaluation
                for i_constraint, ub_constraint in enumerate(upper_bound_constraints):
                    expected_constraint_eval[i_constraint] += prob_weight * ub_constraint.constraint_fun(acc_metrics)
            
            # update lagrangian multipliers
            for i_constraint, ub_constraint in enumerate(upper_bound_constraints):
                subgradient[i_constraint] = ub_constraint.upper_bound - expected_constraint_eval[i_constraint]
                step = subgradient[i_constraint] * self.alpha_0 / np.sqrt(t + 1)
                lagrangian_lambdas[i_constraint] -= step
            lagrangian_lambdas[lagrangian_lambdas < 0] = 0
            t += 1

            if np.linalg.norm(subgradient, ord=np.inf) < self.subgradient_tol:
                break

        return opt_mgr.control_policy
        
     