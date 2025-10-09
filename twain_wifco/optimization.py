from typing import Dict, Any
from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
import itertools
from enum import Enum
from twain_wifco.interface import Ambient, Control, AccumulatedMetric
from twain_wifco.statistics import (
    Statistics,
    DiscreteStatisticsParams,
    DiscreteStatistics)
from twain_wifco.control_input import (
    ControlPolicy,
    DiscreteControlPolicyParams,
    DiscreteControlPolicy)
from twain_wifco.plant_model import PlantModel
from twain_wifco.aggregation import Aggregation
from twain_wifco.metrics_accumulation import MetricsAccumulation
from twain_wifco.accumulated_constraint import AccumulatedConstraint
from twain_wifco.multi_metrics_reduction import MultiMetricsReduction

class AccMetricsEvaluationCache:
    def __init__(self,
                 ambient_condition_statistics: Statistics,
                 control_policy: ControlPolicy,
                 duration: int,
                 acc_metrics: Dict[AccumulatedMetric, float]):
        self.ambient_condition_statistics = deepcopy(ambient_condition_statistics)
        self.control_policy = deepcopy(control_policy)
        self.duration = duration
        self.acc_metrics = deepcopy(acc_metrics)

    def matches(self,
                ambient_condition_statistics: Statistics,
                control_policy: ControlPolicy,
                duration: int):
        return False
        return self.duration == duration and \
            self.control_policy == control_policy and \
            self.ambient_condition_statistics == ambient_condition_statistics

class ControlEvaluationSystem:
    def __init__(self,
                 name: str, 
                 plant_model: PlantModel,
                 aggregation: Aggregation,
                 metrics_accumulation: MetricsAccumulation,
                 accumulated_constraint: AccumulatedConstraint,
                 multi_metrics_reduction: MultiMetricsReduction):
        
        # Parameters
        self.name = name
        self.plant_model = plant_model
        self.aggregation = aggregation
        self.metrics_accumulation = metrics_accumulation
        self.accumulated_constraint = accumulated_constraint
        self.multi_metrics_reduction = multi_metrics_reduction

        # Cache
        self.acc_metrics_eval_cache = None

    def evaluate_ambient_condition(self,
                                   ambient_condition: Dict[Ambient, float],
                                   control_setpoints: Dict[Control, float]):
        
        model_output = self.plant_model.evaluate(
             meteorological_condition=ambient_condition,
             control_input=control_setpoints)    
        return self.aggregation.compute_aggregate(model_output=model_output,
                                                  ambient_condition=ambient_condition,
                                                  control_setpoints=control_setpoints)
    
    def expected_acc_metrics(self,
                             ambient_condition_statistics: Statistics,
                             control_policy: ControlPolicy,
                             duration: int):
        
        if self.acc_metrics_eval_cache is not None and \
            self.acc_metrics_eval_cache.matches(
                ambient_condition_statistics=ambient_condition_statistics,
                control_policy=control_policy,
                duration=duration):
            return self.acc_metrics_eval_cache.acc_metrics
        
        ambient_condition_sample = ambient_condition_statistics.systematic_sample()

        aggregated_variables = self.aggregation.output_variables
        aggregated_support_values = np.empty(shape=(ambient_condition_sample.N, len(aggregated_variables)))
        for i, ambient_condition in enumerate(ambient_condition_sample):
            control_setpoints = control_policy.get_control_setpoints(ambient_condition=ambient_condition)
            model_output = self.plant_model.evaluate(meteorological_condition=ambient_condition,
                                                     control_input=control_setpoints)
            aggregated = self.aggregation.compute_aggregate(model_output=model_output,
                                                            ambient_condition=ambient_condition,
                                                            control_setpoints=control_setpoints)
            aggregated_support_values[i, :] = [aggregated[aggr_var] for aggr_var in aggregated_variables]

        discrete_statistics_params = DiscreteStatisticsParams(
            support_variables=aggregated_variables,
            prevalence = ambient_condition_sample.normalized_weights,
            support_points=aggregated_support_values)
        
        new_name = ambient_condition_statistics.component_name + "_transformed_to_aggregated"
        discrete_aggregate_statistics = DiscreteStatistics(statistics_name=new_name,
                                                           statistics_params=discrete_statistics_params)

        expected_acc_metrics = self.metrics_accumulation.expected_value(
            aggregate_statistics=discrete_aggregate_statistics,
            duration=duration)
        
        # Cache result
        self.acc_metrics_eval_cache = AccMetricsEvaluationCache(
            ambient_condition_statistics=ambient_condition_statistics,
            control_policy=control_policy,
            duration=duration,
            acc_metrics=expected_acc_metrics)
        
        return expected_acc_metrics
    

    def acc_metrics_eval(self,
                         x,
                         ambient_condition_statistics: Statistics,
                         duration: int):
        ambient_condition_sample = ambient_condition_statistics.systematic_sample()
        num_ambient_conditions = ambient_condition_sample.N
        ctrl_vars = list(self.plant_model.input_of_type(t=Control))
        num_ctrls = len(ctrl_vars)        
        
        # Create discrete control policy
        amb_cond_ctrl_setpoints = np.reshape(x, shape=(num_ambient_conditions, num_ctrls))
        discrete_control_policy_params = DiscreteControlPolicyParams(
            ambient_variables=ambient_condition_sample.support_variables,
            ambient_conditions_support=ambient_condition_sample.support_values,
            control_inputs=ctrl_vars,
            control_setpoints=amb_cond_ctrl_setpoints)
        discrete_control_policy = DiscreteControlPolicy(
            policy_name="optimized_discrete_control_policy",
            policy_params=discrete_control_policy_params)
        # compute expected accumulated metrics
        return self.expected_acc_metrics(ambient_condition_statistics=ambient_condition_statistics,
                                            control_policy=discrete_control_policy,
                                            duration=duration)
    
    
    def discrete_policy_cost_function(self,
                                      ambient_condition_statistics: Statistics,
                                      duration: int):
        
        return cost_function
    
    def scipy_constraints(self,
                          ambient_condition_statistics: Statistics,
                          duration: int):
        pass
        
        
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

class GridSearchParams:
    def __init__(self,
                 num_ambient_conditions: int,
                 control_setpoints: Dict[Control, np.ndarray]):
        self.num_ambient_conditions = num_ambient_conditions
        self.control_setpoints = control_setpoints
        
def grid_search_params_from_dict(param_dict: Dict[str, Dict | Any]):
    num_ambient_conditions = param_dict["num_ambient_conditions"]
    control_setpoints = {}
    for ctrl_var, setpoints in param_dict["control_setpoints"].items():
        control_setpoints[Control(ctrl_var)] = np.array(setpoints)
    return GridSearchParams(num_ambient_conditions=num_ambient_conditions,
                            control_setpoints=control_setpoints)


class GridSearch(ControlPolicyOptimization):
    def __init__(self,
                 optimization_name: str,
                 optimization_params: GridSearchParams):
        super().__init__(optimization_name=optimization_name)
        self.num_ambient_conditions = optimization_params.num_ambient_conditions
        self.control_setpoints = optimization_params.control_setpoints

    def optimize_policy(self,
                        control_eval_system: ControlEvaluationSystem,
                        ambient_condition_statistics: Statistics,
                        duration: int,
                        initial_policy: ControlPolicy | None = None):
        
        print(f"GridSearch: Find optimal control policy")
        ambient_condition_sample = ambient_condition_statistics.systematic_sample(N=self.num_ambient_conditions)
        num_ambient_conditions = ambient_condition_sample.N
        ctrl_vars = list(self.control_setpoints.keys())
        ctrl_setpoint_vectors = [
            self.control_setpoints[ctrl_var] for ctrl_var in ctrl_vars]
        
        ctrl_setpoint_combinations = list(itertools.product(*ctrl_setpoint_vectors))
        num_ctrl_settings = len(ctrl_setpoint_combinations)
        
        aggregate_evaluations = {
            agg_var: np.empty(shape=(num_ambient_conditions, num_ctrl_settings)) \
            for agg_var in control_eval_system.aggregation.output_variables
            }
        print(f"Number of ambient conditions: {num_ambient_conditions}.")
        print(f"Number of ctrl settings: {num_ctrl_settings}.")
        num_eval = num_ctrl_settings * num_ambient_conditions
        print(f"Performing {num_eval} system evaluations.")
        
        for n_amb, ambient_condition_vec in enumerate(ambient_condition_sample.support_values):
            ambient_condition={amb_var: val for amb_var, val in \
                               zip(ambient_condition_sample.support_variables, ambient_condition_vec)}
            for n_ctrl, ctrl_setpoints in enumerate(ctrl_setpoint_combinations):
                ctrl_setpoints={ctrl_var: val for ctrl_var, val in \
                                   zip(ctrl_vars, ctrl_setpoints)}
                aggregate = control_eval_system.evaluate_ambient_condition(
                    ambient_condition=ambient_condition,
                    control_setpoints=ctrl_setpoints)
                for agg_var in control_eval_system.aggregation.output_variables:
                    aggregate_evaluations[agg_var][n_amb, n_ctrl] = aggregate[agg_var]
        pass


        num_ctrl_policies = num_ambient_conditions**len(ctrl_setpoint_combinations)
        print(f"Evaluating {num_ctrl_policies} control policies.")
        
        aggr_vars = list(control_eval_system.aggregation.output_variables)
        
        multi_metrics_reduced = []
        constraints_satisfied = []
        control_eval_system.multi_metrics_reduction
        ctrl_settings_indices_product = itertools.product(range(num_ctrl_settings), repeat=num_ambient_conditions)
        for ctrl_indices in ctrl_settings_indices_product:
            # Each ctrl_indices corresponds to a discrete control strategy
            aggr_support_points = np.array([aggregate_evaluations[aggr_var][np.arange(num_ambient_conditions), ctrl_indices] for aggr_var in aggr_vars]).T
            discrete_stat_params = DiscreteStatisticsParams(
                support_variables=aggr_vars,
                prevalence=ambient_condition_sample.normalized_weights,
                support_points=aggr_support_points)
            discrete_aggr_stat = DiscreteStatistics(
                statistics_name="",
                statistics_params=discrete_stat_params)
            expected_acc_metrics = \
                control_eval_system.metrics_accumulation.expected_value(
                aggregate_statistics=discrete_aggr_stat,
                duration=duration)
            multi_metrics_reduced.append(control_eval_system.multi_metrics_reduction.evaluate(acc_metrics=expected_acc_metrics))
            constraint_evals = control_eval_system.accumulated_constraint.evaluate(acc_metrics=expected_acc_metrics)
            constraints_satisfied.append(all(cstr_eval.satisfied() for cstr_eval in constraint_evals.values()))
            
        # Find the optimal control strategy that satisfies the constraints
        if control_eval_system.multi_metrics_reduction.maximize:
            best_index = np.argmax(np.where(constraints_satisfied, np.array(multi_metrics_reduced), -np.inf))
        else:
            best_index = np.argmin(np.where(constraints_satisfied, np.array(multi_metrics_reduced), np.inf))
        
        # Specify discrete control strategy
        # control settings (linear index) for each ambient condition
        amb_cond_ctrl_indices = np.unravel_index(best_index, [num_ctrl_settings] * num_ambient_conditions)
        # controls setpoints for each linear index
        amb_cond_ctrl_setpoints = np.array([np.unravel_index(ctrl_index, [len(setpt_vec) for setpt_vec in ctrl_setpoint_vectors]) for ctrl_index in amb_cond_ctrl_indices])
        discrete_control_policy_params = DiscreteControlPolicyParams(ambient_variables=ambient_condition_sample.support_variables,
                                                                     ambient_conditions_support=ambient_condition_sample.support_values,
                                                                     control_inputs=ctrl_vars,
                                                                     control_setpoints=amb_cond_ctrl_setpoints)
        return DiscreteControlPolicy(policy_name="optimized_discrete_control_policy",
                                     policy_params=discrete_control_policy_params)
    
    def _equals_specific(self, other) -> bool:
        raise NotImplementedError("GridSearch: _equals_specific not implemented.")


class SimultaneousOptimizationParams:
    def __init__(self,
                 num_ambient_conditions: int,
                 max_iter: int):
        self.max_iter = max_iter
        self.num_ambient_conditions = num_ambient_conditions

def simultaneous_optimization_params_from_dict(param_dict: Dict[str, Any]):
    num_ambient_conditions = param_dict["num_ambient_conditions"]
    max_iter = param_dict["max_iter"]
    return SimultaneousOptimizationParams(
        num_ambient_conditions=num_ambient_conditions,
        max_iter=max_iter)

class SimultaneousOptimizationManager:
    def __init__(self,
                 control_eval_system: ControlEvaluationSystem,
                 ambient_condition_statistics: Statistics,
                 duration: int,
                 control_policy: ControlPolicy):
        self.control_eval_system = control_eval_system
        self.ambient_condition_statistics = ambient_condition_statistics
        self.duration = duration
        self.control_policy = control_policy

    def acc_metrics_from_x(self, x):
        self.control_policy.set_from_x_vector(x)
        return self.control_eval_system.expected_acc_metrics(
            ambient_condition_statistics=self.ambient_condition_statistics,
            control_policy=self.control_policy,
            duration=self.duration)


class SimultaneousOptimization(ControlPolicyOptimization):
    def __init__(self,
                 optimization_name: str,
                 optimization_params: SimultaneousOptimizationParams):
        super().__init__(optimization_name=optimization_name)
        self.num_ambient_conditions = optimization_params.num_ambient_conditions
        self.max_iter = optimization_params.max_iter

    def optimize_policy(self,
                        control_eval_system: ControlEvaluationSystem,
                        ambient_condition_statistics: Statistics,
                        duration: int,
                        initial_policy: ControlPolicy | None = None):
        
        # OptimizationManager
        opt_mgr = SimultaneousOptimizationManager(
            control_eval_system=control_eval_system,
            ambient_condition_statistics=ambient_condition_statistics,
            duration=duration,
            control_policy=initial_policy)
        
        # define cost function
        cost_function = \
            opt_mgr.control_eval_system.multi_metrics_reduction.cost_function(
                eval_acc_metrics_from_x=opt_mgr.acc_metrics_from_x)
            
        # define constraints
        constraint = \
            opt_mgr.control_eval_system.accumulated_constraint.scipy_constraint(
                eval_acc_metrics_from_x=opt_mgr.acc_metrics_from_x)
        
        x0 = initial_policy.get_x_vector()
        minimize(cost_function, x0, method='trust-constr',
               constraints=[constraint],
               options={'verbose': 1})
        
        return opt_mgr.control_policy
                        
    def _equals_specific(self, other) -> bool:
        raise NotImplementedError("SimultaneousOptimization: _equals_specific not implemented.")
