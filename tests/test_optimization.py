import pathlib
import numpy as np
import pytest
from typing import Dict
import logging
from twain_wifco.config import (
    parse_json_file,
    control_evaluation_system_from_json)
from twain_wifco.control_policy import DiscreteControlPolicy
from twain_wifco.statistics import statistics_from_dict, AmbientStatistics
from twain_wifco.interface import Control
from twain_wifco.optimization import (
    control_optimization_from_dict,
    ControlEvaluationSystem,
    GridSearch,
    SimultaneousOptimization,
    LagrangianRelaxation)

logger = logging.getLogger(__name__)

test_data_folder = pathlib.Path(__file__).parent / "data"
    
def test_control_evaluation_system():
    json_path = test_data_folder / "control_evaluation_system.jsonc"
    control_evaluation_system = control_evaluation_system_from_json(json_path=json_path)
    assert control_evaluation_system.name == "discrete_evaluation_system"

# System
json_path = test_data_folder / "control_evaluation_system.jsonc"
control_evaluation_system = control_evaluation_system_from_json(json_path=json_path)

# Ambient conditions
json_path = test_data_folder / "statistics_discrete_ambient.jsonc"
statistics_dict = parse_json_file(path=json_path)
ambient_statistics = statistics_from_dict(param_dict=statistics_dict)

def perturbed_control_policy_test(
        control_eval_system: ControlEvaluationSystem,
        ambient_statistics: AmbientStatistics,
        original_policy: DiscreteControlPolicy,
        perturbation_num: int,
        optimality_tol: float = 0,
        perturbation_scale: float = 1,
        discrete_steps: Dict[Control, float] | None = None):
    
    # Validate original result
    ambient_sample = ambient_statistics.systematic_sample()
    ambient = ambient_sample.ambient_support

    def evaluate_policy(policy: DiscreteControlPolicy):


        aggregate, instant_constraints_satisfied = \
            control_eval_system.aggregate_from_amb_constr_eval(
                ambient=ambient,
                control=policy.get_control(ambient=ambient))
    
        if len(instant_constraints_satisfied) < len(ambient):
            return False, None
    
        acc_metrics = control_eval_system.metrics_accumulation.acc_metrics(
            aggregate=aggregate)
        expected_acc_metrics = acc_metrics.expected_value(
            probabilities=ambient_sample.normalized_weights[:, np.newaxis])
        if control_eval_system.constraint_accumulated is not None:
            acc_constraints_satisfied = \
                control_eval_system.constraint_accumulated.evaluate_satisfied(
                constraint_input=expected_acc_metrics)
            
            if not len(acc_constraints_satisfied) == 1:
                return False, None

        multi_metrics_reduction = control_eval_system.multi_metrics_reduction.evaluate(
                acc_metrics=expected_acc_metrics)
            
        return True, multi_metrics_reduction
    
    constraints_satisfied_original, original_reduced_metric = evaluate_policy(
        original_policy)
    assert constraints_satisfied_original

    for _ in np.arange(perturbation_num):
        perturbed_policy = original_policy.random_perturbation(
            scale=perturbation_scale, discrete_steps=discrete_steps)

        constraints_satisfied_perturbed, perturbed_reduced_metric = evaluate_policy(
        perturbed_policy)
            
        if constraints_satisfied_perturbed:
            perturbed_is_suboptimal = (
                original_reduced_metric + optimality_tol >= perturbed_reduced_metric if \
                control_eval_system.multi_metrics_reduction.maximize else \
                original_reduced_metric - optimality_tol <= perturbed_reduced_metric
            )                        
            if not perturbed_is_suboptimal:
                logger.info(f"Suboptimality check failed: Perturbed policy has reduced"
                            f" metric value {perturbed_reduced_metric},"
                            f" vs. {original_reduced_metric} in the original metric.")
                logger.info(f"Improved policy: {perturbed_policy}")
                
            assert perturbed_is_suboptimal

# @pytest.mark.line_profile.with_args(GridSearch.optimize_policy)
def test_grid_search():
    json_path = test_data_folder / "optimization_grid_search.jsonc"
    param_dict = parse_json_file(path=json_path)
    grid_search: GridSearch = control_optimization_from_dict(
        param_dict=param_dict)
    
    # Optimization
    optimal_policy = grid_search.optimize_policy(ctrl_eval_sys=control_evaluation_system,
                                                 ambient_statistics=ambient_statistics)
    
    # Evaluate result
    perturbed_control_policy_test(control_eval_system=control_evaluation_system,
                                  ambient_statistics=ambient_statistics,
                                  original_policy=optimal_policy,
                                  perturbation_num=100,
                                  discrete_steps={Control.POWER_REGULATION: 0.5})

def test_simultaneous_optimization():
    json_path = test_data_folder / "optimization_simultaneous.jsonc"
    param_dict = parse_json_file(path=json_path)
    simultaneous_optimization: SimultaneousOptimization = \
        control_optimization_from_dict(param_dict=param_dict)
    
    # Optimization
    optimal_policy = simultaneous_optimization.optimize_policy(
        control_eval_system=control_evaluation_system,
        ambient_statistics=ambient_statistics)
    
    # Evaluate result
    perturbed_control_policy_test(control_eval_system=control_evaluation_system,
                                  ambient_statistics=ambient_statistics,
                                  original_policy=optimal_policy,
                                  perturbation_num=100,
                                  optimality_tol=1e-1,
                                  perturbation_scale=1)


def test_lagrangian_relaxation():
    json_path = test_data_folder / "optimization_lagrangian_relaxation.jsonc"
    param_dict = parse_json_file(path=json_path)
    lagrangian_relaxation: LagrangianRelaxation = \
        control_optimization_from_dict(param_dict=param_dict)
    
    # Optimization
    optimal_policy = lagrangian_relaxation.optimize_policy(
        control_eval_system=control_evaluation_system,
        ambient_statistics=ambient_statistics)
    
    # Evaluate result
    perturbed_control_policy_test(control_eval_system=control_evaluation_system,
                                  ambient_statistics=ambient_statistics,
                                  original_policy=optimal_policy,
                                  perturbation_num=1000,
                                  optimality_tol=1e-1,
                                  perturbation_scale=1)
    