import pathlib
import numpy as np
from typing import Dict
import logging
from twain_wifco.config import (
    control_evaluation_system_from_json,
    control_optimization_from_json,
    statistics_from_json)
from twain_wifco.control_policy import DiscreteControlPolicy
from twain_wifco.statistics import Statistics
from twain_wifco.interface import (
    Control,
    DataPoint)
from twain_wifco.optimization import (
    ControlEvaluationSystem,
    GridSearch,
    SimultaneousOptimization)

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
ambient_statistics = statistics_from_json(json_path=json_path)

# Duration
duration = 20

def perturbed_control_policy_test(
        control_evaluation_system: ControlEvaluationSystem,
        ambient_condition_statistics: Statistics,
        optimal_policy: DiscreteControlPolicy,
        perturbation_num: int,
        optimality_tol: float = 0,
        perturbation_scale: float = 1,
        discrete_steps: Dict[Control, float] | None = None):
    
    # Compute optimal results
    expected_accumulated_metrics, constraints_satisfied = control_evaluation_system.expected_acc_metrics_constr_eval(
                ambient_condition_statistics=ambient_condition_statistics,
                control_policy=optimal_policy)
    assert constraints_satisfied is True
    optimal_reduced_metric = control_evaluation_system.multi_metrics_reduction.evaluate(acc_metrics=expected_accumulated_metrics)
    

    for _ in np.arange(perturbation_num):
        perturbed_policy = optimal_policy.random_perturbation(scale=perturbation_scale, discrete_steps=discrete_steps)
        expected_accumulated_metrics, constraints_satisfied = \
            control_evaluation_system.expected_acc_metrics_constr_eval(
                ambient_condition_statistics=ambient_condition_statistics,
                control_policy=perturbed_policy)
        if constraints_satisfied:
            reduced_metric = control_evaluation_system.multi_metrics_reduction.evaluate(
                acc_metrics=expected_accumulated_metrics)
            perturbed_is_suboptimal = (
                optimal_reduced_metric + optimality_tol >= reduced_metric if \
                control_evaluation_system.multi_metrics_reduction.maximize else \
                optimal_reduced_metric - optimality_tol <= reduced_metric)                        
            if not perturbed_is_suboptimal:
                logger.info(f"Suboptimality check failed: Perturbed policy has reduced metric value {reduced_metric},"
                            f" vs. {optimal_reduced_metric} in the original metric.")
                logger.info(f"Improved policy: {perturbed_policy}")
                
            assert perturbed_is_suboptimal

def test_grid_search():
    json_path = test_data_folder / "optimization_grid_search.jsonc"
    grid_search: GridSearch = control_optimization_from_json(json_path=json_path)
    
    # Optimization
    optimal_policy = grid_search.optimize_policy(control_eval_system=control_evaluation_system,
                                                 ambient_condition_statistics=ambient_statistics)
    
    # Evaluate result
    perturbed_control_policy_test(control_evaluation_system=control_evaluation_system,
                                  ambient_condition_statistics=ambient_statistics,
                                  optimal_policy=optimal_policy,
                                  perturbation_num=100,
                                  discrete_steps={Control.POWER_REGULATION: 1})

def test_simultaneous_optimization():
    json_path = test_data_folder / "optimization_simultaneous.jsonc"
    simultaneous_optimization: SimultaneousOptimization = control_optimization_from_json(json_path=json_path)
    
    # Optimization
    optimal_policy = simultaneous_optimization.optimize_policy(
        control_eval_system=control_evaluation_system,
        ambient_condition_statistics=ambient_statistics)
    
    # Evaluate result
    perturbed_control_policy_test(control_evaluation_system=control_evaluation_system,
                                  ambient_condition_statistics=ambient_statistics,
                                  optimal_policy=optimal_policy,
                                  perturbation_scale=1,
                                  perturbation_num=100,
                                  optimality_tol=1e-4)


def test_lagrangian_relaxation():
    json_path = test_data_folder / "optimization_lagrangian_relaxation.jsonc"
    lagrangian_relaxation: SimultaneousOptimization = control_optimization_from_json(json_path=json_path)
    
    # Optimization
    optimal_policy = lagrangian_relaxation.optimize_policy(
        control_eval_system=control_evaluation_system,
        ambient_condition_statistics=ambient_statistics)
    
    # Evaluate result
    perturbed_control_policy_test(control_evaluation_system=control_evaluation_system,
                                  ambient_condition_statistics=ambient_statistics,
                                  optimal_policy=optimal_policy,
                                  perturbation_scale=1,
                                  perturbation_num=100,
                                  optimality_tol=1e-3)
    