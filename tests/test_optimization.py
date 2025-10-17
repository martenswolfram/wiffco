import pathlib
import pytest
import numpy as np
from twain_wifco.config import (
    control_evaluation_system_from_json,
    control_optimization_from_json,
    statistics_from_json)
from twain_wifco.control_policy import DiscreteControlPolicy
from twain_wifco.statistics import Statistics
from twain_wifco.interface import (
    Control,
    AccumulatedMetric,
    get_abs_tol)
from twain_wifco.optimization import (
    ControlEvaluationSystem,
    GridSearch,
    SimultaneousOptimization)
from twain_wifco.constraint import SeparateLinearConstraints

test_data_folder = pathlib.Path(__file__).parent / "data"
    
def test_control_evaluation_system():
    json_path = test_data_folder / "control_evaluation_system.json"
    control_evaluation_system = control_evaluation_system_from_json(json_path=json_path)
    assert control_evaluation_system.name == "discrete_evaluation_system"

# System
json_path = test_data_folder / "control_evaluation_system.json"
control_evaluation_system = control_evaluation_system_from_json(json_path=json_path)

# Ambient conditions
json_path = test_data_folder / "discrete_ambient_statistics.json"
ambient_statistics = statistics_from_json(json_path=json_path)

# Duration
duration = 20

def perturbed_discrete_control_policy_test(
        control_evaluation_system: ControlEvaluationSystem,
        ambient_condition_statistics: Statistics,
        duration: int,
        optimal_policy: DiscreteControlPolicy,
        diff: float):
    
    # Compute optimal results
    expected_accumulated_metrics = control_evaluation_system.expected_acc_metrics(
                ambient_condition_statistics=ambient_condition_statistics,
                control_policy=optimal_policy,
                duration=duration)
    optimal_reduced_metric = control_evaluation_system.multi_metrics_reduction.evaluate(acc_metrics=expected_accumulated_metrics)
    # Store original control setpoints
    original_setpoints = np.copy(optimal_policy.control_setpoints)

    # Compare with perturbed control policies
    for i, ambient_condition_setpoints in enumerate(optimal_policy.control_setpoints):
        for j in np.arange(len(optimal_policy.control_variables)):
            for perturbation in [diff, -diff]:
                # Evaluate perturbed result
                ambient_condition_setpoints[j] = original_setpoints[i, j] + perturbation
                control_setpoints = {ctrl_var: ctrl_val for ctrl_var, ctrl_val in zip(optimal_policy.control_variables,
                                                                                    optimal_policy.control_setpoints[i, :])}
                if control_evaluation_system.control_constraint.evaluate(control_setpoints).satisfied():
                    expected_accumulated_metrics = control_evaluation_system.expected_acc_metrics(
                        ambient_condition_statistics=ambient_condition_statistics,
                        control_policy=optimal_policy,
                        duration=duration)
                    reduced_metric = control_evaluation_system.multi_metrics_reduction.evaluate(acc_metrics=expected_accumulated_metrics)
                    if control_evaluation_system.accumulated_constraint.evaluate(expected_accumulated_metrics).satisfied():
                        is_suboptimal = (optimal_reduced_metric > reduced_metric if \
                            control_evaluation_system.multi_metrics_reduction.maximize else \
                            optimal_reduced_metric < reduced_metric)
                        assert is_suboptimal == True
    optimal_policy.control_setpoints = original_setpoints

def test_grid_search():
    json_path = test_data_folder / "grid_search_optimization.json"
    grid_search: GridSearch = control_optimization_from_json(json_path=json_path)
    
    # Initialization
    assert grid_search.optimization_name == "grid_search"
    assert grid_search.control_setpoint_vectors.keys() == set([Control.POWER_REGULATION])
    assert np.array_equal(grid_search.control_setpoint_vectors[Control.POWER_REGULATION],
                          np.array([[0, 1, 2, 3, 4]]))
    assert grid_search.max_num_amb_cond == 6

    # Optimization
    optimal_policy = grid_search.optimize_policy(control_eval_system=control_evaluation_system,
                                                 ambient_condition_statistics=ambient_statistics,
                                                 duration=duration)
    
    # Evaluate result
    perturbed_discrete_control_policy_test(control_evaluation_system=control_evaluation_system,
                                           ambient_condition_statistics=ambient_statistics,
                                           duration=duration,
                                           optimal_policy=optimal_policy,
                                           diff=1)

# def test_simultaneous_optimization():
#     json_path = test_data_folder / "simultaneous_optimization.json"
#     simultaneous_optimization: SimultaneousOptimization = control_optimization_from_json(json_path=json_path)

#     # Initialization
#     assert simultaneous_optimization.optimization_name == "simultaneous_optimization"
#     assert simultaneous_optimization.max_num_amb_cond == 10
    
#     # Optimization
#     optimal_policy = simultaneous_optimization.optimize_policy(
#         control_eval_system=control_evaluation_system,
#         ambient_condition_statistics=ambient_statistics,
#         duration=duration)
    
#     # Evaluate result
#     perturbed_discrete_control_policy_test(control_evaluation_system=control_evaluation_system,
#                                            ambient_condition_statistics=ambient_statistics,
#                                            duration=duration,
#                                            optimal_policy=optimal_policy,
#                                            diff=0.01)


# def test_lagrangian_relaxation():
#     json_path = test_data_folder / "lagrangian_relaxation.json"
#     lagrangian_relaxation: SimultaneousOptimization = control_optimization_from_json(json_path=json_path)

#     # Initialization
#     assert lagrangian_relaxation.optimization_name == "lagrangian_relaxation"
#     # assert lagrangian_relaxation.max_num_amb_cond == 10
    
#     # Optimization
#     optimal_policy = lagrangian_relaxation.optimize_policy(
#         control_eval_system=control_evaluation_system,
#         ambient_condition_statistics=ambient_statistics,
#         duration=duration)
    
#     # Evaluate result
#     expected_accumulated_metrics = control_evaluation_system.expected_acc_metrics(
#         ambient_condition_statistics=ambient_statistics,
#         control_policy=optimal_policy,
#         duration=duration,
#         max_num_amb_cond=lagrangian_relaxation.max_num_amb_cond)
        
#     optimal_revenue = expected_accumulated_metrics[AccumulatedMetric.REVENUE]
#     assert optimal_revenue > 0
#     linear_acc_contraints: SeparateLinearConstraints = control_evaluation_system.accumulated_constraint
#     assert expected_accumulated_metrics[AccumulatedMetric.ACCRUED_DAMAGE] == \
#         pytest.approx(
#         linear_acc_contraints.bounds[AccumulatedMetric.ACCRUED_DAMAGE][1],
#         abs=get_abs_tol(data_var=AccumulatedMetric.ACCRUED_DAMAGE))
#     pass

#     # Evaluate result
#     perturbed_discrete_control_policy_test(control_evaluation_system=control_evaluation_system,
#                                            ambient_condition_statistics=ambient_statistics,
#                                            duration=duration,
#                                            optimal_policy=optimal_policy,
#                                            diff=0.01)
    