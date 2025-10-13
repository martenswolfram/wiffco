import pathlib
import pytest
import numpy as np
from twain_wifco.config import (
    control_evaluation_system_from_json,
    control_optimization_from_json,
    statistics_from_json)
from twain_wifco.interface import (
    Control,
    AccumulatedMetric,
    get_abs_tol)
from twain_wifco.optimization import (
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

def test_grid_search():
    json_path = test_data_folder / "grid_search_optimization.json"
    grid_search: GridSearch = control_optimization_from_json(json_path=json_path)
    
    # Initialization
    assert grid_search.optimization_name == "grid_search"
    assert grid_search.control_setpoints.keys() == set([Control.POWER_REGULATION])
    assert grid_search.control_setpoints[Control.POWER_REGULATION] == pytest.approx(np.array([0, 1, 2, 3, 4]))
    assert grid_search.max_num_amb_cond == 6

    # Optimization
    optimal_policy = grid_search.optimize_policy(control_eval_system=control_evaluation_system,
                                                 ambient_condition_statistics=ambient_statistics,
                                                 duration=duration)
    
    # Evaluate result
    expected_accumulated_metrics = control_evaluation_system.expected_acc_metrics(
        ambient_condition_statistics=ambient_statistics,
        control_policy=optimal_policy,
        duration=duration)
        
    optimal_revenue = expected_accumulated_metrics[AccumulatedMetric.REVENUE]
    assert optimal_revenue > 0
    linear_acc_contraints: SeparateLinearConstraints = control_evaluation_system.accumulated_constraint
    assert expected_accumulated_metrics[AccumulatedMetric.ACCRUED_DAMAGE] <= \
        linear_acc_contraints.scalar_bounds_for_var(AccumulatedMetric.ACCRUED_DAMAGE)[1]
    pass
    
    # Compare with perturbed control policies
    for control_setpoint in optimal_policy.control_setpoints:
        # Evaluate perturbed result (ramp up control)
        control_setpoint += 1
        expected_accumulated_metrics = control_evaluation_system.expected_acc_metrics(
            ambient_condition_statistics=ambient_statistics,
            control_policy=optimal_policy,
            duration=duration)
            
        # Constraint violated
        assert expected_accumulated_metrics[AccumulatedMetric.ACCRUED_DAMAGE] > \
            linear_acc_contraints.scalar_bounds_for_var(AccumulatedMetric.ACCRUED_DAMAGE)[1]

        # Evaluate perturbed result (ramp down control)
        control_setpoint -= 2
        expected_accumulated_metrics = control_evaluation_system.expected_acc_metrics(
            ambient_condition_statistics=ambient_statistics,
            control_policy=optimal_policy,
            duration=duration)
        # Sub-optimal result
        assert expected_accumulated_metrics[AccumulatedMetric.REVENUE] < optimal_revenue

        # back to original
        control_setpoint += 1


def test_simultaneous_optimization():
    json_path = test_data_folder / "simultaneous_optimization.json"
    simultaneous_optimization: SimultaneousOptimization = control_optimization_from_json(json_path=json_path)

    # Initialization
    assert simultaneous_optimization.optimization_name == "simultaneous_optimization"
    assert simultaneous_optimization.max_num_amb_cond == 10
    
    # Optimization
    optimal_policy = simultaneous_optimization.optimize_policy(
        control_eval_system=control_evaluation_system,
        ambient_condition_statistics=ambient_statistics,
        duration=duration)
    
    # Evaluate result
    expected_accumulated_metrics = control_evaluation_system.expected_acc_metrics(
        ambient_condition_statistics=ambient_statistics,
        control_policy=optimal_policy,
        duration=duration,
        max_num_amb_cond=simultaneous_optimization.max_num_amb_cond)
        
    optimal_revenue = expected_accumulated_metrics[AccumulatedMetric.REVENUE]
    assert optimal_revenue > 0
    linear_acc_contraints: SeparateLinearConstraints = control_evaluation_system.accumulated_constraint
    assert expected_accumulated_metrics[AccumulatedMetric.ACCRUED_DAMAGE] == \
        pytest.approx(
        linear_acc_contraints.scalar_bounds_for_var(AccumulatedMetric.ACCRUED_DAMAGE)[1],
        abs=get_abs_tol(data_var=AccumulatedMetric.ACCRUED_DAMAGE))
    pass
    

def test_lagrangian_relaxation():
    json_path = test_data_folder / "lagrangian_relaxation.json"
    lagrangian_relaxation: SimultaneousOptimization = control_optimization_from_json(json_path=json_path)

    # Initialization
    assert lagrangian_relaxation.optimization_name == "lagrangian_relaxation"
    # assert lagrangian_relaxation.max_num_amb_cond == 10
    
    # Optimization
    optimal_policy = lagrangian_relaxation.optimize_policy(
        control_eval_system=control_evaluation_system,
        ambient_condition_statistics=ambient_statistics,
        duration=duration)
    
    # Evaluate result
    expected_accumulated_metrics = control_evaluation_system.expected_acc_metrics(
        ambient_condition_statistics=ambient_statistics,
        control_policy=optimal_policy,
        duration=duration,
        max_num_amb_cond=lagrangian_relaxation.max_num_amb_cond)
        
    optimal_revenue = expected_accumulated_metrics[AccumulatedMetric.REVENUE]
    assert optimal_revenue > 0
    linear_acc_contraints: SeparateLinearConstraints = control_evaluation_system.accumulated_constraint
    assert expected_accumulated_metrics[AccumulatedMetric.ACCRUED_DAMAGE] == \
        pytest.approx(
        linear_acc_contraints.scalar_bounds_for_var(AccumulatedMetric.ACCRUED_DAMAGE)[1],
        abs=get_abs_tol(data_var=AccumulatedMetric.ACCRUED_DAMAGE))
    pass
    