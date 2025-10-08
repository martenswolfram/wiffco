import pathlib
import pytest
import numpy as np
import itertools
from twain_wifco.config import (
    control_evaluation_system_from_json,
    control_optimization_from_json,
    ambient_statistics_from_json)
from twain_wifco.interface import (
    Control,
    AccumulatedMetric)
from twain_wifco.metrics_accumulation import ambient_to_discrete_aggregate_statistics
from twain_wifco.accumulated_constraint import SeparateLinearConstraints

test_data_folder = pathlib.Path(__file__).parent / "data"
    
def test_control_evaluation_system():
    json_path = test_data_folder / "control_evaluation_system.json"
    control_evaluation_system = control_evaluation_system_from_json(json_path=json_path)
    assert control_evaluation_system.name == "discrete_evaluation_system"

def test_grid_search():
    json_path = test_data_folder / "grid_search_optimization.json"
    grid_search = control_optimization_from_json(json_path=json_path)
    
    # Initialization
    assert grid_search.optimization_name == "grid_search"
    assert grid_search.control_setpoints.keys() == set([Control.POWER_REGULATION])
    assert grid_search.control_setpoints[Control.POWER_REGULATION] == pytest.approx(np.array([0, 1, 2, 3, 4]))
    assert grid_search.num_ambient_conditions == 6

    # System
    json_path = test_data_folder / "control_evaluation_system.json"
    control_evaluation_system = control_evaluation_system_from_json(json_path=json_path)
    
    # Ambient conditions
    json_path = test_data_folder / "discrete_ambient_statistics.json"
    ambient_statistics = ambient_statistics_from_json(json_path=json_path)
    
    # Optimization
    duration = 20
    optimal_policy = grid_search.optimize_policy(control_eval_system=control_evaluation_system,
                                                 ambient_condition_statistics=ambient_statistics,
                                                 duration=duration)
    
    # Evaluate result
    # Aggregate statistics
    aggregate_statistics = ambient_to_discrete_aggregate_statistics(
        ambient_condition_statistics=ambient_statistics,
        control_policy=optimal_policy,
        plant_model=control_evaluation_system.plant_model,
        aggregation=control_evaluation_system.aggregation)
    
    # Metrics accumulation
    expected_accumulated_metrics = control_evaluation_system.metrics_accumulation.expected_value(
        aggregate_statistics=aggregate_statistics,
        duration=duration)
    optimal_revenue = expected_accumulated_metrics[AccumulatedMetric.REVENUE]
    assert optimal_revenue > 0
    linear_acc_contraints: SeparateLinearConstraints = control_evaluation_system.accumulated_constraint
    assert expected_accumulated_metrics[AccumulatedMetric.ACCRUED_DAMAGE] <= \
        linear_acc_contraints.constraint_mappings[AccumulatedMetric.ACCRUED_DAMAGE].upper_bound
    pass
    
    # Compare with perturbed control policies
    for control_setpoint in optimal_policy.control_setpoints.T:
        # Evaluate perturbed result (ramp up control)
        control_setpoint += 1
        # Aggregate statistics
        aggregate_statistics = ambient_to_discrete_aggregate_statistics(
            ambient_condition_statistics=ambient_statistics,
            control_policy=optimal_policy,
            plant_model=control_evaluation_system.plant_model,
            aggregation=control_evaluation_system.aggregation)
        
        # Metrics accumulation
        expected_accumulated_metrics = control_evaluation_system.metrics_accumulation.expected_value(
            aggregate_statistics=aggregate_statistics,
            duration=duration)
        # Constraint violated
        assert expected_accumulated_metrics[AccumulatedMetric.ACCRUED_DAMAGE] > \
            linear_acc_contraints.constraint_mappings[AccumulatedMetric.ACCRUED_DAMAGE].upper_bound

        # Evaluate perturbed result (ramp down control)
        control_setpoint -= 2
        # Aggregate statistics
        aggregate_statistics = ambient_to_discrete_aggregate_statistics(
            ambient_condition_statistics=ambient_statistics,
            control_policy=optimal_policy,
            plant_model=control_evaluation_system.plant_model,
            aggregation=control_evaluation_system.aggregation)
        
        # Metrics accumulation
        expected_accumulated_metrics = control_evaluation_system.metrics_accumulation.expected_value(
            aggregate_statistics=aggregate_statistics,
            duration=duration)
        # Sub-optimal result
        assert expected_accumulated_metrics[AccumulatedMetric.REVENUE] < optimal_revenue

        # back to original
        control_setpoint += 1

