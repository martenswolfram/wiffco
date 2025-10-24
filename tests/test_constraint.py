import pathlib
import pytest
import numpy as np
from twain_wifco.config import constraint_from_json
from twain_wifco.interface import (
    Control,
    AccumulatedMetric,
    DataPoint)

    
def test_linear_control_constraint():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "linear_control_constraint.json"
    linear_control_constraint = constraint_from_json(json_path=json_path)
        
    # Constraint evaluation
    constraint_satisfied = linear_control_constraint.evaluate_satisfied(
        constr_input_data=DataPoint(
            {
                Control.YAW_STEERING: np.array([0]),
                Control.POWER_REGULATION: np.array([0])
            }
        ))
    assert constraint_satisfied is not True
    
    constraint_satisfied = linear_control_constraint.evaluate_satisfied(
        constr_input_data=DataPoint(
            {
                Control.YAW_STEERING: np.array([0]),
                Control.POWER_REGULATION: np.array([2])
            }
        ))
    assert constraint_satisfied is True


def test_acc_metrics_constraint():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "linear_accumulated_constraint.json"
    linear_accumulated_constraint = constraint_from_json(json_path=json_path)
        
    # Initialization
    assert linear_accumulated_constraint.component_name == "linear_accumulated_constraint"
    assert linear_accumulated_constraint.bounds.lower_bound.order == \
        [AccumulatedMetric.ACCRUED_DAMAGE]
    
    assert np.array_equal(linear_accumulated_constraint.bounds.
                          lower_bound.data[AccumulatedMetric.ACCRUED_DAMAGE],
                          np.array([-np.inf]))
    
    # Constraint evaluation
    constraint_satisfied = linear_accumulated_constraint.evaluate_satisfied(
        constr_input_data=DataPoint({AccumulatedMetric.ACCRUED_DAMAGE: np.array([700]),
                                     AccumulatedMetric.REVENUE: np.array([0])}))
    assert constraint_satisfied is False
    
    constraint_satisfied = linear_accumulated_constraint.evaluate_satisfied(
        constr_input_data=DataPoint({AccumulatedMetric.ACCRUED_DAMAGE: np.array([300]),
                                     AccumulatedMetric.REVENUE: np.array([0])}))
    assert constraint_satisfied is True