import pathlib
import pytest
import numpy as np
from twain_wifco.config import constraint_from_json
from twain_wifco.interface import (
    Control,
    AccumulatedMetric,
    DataPoint)

    
# def test_linear_control_constraint():
#     test_data_folder = pathlib.Path(__file__).parent / "data"
#     json_path = test_data_folder / "linear_control_constraint.json"
#     linear_control_constraint = constraint_from_json(json_path=json_path)
        
#     # Initialization
#     assert linear_control_constraint.component_name == "linear_control_constraint"
#     assert linear_control_constraint.constraint_variables == \
#         [Control.POWER_REGULATION, Control.YAW_STEERING]
#     assert np.array_equal(linear_control_constraint.stacked_lower_bound,
#                           np.array([1, -np.inf]))
#     assert np.array_equal(linear_control_constraint.stacked_upper_bound,
#                           np.array([4, np.inf]))
        
#     # Constraint evaluation
#     violated_constraint_eval = linear_control_constraint.evaluate(
#         constr_values={Control.YAW_STEERING: np.array([0]),
#                             Control.POWER_REGULATION: np.array([0])})
#     assert np.array_equal(violated_constraint_eval.lower_diff,
#                           np.array([-1, np.inf]))
#     assert np.array_equal(violated_constraint_eval.upper_diff,
#                           np.array([-4, -np.inf]))
#     assert violated_constraint_eval.satisfied() is not True
    
#     satisfied_constraint_eval = linear_control_constraint.evaluate(
#         constr_values={Control.YAW_STEERING: np.array([0]),
#                      Control.POWER_REGULATION: np.array([2])})
#     assert np.array_equal(satisfied_constraint_eval.lower_diff,
#                           np.array([1, np.inf]))
#     assert np.array_equal(satisfied_constraint_eval.upper_diff,
#                           np.array([-2, -np.inf]))
#     assert satisfied_constraint_eval.satisfied() is True


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
    violated_constraint_eval = linear_accumulated_constraint.evaluate(
        constr_input_data=DataPoint({AccumulatedMetric.ACCRUED_DAMAGE: np.array([700]),
                                 AccumulatedMetric.REVENUE: np.array([0])}))
    assert np.array_equal(violated_constraint_eval.lower_diff,
                          np.array([np.inf]))
    assert np.array_equal(violated_constraint_eval.upper_diff,
                          np.array([200]))
    assert violated_constraint_eval.satisfied() is not True
    
    satisfied_constraint_eval = linear_accumulated_constraint.evaluate(
        constr_input_data=DataPoint({AccumulatedMetric.ACCRUED_DAMAGE: np.array([300]),
                                     AccumulatedMetric.REVENUE: np.array([0])}))
    assert np.array_equal(satisfied_constraint_eval.lower_diff,
                          np.array([np.inf]))
    assert np.array_equal(satisfied_constraint_eval.upper_diff,
                          np.array([-200]))
    assert satisfied_constraint_eval.satisfied() is True
    