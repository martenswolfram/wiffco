import pathlib
import pytest
import numpy as np
from twain_wifco.config import constraint_from_json
from twain_wifco.interface import (
    Control,
    AccumulatedMetric)

    
def test_linear_control_constraint():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "linear_control_constraint.json"
    linear_control_constraint = constraint_from_json(json_path=json_path)
        
    # Initialization
    assert linear_control_constraint.component_name == "linear_control_constraint"
    assert linear_control_constraint.constraint_variables == [
        Control.POWER_REGULATION]
    assert linear_control_constraint.bounds.shape == (1, 2)
    assert linear_control_constraint.scalar_bounds_for_var(Control.POWER_REGULATION)[0] == \
        pytest.approx(1)
    assert linear_control_constraint.scalar_bounds_for_var(Control.POWER_REGULATION)[1] == \
        pytest.approx(4)
    
    # Constraint evaluation
    constraint_eval = linear_control_constraint.evaluate(
        constr_values_dict={Control.YAW_STEERING: 0,
                     Control.POWER_REGULATION: 0})
    assert constraint_eval.upper_diff == pytest.approx(np.array([-4]))
    assert constraint_eval.lower_diff == pytest.approx(np.array([-1]))
    assert constraint_eval.satisfied() is not True
    constraint_eval = linear_control_constraint.evaluate(
        constr_values_dict={Control.YAW_STEERING: 0,
                     Control.POWER_REGULATION: 2})
    assert constraint_eval.upper_diff == pytest.approx(np.array([-2]))
    assert constraint_eval.lower_diff == pytest.approx(np.array([ 1]))
    assert constraint_eval.satisfied() is True
    
def test_acc_metrics_constraint():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "linear_accumulated_constraint.json"
    linear_accumulated_constraint = constraint_from_json(json_path=json_path)
        
    # Initialization
    assert linear_accumulated_constraint.component_name == "linear_accumulated_constraint"
    assert linear_accumulated_constraint.constraint_variables == [
        AccumulatedMetric.ACCRUED_DAMAGE]
    assert linear_accumulated_constraint.bounds.shape == (1, 2)
    assert linear_accumulated_constraint.scalar_bounds_for_var(AccumulatedMetric.ACCRUED_DAMAGE)[0] == \
        -np.inf
    assert linear_accumulated_constraint.scalar_bounds_for_var(AccumulatedMetric.ACCRUED_DAMAGE)[1] == \
        pytest.approx(500)
    
    # Constraint evaluation
    constraint_eval = linear_accumulated_constraint.evaluate(
        constr_values_dict={AccumulatedMetric.ACCRUED_DAMAGE: 700,
                            AccumulatedMetric.REVENUE: 0})
    assert constraint_eval.upper_diff == pytest.approx(np.array([   200]))
    assert constraint_eval.lower_diff == pytest.approx(np.array([np.inf]))
    assert constraint_eval.satisfied() is not True
    constraint_eval = linear_accumulated_constraint.evaluate(
        constr_values_dict={AccumulatedMetric.ACCRUED_DAMAGE: 300,
                            AccumulatedMetric.REVENUE: 0})
    assert constraint_eval.upper_diff == pytest.approx(np.array([   -200]))
    assert constraint_eval.lower_diff == pytest.approx(np.array([ np.inf]))
    assert constraint_eval.satisfied() is True
        

    