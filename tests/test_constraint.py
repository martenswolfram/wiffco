import pathlib
import pytest
import numpy as np
from twain_wifco.config import constraint_from_json
from twain_wifco.interface import (
    Control,
    AccumulatedMetric)

    
def test_linear_constraint():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "linear_mixed_constraint.json"
    linear_mixed_constraint = constraint_from_json(json_path=json_path)
        
    # Initialization
    assert linear_mixed_constraint.component_name == "linear_accumulated_constraint"
    assert linear_mixed_constraint.constraint_variables == [
        Control.POWER_REGULATION,
        AccumulatedMetric.ACCRUED_DAMAGE]
    assert linear_mixed_constraint.bounds.shape == (2, 2)
    assert linear_mixed_constraint.scalar_bounds_for_var(Control.POWER_REGULATION)[0] == \
        pytest.approx(1)
    assert linear_mixed_constraint.scalar_bounds_for_var(Control.POWER_REGULATION)[1] == \
        pytest.approx(4)
    assert linear_mixed_constraint.scalar_bounds_for_var(AccumulatedMetric.ACCRUED_DAMAGE)[0] == \
        -np.inf
    assert linear_mixed_constraint.scalar_bounds_for_var(AccumulatedMetric.ACCRUED_DAMAGE)[1] == \
        pytest.approx(500)
    
    # Constraint evaluation
    constraint_eval = linear_mixed_constraint.evaluate(
        constr_vars={AccumulatedMetric.ACCRUED_DAMAGE: 800,
                     Control.POWER_REGULATION: 0})
    assert constraint_eval.upper_diff == pytest.approx(np.array([-4, 300]))
    assert constraint_eval.lower_diff == pytest.approx(np.array([-1, np.inf]))
    assert constraint_eval.satisfied() is not True
    constraint_eval = linear_mixed_constraint.evaluate(
        constr_vars={AccumulatedMetric.ACCRUED_DAMAGE: 300,
                     Control.POWER_REGULATION: 2})
    assert constraint_eval.upper_diff == pytest.approx(np.array([-2, -200]))
    assert constraint_eval.lower_diff == pytest.approx(np.array([1, np.inf]))
    assert constraint_eval.satisfied() is True
    
    # # scipy constraint:
    # compute_acc_metrics_fun = lambda x: 
    # scipy_constraint = linear_accumulated_constraint.scipy_constraint(compute_acc_metrics_fun=compute_acc_metrics_fun)
    

    