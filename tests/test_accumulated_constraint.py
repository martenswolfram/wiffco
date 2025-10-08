import pathlib
import pytest
from twain_wifco.config import accumulated_constraint_from_json
from twain_wifco.interface import AccumulatedMetric

    
def test_linear_constraint():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "linear_accumulated_constraint.json"
    linear_accumulated_constraint = accumulated_constraint_from_json(json_path=json_path)
        
    # Initialization
    assert linear_accumulated_constraint.component_name == "linear_accumulated_constraint"
    assert linear_accumulated_constraint.constraint_mappings.keys() == set([AccumulatedMetric.ACCRUED_DAMAGE])
    assert linear_accumulated_constraint.constraint_mappings[AccumulatedMetric.ACCRUED_DAMAGE].lower_bound is None
    assert linear_accumulated_constraint.constraint_mappings[AccumulatedMetric.ACCRUED_DAMAGE].upper_bound == pytest.approx(500)
    
    # Constraint evaluation
    constraint_evals = linear_accumulated_constraint.evaluate(acc_metrics={AccumulatedMetric.ACCRUED_DAMAGE: 800})
    assert constraint_evals.keys() == set([AccumulatedMetric.ACCRUED_DAMAGE])
    assert constraint_evals[AccumulatedMetric.ACCRUED_DAMAGE].upper_diff == pytest.approx(300)
    assert constraint_evals[AccumulatedMetric.ACCRUED_DAMAGE].lower_diff is None
    constraint_evals = linear_accumulated_constraint.evaluate(acc_metrics={AccumulatedMetric.ACCRUED_DAMAGE: 300})
    assert constraint_evals.keys() == set([AccumulatedMetric.ACCRUED_DAMAGE])
    assert constraint_evals[AccumulatedMetric.ACCRUED_DAMAGE].upper_diff == pytest.approx(-200)
    assert constraint_evals[AccumulatedMetric.ACCRUED_DAMAGE].lower_diff is None
        
    

    