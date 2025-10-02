import pathlib
import pytest
from twain_wifco.config import parse_json_file, accumulated_constraint_from_dict
from twain_wifco.interface import AccumulatedMetric

    
def test_linear_constraint():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "linear_accumulated_constraint.json"
    param_dict = parse_json_file(path=json_path)
    linear_accumulated_constraint = accumulated_constraint_from_dict(param_dict=param_dict)
        
    # Initialization
    assert linear_accumulated_constraint.component_name == "linear_accumulated_constraint"
    assert linear_accumulated_constraint.constraint_mappings.keys() == set([AccumulatedMetric.ACCRUED_DAMAGE])
    assert linear_accumulated_constraint.constraint_mappings[AccumulatedMetric.ACCRUED_DAMAGE].lower_bound is None
    assert linear_accumulated_constraint.constraint_mappings[AccumulatedMetric.ACCRUED_DAMAGE].upper_bound == pytest.approx(1000)
    
    # Constraint evaluation
    constraint_evals = linear_accumulated_constraint.evaluate(acc_metrics={AccumulatedMetric.ACCRUED_DAMAGE: 1200})
    assert constraint_evals.keys() == set([AccumulatedMetric.ACCRUED_DAMAGE])
    assert constraint_evals[AccumulatedMetric.ACCRUED_DAMAGE].upper_diff == pytest.approx(200)
    assert constraint_evals[AccumulatedMetric.ACCRUED_DAMAGE].lower_diff is None
    constraint_evals = linear_accumulated_constraint.evaluate(acc_metrics={AccumulatedMetric.ACCRUED_DAMAGE: 800})
    assert constraint_evals.keys() == set([AccumulatedMetric.ACCRUED_DAMAGE])
    assert constraint_evals[AccumulatedMetric.ACCRUED_DAMAGE].upper_diff == pytest.approx(-200)
    assert constraint_evals[AccumulatedMetric.ACCRUED_DAMAGE].lower_diff is None
        
    

    