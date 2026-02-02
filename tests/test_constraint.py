import pathlib
import numpy as np
from twain_wifco.config import parse_json_file
from twain_wifco.constraint import constraint_from_dict
from twain_wifco.interface import (
    Control,
    Aggregate,
    AccumulatedMetric,
    DataTable)

    
def test_control_constraint():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "constraint_control.jsonc"
    param_dict = parse_json_file(path=json_path)
    control_constraint = constraint_from_dict(param_dict=param_dict)
        
    # Constraint evaluation
    constraint_satisfied = control_constraint.evaluate_satisfied(
        constraint_input=DataTable(
            {
                Control.YAW_ANGLE_DEG: np.array([0,
                                             0]),
                Control.POWER_REGULATION: np.array([[0, 0],
                                                    [5, 5]])
            }
        ))
    assert len(constraint_satisfied) == 0
    
    constraint_satisfied = control_constraint.evaluate_satisfied(
        constraint_input=DataTable(
            {
                Control.YAW_ANGLE_DEG: np.array([0,
                                             0]),
                Control.POWER_REGULATION: np.array([[2, 2],
                                                    [3, 3]])
            }
        ))
    assert len(constraint_satisfied) == 2

def test_aggregate_constraint():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "constraint_aggregate.jsonc"
    param_dict = parse_json_file(path=json_path)
    aggregate_constraint = constraint_from_dict(param_dict=param_dict)
        
    # Constraint evaluation
    constraint_satisfied = aggregate_constraint.evaluate_satisfied(
        constraint_input=DataTable({Aggregate.DAMAGE_RATE: np.array([[41],
                                                                     [41]]),
                                    Aggregate.REVENUE_RATE: np.array([0,
                                                                      0])})
    )
    assert len(constraint_satisfied) == 0
    
    constraint_satisfied = aggregate_constraint.evaluate_satisfied(
        constraint_input=DataTable({Aggregate.DAMAGE_RATE: np.array([[30],
                                                                     [30]]),
                                    Aggregate.REVENUE_RATE: np.array([0,
                                                                      0])})
    )
    assert len(constraint_satisfied) == 2

def test_acc_metrics_constraint():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "constraint_accumulated.jsonc"
    param_dict = parse_json_file(path=json_path)
    accumulated_constraint = constraint_from_dict(param_dict=param_dict)
        
    # Constraint evaluation
    constraint_satisfied = accumulated_constraint.evaluate_satisfied(
        constraint_input=DataTable({AccumulatedMetric.ACCRUED_DAMAGE: np.array([[700],
                                                                                [700]]),
                                    AccumulatedMetric.REVENUE_EUR: np.array([0,
                                                                         0])}))
    assert len(constraint_satisfied) == 0
    
    constraint_satisfied = accumulated_constraint.evaluate_satisfied(
        constraint_input=DataTable({AccumulatedMetric.ACCRUED_DAMAGE: np.array([[300],
                                                                                [300]]),
                                    AccumulatedMetric.REVENUE_EUR: np.array([0,
                                                                         0])}))
    assert len(constraint_satisfied) == 2