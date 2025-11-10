import pathlib
import numpy as np
from twain_wifco.config import parse_json_file
from twain_wifco.metrics_accumulation import metrics_accumulation_from_dict
from twain_wifco.interface import (
    AccumulatedMetric, 
    Aggregate,
    DataTable)

    
def test_simple_product_accumulation():
    
    test_data_folder = pathlib.Path(__file__).parent / "data"
    # Metrics accumulation
    param_dict = parse_json_file(
        path=test_data_folder / "discounted_integration.jsonc")
    symbolic_damage_accumulation = metrics_accumulation_from_dict(
        param_dict=param_dict)

    aggregate = DataTable({Aggregate.REVENUE_RATE: np.array([[15, 15],
                                                              [3,  3]]),
                           Aggregate.DAMAGE_RATE: np.array([[4, 4],
                                                            [2, 2]])})
    # Output accumulation
    accumulated_metrics = symbolic_damage_accumulation.acc_metrics(aggregate=aggregate)

    expected_acc_metrics = accumulated_metrics.expected_value(probabilities=np.array([[0.25, 0.75, 0.5],
                                                                                      [0.75, 0.25, 0.5]]))
    assert np.all(expected_acc_metrics[AccumulatedMetric.REVENUE] > 0)
    assert np.array_equal(expected_acc_metrics[AccumulatedMetric.ACCRUED_DAMAGE], np.array([[50, 50],
                                                                                            [70, 70],
                                                                                            [60, 60]])) 