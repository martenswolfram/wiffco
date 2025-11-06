import pathlib
import numpy as np
from twain_wifco.config import (
    metrics_accumulation_from_json)
from twain_wifco.interface import (
    AccumulatedMetric, 
    Aggregate,
    DataTable)

    
def test_simple_product_accumulation():
    
    test_data_folder = pathlib.Path(__file__).parent / "data"
    # Metrics accumulation
    symbolic_damage_accumulation = metrics_accumulation_from_json(
        json_path=(test_data_folder / "discounted_integration.jsonc"))

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