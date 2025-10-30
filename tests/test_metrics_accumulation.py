import pathlib
from twain_wifco.config import (
    statistics_from_json,
    metrics_accumulation_from_json)
from twain_wifco.interface import AccumulatedMetric, Aggregated

    
def test_simple_product_accumulation():
    
    test_data_folder = pathlib.Path(__file__).parent / "data"
    # Aggregate statistics
    json_path = test_data_folder / "statistics_discrete_aggregate.jsonc"
    aggregate_statistics = statistics_from_json(json_path=json_path)
    
    # Metrics accumulation
    revenue_damage_accumulation = metrics_accumulation_from_json(
        json_path=(test_data_folder / "accumulation_revenue_damage.jsonc"))

    # Output accumulation
    expected_accumulated_metrics = revenue_damage_accumulation.expected_acc_metrics(
        aggregate_statistics=aggregate_statistics)
    assert expected_accumulated_metrics[AccumulatedMetric.REVENUE] > 0
    assert expected_accumulated_metrics[AccumulatedMetric.ACCRUED_DAMAGE] > 0