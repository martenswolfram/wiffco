import pathlib
from twain_wifco.config import (
    statistics_from_json,
    metrics_accumulation_from_json)
from twain_wifco.interface import AccumulatedMetric

    
def test_simple_product_accumulation():
    
    test_data_folder = pathlib.Path(__file__).parent / "data"
    # Aggregate statistics
    json_path = test_data_folder / "discrete_aggregate_statistics.json"
    aggregate_statistics = statistics_from_json(json_path=json_path)
    
    # Metrics accumulation
    revenue_damage_accumulation = metrics_accumulation_from_json(
        json_path=(test_data_folder / "revenue_damage_accumulation.json"))

    # Initialization
    assert revenue_damage_accumulation.component_name == "revenue_damage_accumulation"
    
    # Output accumulation
    duration = 20
    expected_accumulated_metrics = revenue_damage_accumulation.expected_value(
        aggregate_statistics=aggregate_statistics,
        duration=duration)
    assert expected_accumulated_metrics[AccumulatedMetric.REVENUE] > 0
    assert expected_accumulated_metrics[AccumulatedMetric.ACCRUED_DAMAGE] > 0
    