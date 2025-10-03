import pathlib
import pytest
import numpy as np
from twain_wifco.config import (
    ambient_statistics_from_json,
    plant_model_from_json,
    control_policy_from_json,    
    aggregation_from_json,
    metrics_accumulation_from_json)
from twain_wifco.metrics_accumulation import ambient_to_discrete_aggregate_statistics
from twain_wifco.interface import AccumulatedMetric

    
def test_simple_product_accumulation():
    
    test_data_folder = pathlib.Path(__file__).parent / "data"
    # Ambient statistics
    discrete_ambient_statistics = ambient_statistics_from_json(
        json_path=(test_data_folder / "discrete_ambient_statistics.json"))
    # Control policy
    discrete_control_policy = control_policy_from_json(
        json_path=(test_data_folder / "discrete_control_policy.json"))
    # Plant model
    power_damage_model = plant_model_from_json(
        json_path=(test_data_folder / "power_damage_model.json")
    )
    # Aggregation
    revenue_damage_aggregation = aggregation_from_json(
        json_path=(test_data_folder / "revenue_damage_aggregation.json")
    )
    
    # Aggregate statistics
    aggregate_statistics = ambient_to_discrete_aggregate_statistics(
        ambient_condition_statistics=discrete_ambient_statistics,
        control_policy=discrete_control_policy,
        plant_model=power_damage_model,
        aggregation=revenue_damage_aggregation)    
    
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
    