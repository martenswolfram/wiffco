import pathlib
from twain_wifco.config import (
    statistics_from_json,
    plant_model_from_json,
    control_policy_from_json,    
    aggregation_from_json,
    metrics_accumulation_from_json)
from twain_wifco.data_flow import validate_data_graph

def test_data_graph():
    
    test_data_folder = pathlib.Path(__file__).parent / "data"
    # Ambient statistics
    
    discrete_ambient_statistics = statistics_from_json(
        json_path=(test_data_folder / "statistics_discrete_ambient.jsonc"))
    # Control policy
    discrete_control_policy = control_policy_from_json(
        json_path=(test_data_folder / "discrete_control_policy.jsonc"))
    # Plant model
    power_damage_model = plant_model_from_json(
        json_path=(test_data_folder / "model_scattered.jsonc"))
    # Aggregation
    revenue_damage_aggregation = aggregation_from_json(
        json_path=(test_data_folder / "aggregation_revenue_damage.jsonc"))
    # Metrics accumulation
    revenue_damage_accumulation = metrics_accumulation_from_json(
        json_path=(test_data_folder / "accumulation_revenue_damage.jsonc"))

    # Validate inputs and outputs
    validate_data_graph(ambient_statistics=discrete_ambient_statistics,
                        control_policy=discrete_control_policy,
                        plant_model=power_damage_model,
                        aggregation=revenue_damage_aggregation,
                        metrics_accumulation=revenue_damage_accumulation)
    pass
    
