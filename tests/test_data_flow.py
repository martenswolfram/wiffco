import pathlib
from twain_wifco.config import (
    parse_json_file,
    ambient_statistics_from_dict,
    plant_model_from_dict,
    control_policy_from_dict,    
    aggregation_from_dict,
    metrics_accumulation_from_dict)
from twain_wifco.data_flow import validate_data_graph

def test_data_graph():
    
    test_data_folder = pathlib.Path(__file__).parent / "data"
    # Ambient statistics
    discrete_ambient_statistics_dict = parse_json_file(path=(test_data_folder / "discrete_ambient_statistics.json"))
    discrete_ambient_statistics = ambient_statistics_from_dict(param_dict=discrete_ambient_statistics_dict)
    # Control policy
    discrete_control_policy_dict = parse_json_file(path=(test_data_folder / "discrete_control_policy.json"))
    discrete_control_policy = control_policy_from_dict(param_dict=discrete_control_policy_dict)
    # Plant model
    power_damage_model_dict = parse_json_file(path=(test_data_folder / "power_damage_model.json"))
    power_damage_model = plant_model_from_dict(param_dict=power_damage_model_dict)
    # Aggregation
    revenue_damage_aggregation_dict = parse_json_file(path=(test_data_folder / "revenue_damage_aggregation.json"))
    revenue_damage_aggregation = aggregation_from_dict(param_dict=revenue_damage_aggregation_dict)
    # Metrics accumulation
    revenue_damage_accumulation_dict = parse_json_file(path=(test_data_folder / "revenue_damage_accumulation.json"))
    revenue_damage_accumulation = metrics_accumulation_from_dict(param_dict=revenue_damage_accumulation_dict)

    # Validate inputs and outputs
    validate_data_graph(statistics=discrete_ambient_statistics,
                        control_policy=discrete_control_policy,
                        plant_model=power_damage_model,
                        aggregation=revenue_damage_aggregation,
                        metrics_accumulation=revenue_damage_accumulation)
    pass
    
