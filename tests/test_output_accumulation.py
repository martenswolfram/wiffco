import pathlib
import numpy as np
from twain_wifco.config import (
    parse_json_file,
    ambient_statistics_from_dict,
    wind_farm_model_from_dict,
    control_policy_from_dict,    
    output_aggregation_from_dict)
from twain_wifco.output_accumulation import ambient_to_output_statistics
# from twain_wifco.interface import AmbientVariable, OutputVariable

    
def test_simple_product_accumulation():
    
    test_data_folder = pathlib.Path(__file__).parent / "data"
    model_dict = parse_json_file(path=(test_data_folder / "simple_power_model.json"))
    simple_power_model = wind_farm_model_from_dict(param_dict=model_dict)
    ambient_statistics_dict = parse_json_file(path=(test_data_folder / "simple_ambient_statistics.json"))
    simple_ambient_statistics = ambient_statistics_from_dict(param_dict=ambient_statistics_dict)
    control_policy_dict = parse_json_file(path=(test_data_folder / "simple_control_policy.json"))
    simple_control_policy = control_policy_from_dict(param_dict=control_policy_dict)
    output_aggregation_dict = parse_json_file(path=(test_data_folder / "simple_output_aggregation.json"))
    simple_product_aggregation = output_aggregation_from_dict(param_dict=output_aggregation_dict)
    
    output_statistics = ambient_to_output_statistics(
        ambient_condition_statistics=simple_ambient_statistics,
        control_policy=simple_control_policy,
        wind_farm_model=simple_power_model,
        output_aggregation=simple_product_aggregation)
    pass
    
    # # Initialization
    # assert simple_product_aggregation.interface.name == "Output Aggregator 'revenue_aggregation'"
    # assert simple_product_aggregation.from_model == \
    #     set([OutputVariable.ELECTRICAL_POWER])
    # assert simple_product_aggregation.from_context == \
    #     set([AmbientVariable.ELECTRICITY_PRICE])
    # assert simple_product_aggregation.single_output == OutputVariable.REVENUE_RATE

    # # Output aggregation
    # aggregated_output = simple_product_aggregation.compute_aggregate(
    #     output_variables={OutputVariable.ELECTRICAL_POWER: 5},
    #     ambient_condition={AmbientVariable.ELECTRICITY_PRICE: 3})
    # assert aggregated_output == pytest.approx({OutputVariable.REVENUE_RATE: 15})
        
    

    