import pathlib
import pytest
import numpy as np
from twain_wifco.config import (
    parse_json_file,
    ambient_statistics_from_dict,
    plant_model_from_dict,
    control_policy_from_dict,    
    output_aggregation_from_dict,
    output_accumulation_from_dict)
from twain_wifco.output_accumulation import ambient_to_output_statistics
from twain_wifco.interface import OutputVariable, AccumulatedMetric

    
def test_simple_product_accumulation():
    
    test_data_folder = pathlib.Path(__file__).parent / "data"
    power_model_dict = parse_json_file(path=(test_data_folder / "simple_power_model.json"))
    simple_power_model = plant_model_from_dict(param_dict=power_model_dict)
    damage_model_dict = parse_json_file(path=(test_data_folder / "simple_damage_model.json"))
    simple_damage_model = plant_model_from_dict(param_dict=damage_model_dict)
    ambient_statistics_dict = parse_json_file(path=(test_data_folder / "simple_ambient_statistics.json"))
    simple_ambient_statistics = ambient_statistics_from_dict(param_dict=ambient_statistics_dict)
    control_policy_dict = parse_json_file(path=(test_data_folder / "simple_control_policy.json"))
    simple_control_policy = control_policy_from_dict(param_dict=control_policy_dict)
    output_aggregation_dict = parse_json_file(path=(test_data_folder / "simple_output_aggregation.json"))
    simple_product_aggregation = output_aggregation_from_dict(param_dict=output_aggregation_dict)
    
    output_statistics = ambient_to_output_statistics(
        ambient_condition_statistics=simple_ambient_statistics,
        control_policy=simple_control_policy,
        plant_models=[simple_power_model, simple_damage_model],
        output_aggregation=simple_product_aggregation)

    output_accumulation_dict = parse_json_file(path=(test_data_folder / "simple_output_accumulation.json"))
    discounted_integrator_accumulation = output_accumulation_from_dict(param_dict=output_accumulation_dict)
    
    
    # Initialization
    assert discounted_integrator_accumulation.interface.name == "Output Accumulator 'discounted_revenue'"
    assert discounted_integrator_accumulation.params.discount_rates == pytest.approx({OutputVariable.REVENUE_RATE: 0.05,
                                                                                      OutputVariable.DAMAGE_RATE: 0.0})
    assert discounted_integrator_accumulation.params.in_out_mappings == {OutputVariable.REVENUE_RATE: AccumulatedMetric.REVENUE,
                                                                         OutputVariable.DAMAGE_RATE: AccumulatedMetric.ACCRUED_DAMAGE}
    
    # Output accumulation
    duration = 20
    expected_accumulated_metrics = discounted_integrator_accumulation.expected_value(
        output_statistics=output_statistics,
        duration=duration)
    assert expected_accumulated_metrics[AccumulatedMetric.REVENUE] > 0
    assert expected_accumulated_metrics[AccumulatedMetric.ACCRUED_DAMAGE] > 0
    
    
    
    # expected_accumulated_metrics
    
    # .compute_aggregate(
    #     output_variables={OutputVariable.ELECTRICAL_POWER: 5},
    #     ambient_condition={AmbientVariable.ELECTRICITY_PRICE: 3})
    # assert aggregated_output == pytest.approx({OutputVariable.REVENUE_RATE: 15})
        
    

    