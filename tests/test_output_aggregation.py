import pathlib
import pytest
from twain_wifco.config import parse_json_file, output_aggregation_from_dict
from twain_wifco.interface import AmbientVariable, OutputVariable

    
def test_simple_product_aggregation():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "simple_output_aggregation.json"
    param_dict = parse_json_file(path=json_path)
    simple_product_aggregation = output_aggregation_from_dict(param_dict=param_dict)
        
    # Initialization
    assert simple_product_aggregation.interface.name == "Output Aggregator 'revenue_aggregation'"
    assert simple_product_aggregation.aggregate_mappings.keys() == \
        set([OutputVariable.REVENUE_RATE, OutputVariable.DAMAGE_RATE])
    assert simple_product_aggregation.aggregate_mappings[OutputVariable.REVENUE_RATE].from_model == \
        set([OutputVariable.ELECTRICAL_POWER])
    assert simple_product_aggregation.aggregate_mappings[OutputVariable.REVENUE_RATE].from_ambient == \
        set([AmbientVariable.ELECTRICITY_PRICE])
    assert simple_product_aggregation.aggregate_mappings[OutputVariable.DAMAGE_RATE].from_model == \
        set([OutputVariable.DAMAGE_RATE])
    assert simple_product_aggregation.aggregate_mappings[OutputVariable.DAMAGE_RATE].from_ambient == \
        set([])
    
    # Output aggregation
    aggregated_output = simple_product_aggregation.compute_aggregate(
        output_variables={OutputVariable.ELECTRICAL_POWER: 5,
                          OutputVariable.DAMAGE_RATE: 2},
        ambient_condition={AmbientVariable.ELECTRICITY_PRICE: 3})
    assert aggregated_output == pytest.approx({OutputVariable.REVENUE_RATE: 15,
                                               OutputVariable.DAMAGE_RATE: 2})
        
    

    