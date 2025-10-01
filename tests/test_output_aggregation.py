import pathlib
import pytest
from twain_wifco.config import parse_json_file, output_aggregation_from_dict
from twain_wifco.interface import Ambient, ModelOutput, AggregatedOutput

    
def test_simple_product_aggregation():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "simple_output_aggregation.json"
    param_dict = parse_json_file(path=json_path)
    simple_product_aggregation = output_aggregation_from_dict(param_dict=param_dict)
        
    # Initialization
    assert simple_product_aggregation.component_name == "revenue_aggregation"
    assert simple_product_aggregation.aggregate_mappings.keys() == \
        set([AggregatedOutput.REVENUE_RATE])
    assert simple_product_aggregation.aggregate_mappings[AggregatedOutput.REVENUE_RATE].from_model == \
        set([ModelOutput.ELECTRICAL_POWER])
    assert simple_product_aggregation.aggregate_mappings[AggregatedOutput.REVENUE_RATE].from_ambient == \
        set([Ambient.ELECTRICITY_PRICE])
    assert simple_product_aggregation.aggregate_mappings[AggregatedOutput.REVENUE_RATE].from_control == \
        set([])
    
    # Output aggregation
    aggregated_output = simple_product_aggregation.compute_aggregate(
        model_output={ModelOutput.ELECTRICAL_POWER: 5},
        ambient_condition={Ambient.ELECTRICITY_PRICE: 3},
        control_setpoints={})
    assert aggregated_output == pytest.approx({AggregatedOutput.REVENUE_RATE: 15})
        
    

    