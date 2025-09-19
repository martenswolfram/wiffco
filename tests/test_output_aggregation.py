import pathlib
import pytest
from twain_wiffco.config import parse_json_file, output_aggregation_from_dict
from twain_wiffco.ambient_conditions import AmbientVariable
from twain_wiffco.output_aggregation import OutputVariable

    
def test_simple_product_aggregation():
    json_path = pathlib.Path(__file__).parent / "data" / "simple_output_aggregation.json"
    param_dict = parse_json_file(path=json_path)
    simple_product_aggregation = output_aggregation_from_dict(param_dict=param_dict)
        
    # Initialization
    assert simple_product_aggregation.name == "revenue_aggregation"
    assert simple_product_aggregation.params.from_model == \
        set([OutputVariable.ELECTRICAL_POWER])
    assert simple_product_aggregation.params.from_context == \
        set([AmbientVariable.ELECTRICITY_PRICE])
    assert simple_product_aggregation.params.single_output == OutputVariable.REVENUE_RATE

    # Output aggregation
    aggregated_output = simple_product_aggregation.compute_aggregate(
        model_outputs={OutputVariable.ELECTRICAL_POWER: 5},
        ambient_condition={AmbientVariable.ELECTRICITY_PRICE: 3})
    assert aggregated_output == pytest.approx({OutputVariable.REVENUE_RATE: 15})
        
    

    