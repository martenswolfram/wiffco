import pathlib
import pytest
from twain_wifco.config import aggregation_from_json
from twain_wifco.interface import Ambient, ModelOutput, Aggregated

    
def test_simple_product_aggregation():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "revenue_damage_aggregation.json"
    simple_product_aggregation = aggregation_from_json(json_path=json_path)
        
    # Initialization
    assert simple_product_aggregation.component_name == "revenue_damage_aggregation"
    assert simple_product_aggregation.aggregate_mappings.keys() == \
        set([Aggregated.REVENUE_RATE, Aggregated.DAMAGE_RATE])
    assert simple_product_aggregation.aggregate_mappings[Aggregated.REVENUE_RATE].from_model == \
        set([ModelOutput.ELECTRICAL_POWER])
    assert simple_product_aggregation.aggregate_mappings[Aggregated.REVENUE_RATE].from_ambient == \
        set([Ambient.ELECTRICITY_PRICE])
    assert simple_product_aggregation.aggregate_mappings[Aggregated.REVENUE_RATE].from_control == \
        set([])
    assert simple_product_aggregation.aggregate_mappings[Aggregated.DAMAGE_RATE].from_model == \
        set([ModelOutput.DAMAGE_RATE])
    assert simple_product_aggregation.aggregate_mappings[Aggregated.DAMAGE_RATE].from_ambient == \
        set([])
    assert simple_product_aggregation.aggregate_mappings[Aggregated.DAMAGE_RATE].from_control == \
        set([])
    
    # Output aggregation
    aggregated_output = simple_product_aggregation.compute_aggregate(
        model_output={ModelOutput.ELECTRICAL_POWER: 5,
                      ModelOutput.DAMAGE_RATE: 4},
        ambient_condition={Ambient.ELECTRICITY_PRICE: 3},
        control_setpoints={})
    assert aggregated_output == pytest.approx({Aggregated.REVENUE_RATE: 15,
                                               Aggregated.DAMAGE_RATE: 4})
        
    

    