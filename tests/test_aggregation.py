import pathlib
import numpy as np
from twain_wifco.config import aggregation_from_json
from twain_wifco.interface import (
    Ambient,
    ModelOutput,
    Aggregated,
    DataTable)

    
def test_simple_product_aggregation():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "aggregation_revenue_damage.jsonc"
    simple_product_aggregation = aggregation_from_json(json_path=json_path)
    
    # Output aggregation
    aggregated_output = simple_product_aggregation.compute_aggregate(
        model_output=DataTable({ModelOutput.ELECTRICAL_POWER: np.array(5),
                                ModelOutput.DAMAGE_RATE: np.array(4)}),
        ambient_condition=DataTable({Ambient.ELECTRICITY_PRICE: np.array(3)}),
        control_setpoints=DataTable({}))
    expected_output = DataTable({Aggregated.REVENUE_RATE: np.array(15),
                                               Aggregated.DAMAGE_RATE: np.array(4)})
    assert aggregated_output == expected_output