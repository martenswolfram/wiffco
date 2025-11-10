import pathlib
import numpy as np
from twain_wifco.config import parse_json_file
from twain_wifco.aggregation import aggregation_from_dict
from twain_wifco.interface import (
    Ambient,
    ModelOutput,
    Aggregate,
    DataTable)

    
def test_simple_product_aggregation():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "aggregation_symbolic.jsonc"
    param_dict = parse_json_file(path=json_path)
    simple_product_aggregation = aggregation_from_dict(param_dict=param_dict)
    
    # Output aggregation
    aggregated_output = simple_product_aggregation.compute_aggregate(
        model_output=DataTable({ModelOutput.ELECTRICAL_POWER: np.array([[5, 3],
                                                                        [2, 3]]),
                                ModelOutput.DAMAGE_RATE: np.array([[4, 2],
                                                                   [7, 6]])}),
        ambient=DataTable({Ambient.ELECTRICITY_PRICE: np.array([3, 1])}))
    expected_output = DataTable({Aggregate.REVENUE_RATE: np.array([(5 + 3) * 3,
                                                                   (2 + 3) * 1]),
                                 Aggregate.DAMAGE_RATE: np.array([[4, 2],
                                                                  [7, 6]])})
    assert aggregated_output == expected_output