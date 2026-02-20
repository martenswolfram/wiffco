import pathlib
import numpy as np
from wiffco.config import parse_json_file
from wiffco.multi_metrics_reduction import multi_metrics_reduction_from_dict
from wiffco.interface import AccumulatedMetric, DataTable

    
def test_metrics_reduction():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "scalar_weighting.jsonc"
    param_dict = parse_json_file(path=json_path)
    scalar_metrics_weighting = multi_metrics_reduction_from_dict(
        param_dict=param_dict)
        
    acc_metrics = DataTable({AccumulatedMetric.REVENUE_EUR: np.array([17,
                                                                  13]),
                             AccumulatedMetric.ACCRUED_DAMAGE: np.array([23,
                                                                         24])})
    # Initialization
    assert np.array_equal(scalar_metrics_weighting.evaluate(acc_metrics=acc_metrics), np.array([17, 13]))
    

    