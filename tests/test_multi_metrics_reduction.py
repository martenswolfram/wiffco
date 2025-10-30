import pathlib
import numpy as np
from twain_wifco.config import multi_metrics_reduction_from_json
from twain_wifco.interface import AccumulatedMetric, DataPoint

    
def test_linear_constraint():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "metrics_reduction_scalar_weighting.jsonc"
    scalar_metrics_weighting = multi_metrics_reduction_from_json(json_path=json_path)
        
    acc_metrics = DataPoint({AccumulatedMetric.REVENUE: np.array(17),
                             AccumulatedMetric.ACCRUED_DAMAGE: np.array(23)})
    # Initialization
    assert scalar_metrics_weighting.evaluate(acc_metrics=acc_metrics) == 17
    pass
    

    