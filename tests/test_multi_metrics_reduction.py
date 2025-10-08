import pathlib
import pytest
from twain_wifco.config import multi_metrics_reduction_from_json
from twain_wifco.interface import AccumulatedMetric

    
def test_linear_constraint():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "scalar_metrics_weighting.json"
    scalar_metrics_weighting = multi_metrics_reduction_from_json(json_path=json_path)
        
    # Initialization
    assert scalar_metrics_weighting.component_name == "scalar_metrics_weighting"
    assert scalar_metrics_weighting.metric_weights.keys() == set([AccumulatedMetric.REVENUE])
    assert scalar_metrics_weighting.metric_weights[AccumulatedMetric.REVENUE] == pytest.approx(1)
    assert scalar_metrics_weighting.maximize is True
    

    