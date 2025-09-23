import pathlib
import pytest
import numpy as np
from twain_wifco.config import parse_json_file, ambient_statistics_from_dict
from twain_wifco.interface import AmbientVariable

def test_simple_ambient_statistics():
    json_path = pathlib.Path(__file__).parent / "data" / "simple_ambient_statistics.json"
    param_dict = parse_json_file(path=json_path)
    simple_ambient_statistics = ambient_statistics_from_dict(param_dict=param_dict)
    
    ordered_prevalence = np.array([0.4, 0.25, 0.19, 0.1, 0.05, 0.01])
    ordered_support_points = np.array([[   20,   10,  30,  20,   5,  10],
                                       [  180,  240, 120,  60, 300,   0],
                                       [    5,    5,   2,   5,  10,  10]])
    # Initialization
    assert simple_ambient_statistics.name == "simple_ambient_statistics"
    assert simple_ambient_statistics.support_variables == \
        [AmbientVariable.WIND_SPEED, AmbientVariable.WIND_DIRECTION, AmbientVariable.ELECTRICITY_PRICE]
    assert simple_ambient_statistics.ordered_prevalence == pytest.approx(ordered_prevalence)
    assert simple_ambient_statistics.ordered_support_points == pytest.approx(ordered_support_points)
    
    # Sample without N specifed
    sys_sample_default = simple_ambient_statistics.systematic_sample()
    assert sys_sample_default.support_variables == simple_ambient_statistics.support_variables
    assert sys_sample_default.normalized_weights == pytest.approx(ordered_prevalence)
    assert sys_sample_default.values == pytest.approx(ordered_support_points)
    assert sys_sample_default.probability_covered == pytest.approx(1)

    # Invalid N
    with pytest.raises(ValueError) as excinfo: 
        simple_ambient_statistics.systematic_sample(N=0)
    assert "Invalid number of samples" in str(excinfo.value)
    
    # Sample with larger N specifed
    N = 100
    sys_sample_larger = simple_ambient_statistics.systematic_sample(N=N)
    assert sys_sample_larger.support_variables == simple_ambient_statistics.support_variables
    assert sys_sample_larger.normalized_weights == pytest.approx(ordered_prevalence)
    assert sys_sample_larger.values == pytest.approx(ordered_support_points)
    assert sys_sample_larger.probability_covered == pytest.approx(1)

    # Sample with smaller N specifed
    N = 4
    sys_sample_smaller = simple_ambient_statistics.systematic_sample(N=N)
    probability_covered = np.sum(ordered_prevalence[:N])
    assert sys_sample_smaller.support_variables == simple_ambient_statistics.support_variables
    assert sys_sample_smaller.normalized_weights == pytest.approx(ordered_prevalence[:N] / probability_covered)
    assert sys_sample_smaller.values == pytest.approx(ordered_support_points[:, :N])
    assert sys_sample_smaller.probability_covered == pytest.approx(probability_covered)
    

