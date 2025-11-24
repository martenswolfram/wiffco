import pathlib
import pytest
import numpy as np
from twain_wifco.config import parse_json_file
from twain_wifco.statistics import statistics_from_dict
from twain_wifco.interface import Ambient, DataTable

def test_statistics():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "statistics_discrete_ambient.jsonc"
    param_dict = parse_json_file(path=json_path)
    discrete_ambient_statistics = statistics_from_dict(param_dict=param_dict)
    
    ordered_prevalence = np.array([0.4, 0.25, 0.19, 0.1, 0.05, 0.01])
    ordered_support_data = DataTable({
            Ambient.WIND_SPEED_MPS: np.array([ 20, 
                                           10, 
                                           30, 
                                           20, 
                                            5, 
                                           10]),
            Ambient.WIND_DIRECTION_DEG: np.array([180, 
                                              240, 
                                              120, 
                                               60, 
                                              300, 
                                                0]),
            Ambient.ELECTRICITY_PRICE_EPKWH: np.array([ 5,
                                                  5,
                                                  2,
                                                  5,
                                                  7,
                                                  7])})
    

    # Sample without N specifed
    sys_sample_default = discrete_ambient_statistics.systematic_sample()
    assert sys_sample_default.normalized_weights == \
        pytest.approx(ordered_prevalence)
    assert ordered_support_data == \
        sys_sample_default.ambient_support
    assert sys_sample_default.probability_covered == \
        pytest.approx(1)

    # Invalid N
    with pytest.raises(ValueError) as excinfo: 
        discrete_ambient_statistics.systematic_sample(N_max=0)
    assert "N_max must be a positive integer." in str(excinfo.value)
    
    # Sample with larger N specifed
    N = 100
    sys_sample_larger = discrete_ambient_statistics.systematic_sample(N_max=N)
    assert sys_sample_larger.normalized_weights == \
        pytest.approx(ordered_prevalence)
    assert ordered_support_data == sys_sample_larger.ambient_support
    assert sys_sample_larger.probability_covered == \
        pytest.approx(1)

    # Sample with smaller N specifed
    N = 4
    sys_sample_smaller = discrete_ambient_statistics.systematic_sample(N_max=N)
    probability_covered = np.sum(ordered_prevalence[:N])
    assert sys_sample_smaller.normalized_weights == pytest.approx(ordered_prevalence[:N] / probability_covered)
    small_support_data = DataTable({var: supp[:N, ...] for \
                                    var, supp in ordered_support_data.data.items()})
    assert small_support_data == sys_sample_smaller.ambient_support
    assert small_support_data
    assert sys_sample_smaller.probability_covered == pytest.approx(probability_covered)

    # Sample with minimum probability
    for min_prob in [0.2, 0.4, 0.6, 0.8]:
        sys_sample_min_prob = discrete_ambient_statistics.systematic_sample(min_prob=min_prob)
        assert sys_sample_min_prob.probability_covered >= min_prob
    
    # map to different support
    reduced_support = DataTable({
            Ambient.WIND_SPEED_MPS: np.array([10,
                                              20,
                                               5,
                                              10]),
            Ambient.WIND_DIRECTION_DEG: np.array([240,
                                                   60,
                                                  300,
                                                    0]),
            Ambient.ELECTRICITY_PRICE_EPKWH: np.array([2,
                                                       5,
                                                       7,
                                                       7])})
    new_discrete_ambient_statistics = discrete_ambient_statistics.map_to_discrete(
        support=reduced_support
    )
    assert new_discrete_ambient_statistics._ordered_ambient_support == reduced_support
    assert len(new_discrete_ambient_statistics._ordered_probabilities) == len(reduced_support)
    assert np.isclose(np.sum(new_discrete_ambient_statistics._ordered_probabilities), 1)
