from typing import Dict, Any, List
from abc import abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.interface import (
    Component,
    ComponentParams,    
    StatisticalVar,
    Ambient,
    Aggregated,
    retrieve_single_key_str)

class SystematicSample:
    def __init__(self,
                 support_variables: List[StatisticalVar], 
                 support_values: np.ndarray,
                 normalized_weights: np.ndarray,
                 probability_covered: float = 1):
        self.support_variables = support_variables
        self.support_values = support_values
        self.normalized_weights = normalized_weights
        self.probability_covered = probability_covered
        self.N = len(self.normalized_weights)

    def variables_iter(self):
        for values in self.support_values:
            sample_dict = {var: values[m] for m, var in enumerate(self.support_variables)}
            yield sample_dict

    def weighted_variables_iter(self):
        for weight, values in zip(self.normalized_weights, self.support_values):
            sample_dict = {var: values[m] for m, var in enumerate(self.support_variables)}
            yield weight, sample_dict

    def discrete_statistics(self):
        discrete_statistics_params =  DiscreteStatisticsParams(
            support_variables=self.support_variables,
            prevalence=self.normalized_weights,
            support_points=self.support_values)
        return DiscreteStatistics(statistics_name="discrete_stats_from_sample",
                                  statistics_params=discrete_statistics_params)

class Statistics(Component):
    def __init__(self,
                 statistics_name: str,
                 statistics_params: ComponentParams):
        super().__init__(component_name=statistics_name,
                         component_params=statistics_params)

    @abstractmethod
    def systematic_sample(self, N_max: int = None) -> SystematicSample:
        pass

    @abstractmethod
    def expected_value(self):
        pass


    
class StatisticsType(Enum):
    DISCRETE_STATISTICS = "discrete_statistics"

class DiscreteStatisticsParams(ComponentParams):
    def __init__(self,
                 support_variables: List[StatisticalVar],
                 prevalence: np.ndarray,
                 support_points: np.ndarray):
        self.support_variables = support_variables
        self.prevalence = prevalence
        self.support_points = support_points

    def input_variables(self):
        return set()

    def output_variables(self):
        return set(self.support_variables)
                
def discrete_statistics_params_from_dict(param_dict: Dict[str, Any]):
    
    key_str = retrieve_single_key_str(param_dict, set(["aggregated_variables", "ambient_variables"]))    
    if key_str == "ambient_variables":
        support_variables = \
            [Ambient(support_var) for support_var in param_dict["ambient_variables"]]
    else:
        support_variables = \
            [Aggregated(support_var) for support_var in param_dict["aggregated_variables"]]
    prevalence = np.array(param_dict["prevalence"])
    support_points = np.array(param_dict["support_points"])
    if support_points.shape[1] != len(support_variables):
        raise ValueError("DiscreteStatisticsParams: Support data and support variables dimensions mismatch.")
    if len(prevalence.shape) != 1 or prevalence.shape[0] != support_points.shape[0]:
        raise ValueError("DiscreteStatisticsParams: Prevalence and support data dimensions mismatch.")
    prevalence = prevalence / np.sum(prevalence)
    
    return DiscreteStatisticsParams(support_variables=support_variables,
                                    prevalence=prevalence,
                                    support_points=support_points)

class DiscreteStatistics(Statistics):
    def __init__(self,
                 statistics_name: str,
                 statistics_params: DiscreteStatisticsParams):
        super().__init__(statistics_name=statistics_name,
                         statistics_params=statistics_params)
        
        # Ordered by prevalence
        prevalence_index = np.argsort(statistics_params.prevalence)[::-1]
        self.support_variables = statistics_params.support_variables
        self.ordered_support_points = statistics_params.support_points[prevalence_index, :]
        self.ordered_prevalence = statistics_params.prevalence[prevalence_index]

    def systematic_sample(self, N_max: int = None):
        if N_max is None or N_max >= len(self.ordered_prevalence):
            return SystematicSample(support_variables=self.support_variables,
                                    support_values=self.ordered_support_points,
                                    normalized_weights=self.ordered_prevalence)    
        elif 1 <= N_max < len(self.ordered_prevalence):
            values = self.ordered_support_points[:N_max, :]
            weights = self.ordered_prevalence[:N_max]
            probability_covered = np.sum(weights)
            return SystematicSample(support_variables=self.support_variables,
                                    support_values=values,
                                    normalized_weights=(weights / probability_covered),
                                    probability_covered=probability_covered)
        else:
            raise ValueError("DiscreteStatistics: Invalid number of samples N.")
        
    def expected_value(self):
        expectation = self.ordered_prevalence @ self.ordered_support_points
        return {supp_var: val for supp_var, val in zip(self.support_variables, expectation)}
