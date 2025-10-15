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
                 support_points: Dict[StatisticalVar, np.ndarray],
                 normalized_weights: np.ndarray,
                 probability_covered: float = 1):
        self.support_points = support_points
        self.normalized_weights = normalized_weights
        self.probability_covered = probability_covered
        self.N = len(self.normalized_weights)

    def variables_iter(self):
        keys = list(self.support_points.keys())
        for i in range(self.N):
            yield {key: self.support_points[key][i] for key in keys}
        
    def weighted_variables_iter(self):
        keys = list(self.support_points.keys())
        for i in range(self.N):
            yield self.normalized_weights[i], {key: self.support_points[key][i] for key in keys}
        
    def discrete_statistics(self):
        discrete_statistics_params =  DiscreteStatisticsParams(
            support_points=self.support_points,
            prevalence=self.normalized_weights)
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
                 support_points: Dict[StatisticalVar, np.ndarray],
                 prevalence: np.ndarray):
        self.support_points = support_points
        self.prevalence = prevalence

    def input_variables(self):
        return set()

    def output_variables(self):
        return {out_var: supp.shape[1] for \
                out_var, supp in self.support_points.items()}
                
def discrete_statistics_params_from_dict(param_dict: Dict[str, Any]):
    
    if param_dict["data_type"] == "ambient":
        data_type = Ambient
    elif param_dict["data_type"] == "aggregate":
        data_type = Aggregated
    support_points = {data_type(in_var): np.array(supp) for \
                      in_var, supp in param_dict["support_points"].items()}
    prevalence = np.array(param_dict["prevalence"])
    prevalence = prevalence / np.sum(prevalence)    
    return DiscreteStatisticsParams(support_points=support_points,
                                    prevalence=prevalence)

class DiscreteStatistics(Statistics):
    def __init__(self,
                 statistics_name: str,
                 statistics_params: DiscreteStatisticsParams):
        super().__init__(statistics_name=statistics_name,
                         statistics_params=statistics_params)
        
        # Ordered by prevalence
        prevalence_index = np.argsort(statistics_params.prevalence)[::-1]
        self.ordered_support_points = {var: supp[prevalence_index, :] for \
                                       var, supp in statistics_params.support_points.items()}
        self.ordered_prevalence = statistics_params.prevalence[prevalence_index]

    def systematic_sample(self, N_max: int = None):
        if N_max is None or N_max >= len(self.ordered_prevalence):
            return SystematicSample(support_points=self.ordered_support_points,
                                    normalized_weights=self.ordered_prevalence)    
        elif 1 <= N_max < len(self.ordered_prevalence):
            weights = self.ordered_prevalence[:N_max]
            probability_covered = np.sum(weights)
            support_points = {var: supp[:N_max, :] for \
                              var, supp in self.ordered_support_points.items()}
            return SystematicSample(support_points=support_points,
                                    normalized_weights=(weights / probability_covered),
                                    probability_covered=probability_covered)
        else:
            raise ValueError("DiscreteStatistics: Invalid number of samples N.")
        
    def expected_value(self):
        return {var: self.ordered_prevalence @ supp for \
                var, supp in self.ordered_support_points.items()}
