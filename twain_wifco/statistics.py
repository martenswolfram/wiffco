from typing import Dict, Any, Type
from abc import abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.interface import (
    Component,
    ComponentParams,    
    StatisticalType,
    Ambient,
    Aggregated,
    DataTable,
    DataPoint,
    Interface,
    retrieve_single_key_str)

class SystematicSample:
    def __init__(self,
                 support_data: DataTable[StatisticalType],
                 normalized_weights: np.ndarray,
                 probability_covered: float = 1):
        self.support_data = support_data
        self.normalized_weights = normalized_weights
        self.probability_covered = probability_covered
        self.N = len(self.normalized_weights)

    def variables_iter(self):
        for i in range(self.N):
            yield self.support_data.get_point(i)
        
    def weighted_variables_iter(self):
        for i in range(self.N):
            yield self.normalized_weights[i], self.support_data.get_point(i)
        
    def discrete_statistics(self):
        discrete_statistics_params =  DiscreteStatisticsParams(
            support_data=self.support_data,
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
                 statistical_type: Type[StatisticalType],
                 support_data: DataTable[StatisticalType],
                 prevalence: np.ndarray):
        self.statistical_type = statistical_type
        self.support_data = support_data
        self.prevalence = prevalence

    def input_interface(self) -> Interface:
        return Interface()

    def output_interface(self):
        if self.statistical_type == Ambient:
            return Interface(ambient_shapes=self.support_data.shapes())
        else:
            return Interface(aggregated_shapes=self.support_data.shapes())
        
def discrete_statistics_params_from_dict(param_dict: Dict[str, Any]):
    
    if param_dict["data_type"] == "ambient":
        data_type = Ambient
    elif param_dict["data_type"] == "aggregate":
        data_type = Aggregated
    support_data = DataTable({data_type(in_var): np.array(supp) for \
                              in_var, supp in param_dict["support_data"].items()})
    prevalence = np.array(param_dict["prevalence"])
    prevalence = prevalence / np.sum(prevalence)    
    return DiscreteStatisticsParams(statistical_type=data_type,
                                    support_data=support_data,
                                    prevalence=prevalence)

class DiscreteStatistics(Statistics):
    def __init__(self,
                 statistics_name: str,
                 statistics_params: DiscreteStatisticsParams):
        super().__init__(statistics_name=statistics_name,
                         statistics_params=statistics_params)
        
        # Ordered by prevalence
        prevalence_index = np.argsort(statistics_params.prevalence)[::-1]
        self.ordered_support_data = statistics_params.support_data
        for var in self.ordered_support_data.keys():
            self.ordered_support_data.data[var] = \
                self.ordered_support_data.data[var][prevalence_index, :]
        self.ordered_prevalence = statistics_params.prevalence[prevalence_index]

    def systematic_sample(self, N_max: int = None):
        if N_max is None or N_max >= len(self.ordered_prevalence):
            return SystematicSample(support_data=self.ordered_support_data,
                                    normalized_weights=self.ordered_prevalence)    
        elif 1 <= N_max < len(self.ordered_prevalence):
            weights = self.ordered_prevalence[:N_max]
            probability_covered = np.sum(weights)
            support_data = {var: supp[:N_max, :] for \
                              var, supp in self.ordered_support_data.data.items()}
            return SystematicSample(support_data=DataPoint(support_data),
                                    normalized_weights=(weights / probability_covered),
                                    probability_covered=probability_covered)
        else:
            raise ValueError("DiscreteStatistics: Invalid number of samples N.")
        
    def expected_value(self):
        return DataPoint({var: self.ordered_prevalence @ supp for \
                          var, supp in self.ordered_support_data.data.items()})
