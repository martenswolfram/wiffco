from typing import Dict, Any, List
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.interface import AmbientVariable, OutputVariable

class StatisticsType(Enum):
    DISCRETE_STATISTICS = "discrete_statistics"
    
class Statistics(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def systematic_sample(self, N: int = None):
        pass
    
class SystematicSample:
    def __init__(self,
                 support_variables: List[AmbientVariable | OutputVariable], 
                 support_values: np.ndarray,
                 normalized_weights: np.ndarray,
                 probability_covered: float = 1):
        self.support_variables = support_variables
        self.support_values = support_values
        self.normalized_weights = normalized_weights
        self.probability_covered = probability_covered
        self.N = len(self.normalized_weights)

class DiscreteStatisticsParams:
    def __init__(self,
                 support_variables: List[AmbientVariable | OutputVariable],
                 prevalence: np.ndarray,
                 support_points: np.ndarray):
        self.support_variables = support_variables
        self.prevalence = prevalence
        self.support_points = support_points
        if self.support_points.shape[0] != len(self.support_variables):
            raise ValueError("DiscreteAmbientStatisticsParams: Support data and ambient variables dimensions mismatch.")
        if len(self.prevalence.shape) != 1 or self.prevalence.shape[0] != self.support_points.shape[1]:
            raise ValueError("DiscreteAmbientStatisticsParams: Prevalence and support data dimensions mismatch.")
        self.prevalence = self.prevalence / np.sum(self.prevalence)

def discrete_statistics_params_from_dict(param_dict: Dict[str, Any]):
    if "ambient_variables" in param_dict.keys():
        support_variables = \
            [AmbientVariable(support_var) for support_var in param_dict["ambient_variables"]]
    elif "output_variables" in param_dict.keys():
        support_variables = \
            [OutputVariable(support_var) for support_var in param_dict["output_variables"]]
    prevalence = np.array(param_dict["prevalence"])
    support_points = np.array(param_dict["support_points"])
    return DiscreteStatisticsParams(support_variables=support_variables,
                                    prevalence=prevalence,
                                    support_points=support_points)

class DiscreteStatistics(Statistics):
    def __init__(self,
                 name: str,
                 params: DiscreteStatisticsParams):
        super().__init__(name=name)
        
        # Ordered by prevalence
        prevalence_index = np.argsort(params.prevalence)[::-1]
        self.support_variables = params.support_variables
        self.ordered_support_points = params.support_points[:, prevalence_index]
        self.ordered_prevalence = params.prevalence[prevalence_index]

    def systematic_sample(self, N: int = None):
        if N is None or N >= len(self.ordered_prevalence):
            return SystematicSample(support_variables=self.support_variables,
                                    support_values=self.ordered_support_points,
                                    normalized_weights=self.ordered_prevalence)    
        elif 1 <= N < len(self.ordered_prevalence):
            values = self.ordered_support_points[:, :N]
            weights = self.ordered_prevalence[:N]
            probability_covered = np.sum(weights)
            return SystematicSample(support_variables=self.support_variables,
                                    support_values=values,
                                    normalized_weights=(weights / probability_covered),
                                    probability_covered=probability_covered)
        else:
            raise ValueError("DiscreteStatistics: Invalid number of samples N.")
            