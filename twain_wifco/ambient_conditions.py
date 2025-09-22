from typing import Dict, Any
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.interface import AmbientVariable

class AmbientStatisticsType(Enum):
    DISCRETE_ABIENT_STATISTICS = "discrete_ambient_statistics"
    
class AmbientStatistics(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def systematic_sample(self, N: int = None):
        pass
    
    @abstractmethod
    def random_sample(self, N: int = None):
        pass

class SystematicSample:
    def __init__(self,
                 values: np.ndarray,
                 normalized_weights: np.ndarray,
                 probability_covered: float = 1):
        self.values = values
        self.normalized_weights = normalized_weights
        self.probability_covered = probability_covered

class DiscreteAmbientStatisticsParams:
    def __init__(self, param_dict: Dict[str, Any]):
        self.ambient_variables = \
            [AmbientVariable(ambient_var) for ambient_var in param_dict["ambient_variables"]]
        self.prevalence = np.array(param_dict["prevalence"])
        self.support_points = np.array(param_dict["support_points"])
        if self.support_points.shape[0] != len(self.ambient_variables):
            raise ValueError("DiscreteAmbientStatisticsParams: Support data and ambient variables dimensions mismatch.")
        if len(self.prevalence.shape) != 1 or self.prevalence.shape[0] != self.support_points.shape[1]:
            raise ValueError("DiscreteAmbientStatisticsParams: Prevalence and support data dimensions mismatch.")
        self.prevalence = self.prevalence / np.sum(self.prevalence)

class DiscreteAmbientStatistics(AmbientStatistics):
    def __init__(self,
                 name: str,
                 params: DiscreteAmbientStatisticsParams):
        super().__init__(name=name)
        
        # Ordered by prevalence
        prevalence_index = np.argsort(params.prevalence)[::-1]
        self.ambient_variables = params.ambient_variables
        self.ordered_support_points = params.support_points[:, prevalence_index]
        self.ordered_prevalence = params.prevalence[prevalence_index]

    def systematic_sample(self, N: int = None):
        if N is None or N >= len(self.ordered_prevalence):
            return SystematicSample(values=self.ordered_support_points,
                                    normalized_weights=self.ordered_prevalence)    
        elif 1 <= N < len(self.ordered_prevalence):
            values = self.ordered_support_points[:, :N]
            weights = self.ordered_prevalence[:N]
            probability_covered = np.sum(weights)
            return SystematicSample(values=values,
                                    normalized_weights=(weights / probability_covered),
                                    probability_covered=probability_covered)
        else:
            raise ValueError("DiscreteAmbientStatistics: Invalid number of samples N.")
            
    def random_sample(self, N: int = None):
        raise NotImplementedError("DiscreteAmbientStatistics: random_sample not implemented.")
