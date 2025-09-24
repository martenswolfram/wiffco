from typing import Dict, Any, List
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.interface import (
    AmbientVariable,
    OutputVariable,
    InterfaceVariables,
    ComponentParams,
    ComponentType)
    
class Statistics(ABC):
    def __init__(self,
                 params: ComponentParams):
        self.interface = params.interface()

    @abstractmethod
    def systematic_sample(self, N: int = None):
        pass

    @abstractmethod
    def expected_value(self):
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

class StatisticsType(Enum):
    DISCRETE_STATISTICS = "discrete_statistics"

class DiscreteStatisticsParams(ComponentParams):
    def __init__(self,
                 component_name: str,
                 support_variables: List[AmbientVariable | OutputVariable],
                 prevalence: np.ndarray,
                 support_points: np.ndarray):
        super().__init__(component_type=ComponentType.STATISTICS,
                         component_name=component_name)
        self.support_variables = support_variables
        self.prevalence = prevalence
        self.support_points = support_points
    
    def _interface(self):
        inputs=InterfaceVariables()
        if all(isinstance(var, AmbientVariable) for var in self.support_variables):
            outputs=InterfaceVariables(ambient_variables=set(self.support_variables))
        elif all(isinstance(var, OutputVariable) for var in self.support_variables):
            outputs=InterfaceVariables(output_variables=set(self.support_variables))
        else:
            raise ValueError("DiscreteStatisticsParams: Invalid support variables specified.")
        return inputs, outputs
        
def discrete_statistics_params_from_dict(name: str,
                                         param_dict: Dict[str, Any]):
    if "ambient_variables" in param_dict.keys():
        support_variables = \
            [AmbientVariable(support_var) for support_var in param_dict["ambient_variables"]]
    elif "output_variables" in param_dict.keys():
        support_variables = \
            [OutputVariable(support_var) for support_var in param_dict["output_variables"]]
    prevalence = np.array(param_dict["prevalence"])
    support_points = np.array(param_dict["support_points"])
    if support_points.shape[0] != len(support_variables):
        raise ValueError("DiscreteAmbientStatisticsParams: Support data and ambient variables dimensions mismatch.")
    if len(prevalence.shape) != 1 or prevalence.shape[0] != support_points.shape[1]:
        raise ValueError("DiscreteAmbientStatisticsParams: Prevalence and support data dimensions mismatch.")
    prevalence = prevalence / np.sum(prevalence)
    
    return DiscreteStatisticsParams(component_name=name,
                                    support_variables=support_variables,
                                    prevalence=prevalence,
                                    support_points=support_points)

class DiscreteStatistics(Statistics):
    def __init__(self,
                 params: DiscreteStatisticsParams):
        super().__init__(params=params)
        
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
        
    def expected_value(self):
        expectation = self.ordered_support_points @ self.ordered_prevalence.T
        return {supp_var: val for supp_var, val in zip(self.support_variables, expectation)}
            