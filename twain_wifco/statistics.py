from typing import Dict, Any, Generic
from abc import abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.interface import (
    Component,
    Ambient,
    DataTable,
    Interface,
    DataType,
    MAP_STR_TO_ENUM
)


class SystematicAmbientSample:
    """A systematic sample of ambient data points with weights.

    Attributes:
        support_data (DataTable[Ambient]): Support data for sampling.
        normalized_weights (np.ndarray): Normalized weights of each data point.
        probability_covered (float): Total probability covered by the sample.
        N (int): Number of samples.
    """
    def __init__(self,
                 ambient_support: DataTable[Ambient],
                 normalized_weights: np.ndarray,
                 probability_covered: float = 1.0):
        self.ambient_support = ambient_support
        self.normalized_weights = normalized_weights
        self.probability_covered = probability_covered
        self.N = len(self.normalized_weights)
        
class AmbientStatistics(Component):
    """Abstract base class for ambient statistics computations.

    """
    @abstractmethod
    def systematic_sample(self,
                          N_max: int = None,
                          min_prob: float = None) -> SystematicAmbientSample:
        """Return a systematic ambient sample.

        Args:
            N_max (int, optional): Maximum number of points to include.
            min_prob (float, optional): Minimum probability to be covered.

        Returns:
            SystematicAmbientSample: The generated sample.
        """
        pass

class AmbientStatisticsType(Enum):
    DISCRETE_STATISTICS = "discrete_statistics"

class DiscreteAmbientStatistics(AmbientStatistics):
    """Concrete implementation of Statistics for discrete distributions."""

    def __init__(self,
                 name: str,
                 ambient_support: DataTable[Ambient],
                 probabilities: np.ndarray):
        
        self.component_name = name
    
        # Discard zero-probability points
        probabilities_reduced = probabilities[probabilities > 0]
        ambient_support_reduced = {key: key_data[probabilities > 0, ...] for \
                                   key, key_data in ambient_support.data.items()}
        
        #  Order support data by prevalence descending
        sorted_index = np.argsort(probabilities_reduced)[::-1]
        ordered_support = {}
        for var, data in ambient_support_reduced.items():
            ordered_support[var] = data[sorted_index, ...]
        self._ordered_probabilities = probabilities_reduced[sorted_index]
        self._ordered_ambient_support = DataTable(data=ordered_support,
                                                  order=ambient_support.order)

        self.input_interface = Interface()
        self.output_interface = Interface(
            all_shapes={
                self._ordered_ambient_support.data_type: self._ordered_ambient_support.shapes()})
    
    def systematic_sample(self,
                          N_max: int = None,
                          min_prob: float = None) -> SystematicAmbientSample:
        """Return a systematic ambient sample.

        Args:
            N_max (int, optional): Maximum number of samples. If None, include all points.
            min_prob (float, optional): Minimum probability to be covered. If both are specified, N_max has priority
            
        Returns:
            SystematicAmbientSample: Sampled ambient points with normalized weights.

        Raises:
            ValueError: If N_max < 1.
        """
        if N_max is None:
            if min_prob is not None:
                if not 0 < min_prob:
                    raise ValueError("min_prob must be a positive number.")
                cumsum = np.cumsum(self._ordered_probabilities)
                if min_prob < 1:
                    N_max = np.searchsorted(cumsum, min_prob) + 1
        elif N_max <= 0:
            raise ValueError("N_max must be a positive integer.")

        weights = self._ordered_probabilities[:N_max]
        probability_covered = np.sum(weights)
        support_data_subset = {var: supp[:N_max, ...] for \
                               var, supp in self._ordered_ambient_support.data.items()}
        return SystematicAmbientSample(
            ambient_support=DataTable(data=support_data_subset,
                                   order=self._ordered_ambient_support.order),
            normalized_weights=weights / probability_covered,
            probability_covered=probability_covered
        )

def statistics_from_dict(param_dict: Dict[str, Any]) -> AmbientStatistics:
    """Construct DiscreteStatistics from a dictionary.

    Args:
        param_dict (Dict[str, Any]): Dictionary containing 'data_type', 'support_data', and 'prevalence'.

    Returns:
        DiscreteStatistics: Constructed DIscreteStatistics object.
    """
    name = param_dict["name"]
    statistics_type = AmbientStatisticsType(param_dict["statistics_type"])
    
    if statistics_type == AmbientStatisticsType.DISCRETE_STATISTICS:
        ambient_support = DataTable({Ambient(var): np.array(supp) for \
                                     var, supp in param_dict["ambient_support"].items()})
        prevalence = np.array(param_dict["prevalence"])
        probabilities = prevalence / np.sum(prevalence)
            
        return DiscreteAmbientStatistics(
            name=name,
            ambient_support=ambient_support,
            probabilities=probabilities
        )
    else:
        raise NotImplementedError("Only discrete ambient statistics implemented.")

def statistics_from_table(data_dict: Dict[str, np.ndarray],
                          support_names: Dict[str, DataType],
                          prevalence_name: str,
                          statistics_name: str = "statistics_from_table"):

    table_data = {}
    prevalence = None
    for h_str, data in data_dict.items():
        if h_str in support_names:
            key = support_names[h_str]
            table_data[key] = data
        elif h_str == prevalence_name:
            prevalence = data
    order = list(table_data.keys())
    
    if prevalence is not None and len(table_data):
        probabilities = prevalence / np.sum(prevalence)
        support_data = DataTable(table_data, order)
        return DiscreteAmbientStatistics(
            name=statistics_name,
            ambient_support=support_data,
            probabilities=probabilities)
    else:
        raise ValueError("Could not parse table data to statistics")
