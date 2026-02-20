from typing import Dict, Any
from matplotlib import pyplot as plt
from abc import abstractmethod
from enum import Enum
import numpy as np
from wiffco.interface import (
    Component,
    Ambient,
    DataTable,
    Interface,
    DataType
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

    @abstractmethod
    def map_to_discrete(self,
                        support: DataTable[Ambient]
                        ) -> "DiscreteAmbientStatistics":
        """Create a DiscreteAmbientStatistics object based on support
        
        Args:
            support (DataTable[Ambient]): discrete support points

        Returns:
            DiscreteAmbientStatistics: The created discrete statistics
        """
        


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
    
    def map_to_discrete(self, support: DataTable[Ambient]) -> "DiscreteAmbientStatistics":
        nearest_support_indices = support.find_nearest_points(self._ordered_ambient_support)
        probabilities = np.bincount(nearest_support_indices,
                                    weights=self._ordered_probabilities,
                                    minlength=len(support))
        return DiscreteAmbientStatistics(name=self.component_name + "_mapped",
                                         ambient_support=support,
                                         probabilities=probabilities)

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
                          support_names: Dict[str, Ambient],
                          prevalence_name: str,
                          statistics_name: str = "statistics_from_table",
                          fill_variables: Dict[Ambient, np.ndarray] = None):

    table_data = {}
    prevalence = None
    for h_str, data in data_dict.items():
        if h_str in support_names:
            key = support_names[h_str]
            table_data[key] = data
        elif h_str == prevalence_name:
            prevalence = data
    order = list(table_data.keys())
    size = table_data[order[0]].shape[0]
    if fill_variables is not None:
        for fill_var, value in fill_variables.items():
            order.append(fill_var)
            table_data[fill_var] = np.repeat(np.array(value)[np.newaxis, ...],
                                             repeats=size,
                                             axis=0)
    
    if prevalence is not None and len(table_data):
        probabilities = prevalence / np.sum(prevalence)
        support_data = DataTable(table_data, order)
        return DiscreteAmbientStatistics(
            name=statistics_name,
            ambient_support=support_data,
            probabilities=probabilities)
    else:
        raise ValueError("Could not parse table data to statistics")

def plot_statistics_2d(discrete_statistics: DiscreteAmbientStatistics):
    if len(discrete_statistics._ordered_ambient_support.order) != 2:
        raise ValueError("Expected 2D-data table.")
    support = discrete_statistics._ordered_ambient_support
    var1 = support.order[0]
    var2 = support.order[1]

    plt.scatter(support[var1], support[var2], marker='o',
                c=discrete_statistics._ordered_probabilities)
    plt.xlabel(var1.value)
    plt.ylabel(var2.value)
    plt.show()