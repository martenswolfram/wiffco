from typing import Dict, Any, Generic
from abc import abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.interface import (
    Component,
    DataType,
    DataTable,
    DataPoint,
    Interface,
    MAP_STR_TO_ENUM
)


class SystematicSample:
    """A systematic sample of data points with optional weights.

    Attributes:
        support_data (DataTable[DataType]): Support data for sampling.
        normalized_weights (np.ndarray): Normalized weights of each data point.
        probability_covered (float): Total probability covered by the sample.
        N (int): Number of samples.
    """
    def __init__(self,
                 support_data: DataTable[DataType],
                 normalized_weights: np.ndarray,
                 probability_covered: float = 1.0):
        self.support_data = support_data
        self.normalized_weights = normalized_weights
        self.probability_covered = probability_covered
        self.N = len(self.normalized_weights)

    def variables_iter(self):
        """Iterate over all data points in the sample."""
        for i in range(self.N):
            yield self.support_data.get_point(i)

    def weighted_variables_iter(self):
        """Iterate over all data points with their corresponding weights."""
        for i in range(self.N):
            yield self.normalized_weights[i], self.support_data.get_point(i)

    def discrete_statistics(self):
        return DiscreteStatistics(
            name="discrete_stats_from_sample",
            support_data=self.support_data,
            probabilities=self.normalized_weights
        )


class Statistics(Component, Generic[DataType]):
    """Abstract base class for statistics computations.

    """
    @abstractmethod
    def systematic_sample(self,
                          N_max: int = None,
                          min_prob: float = None) -> SystematicSample:
        """Return a systematic sample of the underlying data.

        Args:
            N_max (int, optional): Maximum number of points to include.
            min_prob (float, optional): Minimum probability to be covered.

        Returns:
            SystematicSample: The generated sample.
        """
        pass

    @abstractmethod
    def expected_value(self) -> DataTable[DataType]:
        """Compute the expected value of the data.

        Returns:
            DataTable[DataType]: Expected value for each variable.
        """
        pass

class StatisticsType(Enum):
    DISCRETE_STATISTICS = "discrete_statistics"

class DiscreteStatistics(Statistics):
    """Concrete implementation of Statistics for discrete distributions."""

    def __init__(self,
                 name: str,
                 support_data: DataTable[DataType],
                 probabilities: np.ndarray):
        
        self.component_name = name
    
        # Discard zero-probability points
        probabilities_reduced = probabilities[probabilities > 0]
        support_data_reduced = {key: key_data[probabilities > 0, ...] for \
                                key, key_data in support_data.data.items()}
        
        #  Order support data by prevalence descending
        sorted_index = np.argsort(probabilities_reduced)[::-1]
        ordered_support = {}
        for var, data in support_data_reduced.items():
            ordered_support[var] = data[sorted_index, ...]
        self._ordered_probabilities = probabilities_reduced[sorted_index]
        self._ordered_support_data = DataTable(data=ordered_support,
                                              order=support_data.order)

        self.input_interface = Interface()
        self.output_interface = Interface(
            all_shapes={
                self._ordered_support_data.data_type: self._ordered_support_data.shapes()})
    
    def systematic_sample(self,
                          N_max: int = None,
                          min_prob: float = None) -> SystematicSample:
        """Return a systematic sample of the data points.

        Args:
            N_max (int, optional): Maximum number of samples. If None, include all points.
            min_prob (float, optional): Minimum probability to be covered. If both are specified, N_max has priority
            
        Returns:
            SystematicSample: Sampled points with normalized weights.

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
                               var, supp in self._ordered_support_data.data.items()}
        return SystematicSample(
            support_data=DataTable(data=support_data_subset,
                                   order=self._ordered_support_data.order),
            normalized_weights=weights / probability_covered,
            probability_covered=probability_covered
        )


    def expected_value(self) -> DataTable[DataType]:
        """Compute expected value of the discrete distribution.

        Returns:
            DataTable[DataType]: Expected value per variable.
        """
        return DataPoint({
            var: self._ordered_probabilities[np.newaxis, :] @ supp
            for var, supp in self._ordered_support_data.data.items()
        })

def statistics_from_dict(param_dict: Dict[str, Any]) -> Statistics:
    """Construct DiscreteStatistics from a dictionary.

    Args:
        param_dict (Dict[str, Any]): Dictionary containing 'data_type', 'support_data', and 'prevalence'.

    Returns:
        DiscreteStatistics: Constructed DIscreteStatistics object.
    """
    name = param_dict["name"]
    statistics_type = StatisticsType(param_dict["statistics_type"])
    data_type = MAP_STR_TO_ENUM[param_dict["data_type"]]
    
    if statistics_type == StatisticsType.DISCRETE_STATISTICS:
        support_data = DataTable({data_type(var): np.array(supp) for var, supp in param_dict["support_data"].items()})
        prevalence = np.array(param_dict["prevalence"])
        probabilities = prevalence / np.sum(prevalence)
            
        return DiscreteStatistics(
            name=name,
            support_data=support_data,
            probabilities=probabilities
        )
    else:
        raise NotImplementedError("Only discrete_statistics implemented.")
