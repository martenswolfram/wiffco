from typing import Dict, Any, Generic
from abc import abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.interface import (
    Component,
    ComponentParams,
    DataType,
    DataTable,
    DataPoint,
    Interface,
    data_type_from_string
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
        """Return a DiscreteStatistics object representing this sample."""
        discrete_statistics_params = DiscreteStatisticsParams(
            support_data=self.support_data,
            probabilities=self.normalized_weights
        )
        return DiscreteStatistics(
            statistics_name="discrete_stats_from_sample",
            statistics_params=discrete_statistics_params
        )


class Statistics(Component, Generic[DataType]):
    """Abstract base class for statistics computations.

    Args:
        statistics_name (str): Name of the statistics component.
        statistics_params (ComponentParams): Parameters for the statistics.
    """
    def __init__(self,
                 statistics_name: str,
                 statistics_params: ComponentParams):
        super().__init__(component_name=statistics_name,
                         component_params=statistics_params)

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
    def expected_value(self) -> DataPoint[DataType]:
        """Compute the expected value of the data.

        Returns:
            DataPoint[DataType]: Expected value for each variable.
        """
        pass


class StatisticsType(Enum):
    DISCRETE_STATISTICS = "discrete_statistics"


class DiscreteStatisticsParams(ComponentParams):
    """Parameters for DiscreteStatistics.

    Args:
        support_data (DataTable[DataType]): Data points.
        probabilities (np.ndarray): Weights/probabilities of data points.
    """
    def __init__(self,
                 support_data: DataTable[DataType],
                 probabilities: np.ndarray):
        self.support_data = support_data
        self.probabilities = probabilities / np.sum(probabilities)

    def input_interface(self) -> Interface:
        """Input interface (empty for DiscreteStatistics)."""
        return Interface(all_shapes={})

    def output_interface(self) -> Interface:
        """Output interface containing support data shapes."""
        return Interface(all_shapes={
            self.support_data.data_type: self.support_data.shapes()
        })

def discrete_statistics_params_from_dict(param_dict: Dict[str, Any]) -> DiscreteStatisticsParams:
    """Construct DiscreteStatisticsParams from a dictionary.

    Args:
        param_dict (Dict[str, Any]): Dictionary containing 'data_type', 'support_data', and 'prevalence'.

    Returns:
        DiscreteStatisticsParams: Constructed parameters object.
    """
    data_type = data_type_from_string(param_dict["data_type"])
    support_data = DataTable({data_type(var): np.array(supp) for var, supp in param_dict["support_data"].items()})
    prevalence = np.array(param_dict["prevalence"])
    probabilities = prevalence / np.sum(prevalence)
        
    return DiscreteStatisticsParams(
        support_data=support_data,
        probabilities=probabilities
    )

class DiscreteStatistics(Statistics):
    """Concrete implementation of Statistics for discrete distributions."""

    def __init__(self,
                 statistics_name: str,
                 statistics_params: DiscreteStatisticsParams):
        super().__init__(statistics_name=statistics_name,
                         statistics_params=statistics_params)

        # Discard zero-probability points
        probabilities = statistics_params.probabilities[statistics_params.probabilities > 0]
        support = {key: key_data[statistics_params.probabilities > 0, ...] for \
                        key, key_data in statistics_params.support_data.data.items()}
        
        #  Order support data by prevalence descending
        sorted_index = np.argsort(probabilities)[::-1]
        ordered_support = {}
        for var, data in support.items():
            ordered_support[var] = data[sorted_index, ...]
        self.ordered_probabilities = probabilities[sorted_index]
        self.ordered_support_data = DataTable(data=ordered_support,
                                              order=statistics_params.support_data.order)

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
                cumsum = np.cumsum(self.ordered_probabilities)
                if min_prob < 1:
                    N_max = np.searchsorted(cumsum, min_prob) + 1
        elif N_max <= 0:
            raise ValueError("N_max must be a positive integer.")

        weights = self.ordered_probabilities[:N_max]
        probability_covered = np.sum(weights)
        support_data_subset = {var: supp[:N_max, ...] for var, supp in self.ordered_support_data.data.items()}
        return SystematicSample(
            support_data=DataTable(data=support_data_subset,
                                   order=self.ordered_support_data.order),
            normalized_weights=weights / probability_covered,
            probability_covered=probability_covered
        )


    def expected_value(self) -> DataPoint[DataType]:
        """Compute expected value of the discrete distribution.

        Returns:
            DataPoint[DataType]: Expected value per variable.
        """
        return DataPoint({
            var: self.ordered_probabilities @ supp
            for var, supp in self.ordered_support_data.data.items()
        })
