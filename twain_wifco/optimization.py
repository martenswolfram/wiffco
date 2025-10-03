from typing import Dict, Any, Set
from abc import ABC, abstractmethod
import numpy as np
import itertools
from enum import Enum
from twain_wifco.interface import Ambient, Control, AccumulatedMetric
from twain_wifco.statistics import Statistics
from twain_wifco.control_input import ControlPolicy
from twain_wifco.plant_model import PlantModel
from twain_wifco.aggregation import Aggregation
from twain_wifco.metrics_accumulation import compute_aggregate, MetricsAccumulation
from twain_wifco.accumulated_constraint import AccumulatedConstraint
from twain_wifco.multi_metrics_handling import MultiMetricsHandling

class ControlEvaluationSystem:
    def __init__(self,
                 name: str, 
                 plant_model: PlantModel,
                 aggregation: Aggregation,
                 metrics_accumulation: MetricsAccumulation,
                 accumulated_constraint: AccumulatedConstraint,
                 multi_metrics_handling: MetricsAccumulation,
                 initial_control_policy: ControlPolicy):
        self.name = name
        self.plant_model = plant_model
        self.aggregation = aggregation
        self.metrics_accumulation = metrics_accumulation
        self.accumulated_constraint = accumulated_constraint
        self.multi_metrics_handling = multi_metrics_handling
        self.initial_control_policy = initial_control_policy

    def evaluate_ambient_condition(self,
                                   ambient_condition: Dict[Ambient, float],
                                   control_policy: ControlPolicy):
        return compute_aggregate(ambient_condition=ambient_condition,
                                 control_policy=control_policy,
                                 plant_model=self.plant_model,
                                 aggregation=self.aggregation)    

class OptimizationMethod(Enum):
    GRID_SEARCH = "grid_search"
    LAGRANGIAN_RELAXATION = "lagrangian_relaxation"

class ControlPolicyOptimization(ABC):
    def __init__(self,
                 optimization_name: str):
        self.optimization_name = optimization_name

    @abstractmethod
    def optimize_policy(self,
                        control_problem: ControlEvaluationSystem,
                        ambient_condition_statistics: Statistics,
                        initial_policy: ControlPolicy | None = None):
        pass

class GridSearchParams:
    def __init__(self,
                 num_ambient_conditions: int,
                 control_setpoints: Dict[Control, np.ndarray]):
        self.num_ambient_conditions = num_ambient_conditions
        self.control_setpoints = control_setpoints
        
def grid_search_params_from_dict(param_dict: Dict[str, Dict | Any]):
    num_ambient_conditions = param_dict["num_ambient_conditions"]
    control_setpoints = {}
    for ctrl_var, setpoints in param_dict["control_setpoints"].items():
        control_setpoints[Control(ctrl_var)] = np.array(setpoints)
    return GridSearchParams(num_ambient_conditions=num_ambient_conditions,
                            control_setpoints=control_setpoints)

class GridSearch(ControlPolicyOptimization):
    def __init__(self,
                 optimization_name: str,
                 optimization_params: GridSearchParams):
        super().__init__(optimization_name=optimization_name)
        self.num_ambient_conditions = optimization_params.num_ambient_conditions
        self.control_setpoints = optimization_params.control_setpoints

    def optimize_policy(self,
                        control_problem: ControlEvaluationSystem,
                        ambient_condition_statistics: Statistics,
                        initial_policy: ControlPolicy | None = None):
        # Number of model evaluations
        model_effective_controls = control_problem.plant_model.input_of_type(t=Control)
        model_effective_control_setpoint_vectors = [
            setpoint_vector for ctrl_var, setpoint_vector in self.control_setpoints.items() \
            if ctrl_var in model_effective_controls]
        ambient_condition_sample = ambient_condition_statistics.systematic_sample(N=self.num_ambient_conditions)
        
        model_effective_control_setpoint_combinations = list(itertools.product(*model_effective_control_setpoint_vectors))
        num_model_eval = len(model_effective_control_setpoint_combinations) * \
            self.num_ambient_conditions
        print(f"GridSearch: Performing {num_model_eval} model evaluations.")
        
        for ambient_condition in ambient_condition_sample.support_values.T:
            for control_setpoints in model_effective_control_setpoint_combinations:
                print(f"Control setpoints: {control_setpoints}")
        