from typing import Dict, Any
from abc import ABC, abstractmethod
from enum import Enum
from twain_wiffco.ambient_conditions import AmbientStatistics
from twain_wiffco.control_input import ControlPolicy
from twain_wiffco.wind_farm_model import WindFarmModel
from twain_wiffco.output_aggregation import OutputAggregation
from twain_wiffco.model_output import OutputVariable

class AccumulationType(Enum):
    DISCOUNTED_INTEGRATION = "discounted_integration"

class AccumulatedMetric(Enum):
    REVENUE = "revenue"

def expected_accumulated_metric(duration: float,
                                ambient_condition_statistics: AmbientStatistics,
                                control_policy: ControlPolicy,
                                wind_farm_model: WindFarmModel,
                                output_aggregation: OutputAggregation):
    pass

class OutputAcumulationParams:
    def __init__(self, param_dict: Dict[str, Any]):
        self.in_out_mappings = {OutputVariable(instant_var): AccumulatedMetric(acc_metric) for \
                                 instant_var, acc_metric in param_dict["in_out_mappings"].items()}
        

class OutputAccumulation(ABC):
    
    def __init__(self,
                 name: str,
                 ambient_condition_statistics: AmbientStatistics,
                 control_policy: ControlPolicy,
                 wind_farm_model: WindFarmModel,
                 output_aggregation: OutputAggregation):
        
        # Initialization
        self.name = name
        self.ambient_condition_statistics = ambient_condition_statistics
        self.control_policy = control_policy
        self.wind_farm_model = wind_farm_model 
        self.output_aggregation = output_aggregation

        # Validation
        # Ambient condition statistics to wind farm model
        

    def validate_inputs(self,
                        ambient_condition_statistics: AmbientStatistics,
                        control_policy: ControlPolicy,
                        wind_farm_model: WindFarmModel,
                        output_aggregation: OutputAggregation):
        pass

    @abstractmethod
    def _compute_accumulation(self,
                              duration: float,
                              ambient_condition_statistics: AmbientStatistics,
                              control_policy: ControlPolicy,
                              wind_farm_model: WindFarmModel):
        pass

class DiscountedIntegratorParams:
    def __init__(self,
                 param_dict: Dict[str, Any]):
        self.in_out_mappings = {OutputVariable(instant_var): AccumulatedMetric(acc_metric) for \
                                 instant_var, acc_metric in param_dict["in_out_mappings"].items()}
        self.discount_rate = float(param_dict["discount_rate"]) 
        
class DiscountedIntegrator(OutputAccumulation):
    def __init__(self,
                 name: str,
                 params: DiscountedIntegratorParams):
        super().__init__(name=name)
        self.in_out_mappings = params.in_out_mappings
        self.discount_factor = 1 / (1 + params.discount_rate)

    def _validate_input(self,
                        input_variables:  Dict[OutputVariable, float]):
        if not (self.in_out_mappings.keys() <= input_variables.keys()):
            raise ValueError("Insufficient input variables for Discounted Integrator '{}'.".format(self.name))
        
    def _compute_accumulation(self,
                              duration: float,
                              input_variables: Dict[OutputVariable, float]):
        accumulated_metrics = {}
        for input, output in self.in_out_mappings.items():
