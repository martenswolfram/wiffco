from typing import Dict, Any, List, Type, Callable
from abc import abstractmethod
from enum import Enum
import numpy as np
import sympy as sp
from twain_wifco.interface import (
    DataPoint,
    Component,
    ComponentParams,
    Ambient,
    Control,
    ModelOutput,
    Interface,
    DataVariable)
from twain_wifco.utils import (
    ScatteredInterpolatorParams,
    ScatteredInterpolator,
    scattered_interpolator_params_from_dict)

class PlantModel(Component):
    def __init__(self,
                 plant_name: str,
                 plant_params: ComponentParams):
        super().__init__(component_name=plant_name,
                         component_params=plant_params)
                
    def evaluate(self,
                 meteorological_condition: DataPoint[Ambient],
                 control_input: DataPoint[Control]):

        self.validate_inputs(input_data={Ambient: meteorological_condition,
                                         Control: control_input})

        return self._evaluate(meteorological_condition=meteorological_condition,
                              control_input=control_input)

    @abstractmethod
    def _evaluate(self,
                  meteorological_condition: Dict[Ambient, np.ndarray],
                  control_input: Dict[Control, np.ndarray]):
        pass

class ModelType(Enum):
    FACTORIZED_SCATTERED_INTERPOLATOR = "factorized_scattered_interpolator"
    SYMBOLIC = "symbolic"

class FactorizedScatteredInterpParams(ComponentParams):
    def __init__(self,
                 control_interp_params: ScatteredInterpolatorParams,
                 ambient_interp_params: ScatteredInterpolatorParams):
        if control_interp_params.out_data.shapes() != control_interp_params.out_data.shapes():
            raise ValueError("Out variables shapes of interpolation factors must be identical.")
        self.control_interp_params = control_interp_params
        self.ambient_interp_params = ambient_interp_params
        
    def input_interface(self) -> Interface:
        return Interface(all_shapes={Ambient: self.ambient_interp_params.support_data.shapes(),
                                               Control: self.control_interp_params.support_data.shapes()})

    def output_interface(self) -> Interface:
        return Interface(all_shapes={ModelOutput: self.ambient_interp_params.out_data.shapes()})
    
def factorized_scattered_interp_params_from_dict(
        param_dict: Dict[str, Dict | Any]):
    control_interp_params = scattered_interpolator_params_from_dict(
        param_dict=param_dict["control"],
        support_data_type=Control,
        out_data_type=ModelOutput)
    ambient_interp_params = scattered_interpolator_params_from_dict(
        param_dict=param_dict["ambient"],
        support_data_type=Ambient,
        out_data_type=ModelOutput)
    return FactorizedScatteredInterpParams(
        control_interp_params=control_interp_params,
        ambient_interp_params=ambient_interp_params
    )

class FactorizedScatteredInterp(PlantModel):
    def __init__(self,
                 plant_name: str,
                 plant_params: FactorizedScatteredInterpParams):
        super().__init__(plant_name=plant_name,
                         plant_params=plant_params)
        
        # Initialize interpolation factors
        self.ambient_interp = ScatteredInterpolator(
            interpolator_params=plant_params.ambient_interp_params)
        self.control_interp = ScatteredInterpolator(
            interpolator_params=plant_params.control_interp_params)
        
    def _evaluate(self,
                  meteorological_condition: Dict[Ambient, np.ndarray],
                  control_input: Dict[Control, np.ndarray]):
        ambient_eval = self.ambient_interp.evaluate(query=meteorological_condition)
        control_eval = self.control_interp.evaluate(query=control_input)
        return DataPoint({out_var: ambient_eval[out_var] * control_eval[out_var] for \
                          out_var in self.ambient_interp.out_data_point.keys()})
            
        
class SymbolicModelParams(ComponentParams):
    def __init__(self,
                 symbols: Dict[str, sp.Symbol],
                 control_mappings: Dict[Control, str],
                 ambient_mappings: Dict[Ambient, str],
                 output_functions: Dict[ModelOutput, Callable[..., np.ndarray]]):
        self.symbols = symbols
        self.control_mappings = control_mappings
        self.ambient_mappings = ambient_mappings
        self.output_functions = output_functions
        
    def input_interface(self) -> Interface:
        return Interface(
            all_shapes={    
                Ambient: {
                    amb_var: None for \
                    amb_var in self.ambient_mappings.keys()},
                Control: {
                    ctrl_var: None for \
                    ctrl_var in self.control_mappings.keys()}
                }
            )

    def output_interface(self) -> Interface:
        return Interface(all_shapes={
            Type[ModelOutput]: {out_var: None for \
                                out_var in self.output_functions.keys()}})
    
def symbolic_model_params_from_dict(
        param_dict: Dict[str, Dict | Any]):
    symbol_mapping_dict: Dict[str, Dict] = param_dict["symbol_mappings"]
    symbols = {}
    control_mappings = {}
    for ctrl_var, sym_name in symbol_mapping_dict["control"].items():
        symbols[sym_name] = sp.Symbol(sym_name)
        control_mappings[Control(ctrl_var)] = sym_name
    ambient_mappings = {}
    for amb_var, sym_name in symbol_mapping_dict["ambient"].items():
        symbols[sym_name] = sp.Symbol(sym_name)
        ambient_mappings[Ambient(amb_var)] = sym_name
    
    output_functions = {}
    for output, expr_str in param_dict["output_functions"].items():
        expr = sp.sympify(expr_str, locals=symbols)
        output_functions[ModelOutput(output)] = \
            sp.lambdify(tuple(symbols.values()), expr, "numpy")

    return SymbolicModelParams(
        symbols=symbols,
        control_mappings=control_mappings,
        ambient_mappings=ambient_mappings,
        output_functions=output_functions
    )

class SymbolicModel(PlantModel):
    def __init__(self,
                 plant_name: str,
                 plant_params: SymbolicModelParams):
        super().__init__(plant_name=plant_name,
                         plant_params=plant_params)
        
        self.params = plant_params
        
    def _evaluate(self,
                  meteorological_condition: Dict[Ambient, np.ndarray],
                  control_input: Dict[Control, np.ndarray]):
        value_map = {}
        for ctrl_var, symbol_name in self.params.control_mappings.items():
            value_map[symbol_name] = control_input[ctrl_var]
        for amb_var, symbol_name in self.params.ambient_mappings.items():
            value_map[symbol_name] = meteorological_condition[amb_var]
        
        results = {}
        symbol_values = [value_map[s] for s in self.params.symbols.keys()]
        for model_output, fun in self.params.output_functions.items():
            results[model_output] = fun(*symbol_values)

        return DataPoint(results)
            
        
