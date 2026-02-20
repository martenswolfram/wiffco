from typing import Dict, Any, Callable, Tuple, List, Type
import numpy as np
import sympy as sp

from wiffco.interface import (
    DataTable,
    MAP_ENUM_TO_STR,
    DataEnum,
    Ambient,
    Control,
    ModelOutput,
    Aggregate,
    AccumulatedMetric,
    Interface,
)

NUMPY_FUNC_MAPPINGS = {"sum": np.sum}
ENUM_ORDER = [Ambient, Control, ModelOutput, Aggregate, AccumulatedMetric]

class SymbolicFunction:

    def __init__(
        self,
        input_lists: Dict[Type[DataEnum], List[DataEnum]],
        output_functions: Dict[Type[DataEnum], Dict[DataEnum, Callable[..., np.ndarray]]],
    ):
        
        self.input_lists = input_lists
        if not len(self.input_lists):
            raise ValueError("At least one input must be specified for symbolic function.")
        self.output_functions = output_functions

    def evaluate(
        self,
        input_data: Dict[Type[DataEnum], DataTable[DataEnum] | None]
    ) -> Dict[Type[DataEnum], DataTable[DataEnum]]:
                
        # Input values in correct order for arguments of output function
        values = []
        for data_enum in ENUM_ORDER:
            if data_enum in self.input_lists and input_data[data_enum] is not None:
                values.extend(input_data[data_enum][input_var] for \
                              input_var in self.input_lists[data_enum] if input_var in self.input_lists[data_enum])
        num_points = len(values[0])
        # First compute numpy arrays
        out_data_arrays = {out_data_enum: 
            {out_var: None for \
            out_var in output_functions} for \
            out_data_enum, output_functions in self.output_functions.items()}
        for pt in np.arange(num_points):
            for out_data_enum, output_functions in self.output_functions.items():
                for out_var, func in output_functions.items():
                    res = func(*[value[pt, ...] for value in values])
                    if out_data_arrays[out_data_enum][out_var] is None:
                        # Infer shape from first output
                        out_data_arrays[out_data_enum][out_var] = np.empty(shape=((num_points, ) + res.shape))
                    out_data_arrays[out_data_enum][out_var][pt, ...] = res
        # Make DataTable
        return {out_data_enum: DataTable(data=data_arrays) for out_data_enum, data_arrays in out_data_arrays.items()}
    
    def input_interface(self):
        return Interface(
            all_shapes={data_enum: {in_var: None for \
                                    in_var in in_vars} for \
                                        data_enum, in_vars in self.input_lists.items()}
        )

    def output_interface(self):
        return Interface(
            all_shapes={data_enum: {out_var: None for \
                                    out_var in out_functions} for \
                        data_enum, out_functions in self.output_functions.items()}
        )

def sym_input_mappings(data_enum: Type[DataEnum],
                       sym_mappings_dict: Dict[str, Any]):
    str_to_symbol = {}
    input_list = []
    symbols_list = []
    for input_var_str, sym_name in sym_mappings_dict.items():
        input_var = data_enum(input_var_str)
        var_sym = sp.Symbol(sym_name)
        str_to_symbol[sym_name] = var_sym
        symbols_list.append(var_sym)
        input_list.append(input_var)
    return str_to_symbol, input_list, symbols_list

def output_function(data_enum: Type[DataEnum],
                    str_to_symbol: Dict[str, sp.Symbol],
                    symbols_list: List[sp.Symbol],
                    output_function_dict: Dict[str, Any]):
        
    output_functions = {}
    for out_var_str, out_function_str in output_function_dict.items():
        expr = sp.sympify(
            out_function_str, locals=(
                str_to_symbol |
                {name: sp.Function(name) for name in NUMPY_FUNC_MAPPINGS.keys()}))
        # Note that the output functions are defined for a single data point,
        # and need to be cast over the data table
        output_functions[data_enum(out_var_str)] = sp.lambdify(
            symbols_list,
            expr,
            modules=[NUMPY_FUNC_MAPPINGS, "numpy"]
        )
    return output_functions

def symbolic_function_from_dict(param_dict: Dict[str, Any]) -> SymbolicFunction:
    str_to_symbol: Dict[str, sp.Symbol] = {}
    input_lists: Dict[Type[DataEnum], List[DataEnum]] = {}
    symbols_list = []

    # Define input symbols and mappings
    input_sym_mappings_dict: Dict[str, Dict] = param_dict["input_symbol_mappings"]
    for data_enum in ENUM_ORDER:
        data_enum_str = MAP_ENUM_TO_STR[data_enum]
        if data_enum_str not in input_sym_mappings_dict:
            continue
        enum_str_to_symbol, enum_input_list, enum_symbols_list = sym_input_mappings(
            data_enum=data_enum,
            sym_mappings_dict=input_sym_mappings_dict[data_enum_str])
        str_to_symbol |= enum_str_to_symbol
        input_lists[data_enum] = enum_input_list
        symbols_list += enum_symbols_list
            
    output_functions: Dict[Type[DataEnum], Dict[DataEnum, Callable[..., np.ndarray]]] = {}
    output_functions_dict: Dict[str, Dict] = param_dict["output_functions"]
    # Define output functions
    for data_enum in ENUM_ORDER:
        data_enum_str = MAP_ENUM_TO_STR[data_enum]
        if data_enum_str not in output_functions_dict:
            continue        
        output_functions[data_enum] = output_function(
            data_enum=data_enum,
            str_to_symbol=str_to_symbol,
            symbols_list=symbols_list,
            output_function_dict=output_functions_dict[data_enum_str]
        )
    
    return SymbolicFunction(
        input_lists=input_lists,
        output_functions=output_functions,
    )