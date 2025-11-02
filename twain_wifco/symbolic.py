from typing import Dict, Any, Callable, Tuple, List, Type
import numpy as np
import sympy as sp

from twain_wifco.interface import (
    DataTable,
    MAP_ENUM_TO_STR,
    DataEnum,
    Ambient,
    Control,
    ModelOutput,
    Aggregated,
    AccumulatedMetric,
    get_default_value,
)

ENUM_ORDER = [Ambient, Control, ModelOutput, Aggregated, AccumulatedMetric]

class SymbolicFunction:

    def __init__(
        self,
        input_lists: Dict[Type[DataEnum], List[DataEnum]],
        input_shapes: Dict[Type[DataEnum], Dict[DataEnum, Tuple[int, ...]]],
        output_functions: Dict[Type[DataEnum], Dict[DataEnum, Callable[..., np.ndarray]]],
        output_shapes: Dict[Type[DataEnum], Dict[DataEnum, Tuple[int, ...]]],
    ):
        
        self.input_lists = input_lists
        if not len(self.input_lists):
            raise ValueError("At least one input must be specified for symbolic function.")
        self.input_shapes = input_shapes
        self.output_functions = output_functions
        self.output_shapes = output_shapes

    def evaluate(
        self,
        input_data: Dict[Type[DataEnum], DataTable[DataEnum]]
    ) -> Dict[Type[DataEnum], DataTable[DataEnum]]:
                
        # Input values in correct order for arguments of output function
        values = []
        for data_enum in ENUM_ORDER:
            if data_enum in self.input_lists:
                values.extend(input_data[data_enum][input_var] for \
                              input_var in self.input_lists[data_enum] if input_var in self.input_shapes[data_enum])
        num_points = len(values[0])
        out_data = {out_data_enum: DataTable(
            data={out_var: np.empty(shape=((num_points, ) + shape)) for \
            out_var, shape in output_shapes.items()}) for \
            out_data_enum, output_shapes in self.output_shapes.items()}
        for pt in np.arange(num_points):
            for out_data_enum, output_functions in self.output_functions.items():
                for out_var, func in output_functions.items():
                    out_data[out_data_enum][out_var][pt, ...] = \
                        func(*[value[pt, ...] for value in values])
        return out_data

def symbolic_function_from_dict(param_dict: Dict[str, Any]) -> SymbolicFunction:
    """Construct `SymbolicFunction` from a dictionary definition."""
    str_to_symbol: Dict[str, sp.Symbol] = {}
    input_lists: Dict[Type[DataEnum], List[DataEnum]] = {}
    input_shapes: Dict[Type[DataEnum], Dict[DataEnum, Tuple[int, ...]]] = {}
    symbols_list = []

    # Define input symbols and mappings
    symbol_mapping_dict: Dict[str, Dict] = param_dict["symbol_mappings"]
    for data_enum in ENUM_ORDER:
        if MAP_ENUM_TO_STR[data_enum] not in symbol_mapping_dict:
            continue        
        input_lists[data_enum] = []
        input_shapes[data_enum] = {}
        for input_var_str, sym_mapping in symbol_mapping_dict[MAP_ENUM_TO_STR[data_enum]].items():
            input_var = data_enum(input_var_str)
            sym_str = sym_mapping["name"]
            var_sym = sp.Symbol(sym_str)
            str_to_symbol[sym_str] = var_sym
            symbols_list.append(var_sym)
            input_lists[data_enum].append(input_var)
            input_shapes[data_enum][input_var] = tuple(sym_mapping["shape"])
    
    output_functions: Dict[Type[DataEnum], Dict[DataEnum, Callable[..., np.ndarray]]] = {}
    output_functions_dict: Dict[str, Dict] = param_dict["output_functions"]
    # Define output functions
    for data_enum in ENUM_ORDER:
        if MAP_ENUM_TO_STR[data_enum] not in output_functions_dict:
            continue        
        output_functions[data_enum] = {}
        for out_var_str, out_function_str in output_functions_dict[MAP_ENUM_TO_STR[data_enum]].items():
            expr = sp.sympify(out_function_str, locals=str_to_symbol)
            # Note that the output functions are defined for a single data point,
            # and need to be cast over the data table
            output_functions[data_enum][data_enum(out_var_str)] = sp.lambdify(
                symbols_list,
                expr,
                "numpy"
            )

    # Determine output shapes via default computation
    default_inputs = [get_default_value(data_var=in_var, shape=shape) for \
        data_enum in ENUM_ORDER if data_enum in input_shapes for \
            in_var, shape in input_shapes[data_enum].items()] 

    default_result = {
        data_enum: {
            out_var: func(*default_inputs) for \
                out_var, func in output_functions[data_enum].items()} for \
                    data_enum in ENUM_ORDER if data_enum in output_functions}
    output_shapes = {
        data_enum: {
            out_var: res.shape for out_var, res in default_result[data_enum].items()} for \
            data_enum in ENUM_ORDER if data_enum in default_result}
    
    return SymbolicFunction(
        input_lists=input_lists,
        input_shapes=input_shapes,
        output_functions=output_functions,
        output_shapes=output_shapes,
    )