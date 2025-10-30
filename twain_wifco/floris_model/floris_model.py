from typing import Dict, List, Any
import pathlib
import csv
import numpy as np
from floris import FlorisModel
from twain_wifco.interface import (
    Component,
    Interface,
    DataPoint,
    Ambient,
    Control,
    ModelOutput)
from twain_wifco.plant_model import PlantModel
from twain_wifco.floris_model.floris_interface import wifco2floris

class FlorisWindFarmModel(PlantModel):

    def __init__(self,
                 name: str,
                 floris_model: FlorisModel,
                 ambient_variables: List[Ambient] = [],
                 control_variables: List[Control] = [],
                 output_variables: List[ModelOutput] = []):
        
        self.component_name = name
        self._floris_model = floris_model
        self._n_turbines = self._floris_model.n_turbines
        self.input_interface = Interface(
            all_shapes={Ambient: {amb_var: () for amb_var in ambient_variables},
                        Control: {ctrl_var: (self._n_turbines,) for ctrl_var in control_variables}})

        self.output_interface = Interface(
            all_shapes={ModelOutput: {out_var: (self._n_turbines,) for out_var in output_variables}})

    @Component.with_validation
    def evaluate(
        self,
        meteorological_condition: DataPoint[Ambient],
        control_input: DataPoint[Control],
    ) -> DataPoint[ModelOutput]:
        
        # Assume all ambient conditions are scalar, all control inputs are per turbine
        kwargs = {wifco2floris(amb): np.expand_dims(val, 0) for amb, val in meteorological_condition.items()} | \
                {wifco2floris(ctrl): np.expand_dims(val, 0) for ctrl, val in control_input.items()}
        
        self._floris_model.set(**kwargs)
        self._floris_model.run()

        output_data = {}
        for output, shape in self.output_interface.shapes[ModelOutput].items():
            match output:
                case ModelOutput.ELECTRICAL_POWER:
                    if shape == ():
                        output_data[output] = self._floris_model.get_farm_power()
                    else:
                        output_data[output] = self._floris_model.get_turbine_powers().reshape(shape)
                case _:
                    raise ValueError(f"Model output {output.value} not provided by FLORIS model at this point")

        return DataPoint(output_data)
    
def floris_model_from_dict(param_dict: Dict[str, Any]):
    name = param_dict.get("name", "FLORIS wind farm model")
    floris_config_file = pathlib.Path(param_dict["floris_config_file"])
    floris_model = FlorisModel(configuration=floris_config_file)
    wind_farm_layout_file = param_dict.get("wind_farm_layout_file", None)
    if wind_farm_layout_file is not None:
        x = []
        y = []
        with open(pathlib.Path(wind_farm_layout_file), newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=";")
            for row in reader:
                x.append(row['x'])
                y.append(row['y'])
        floris_model.set(layout_x=x, layout_y=y)

    ambient_variables = [Ambient(amb_var) for amb_var in param_dict["ambient_variables"]]
    control_variables = [Control(ctrl_var) for ctrl_var in param_dict["control_variables"]]
    output_variables = [ModelOutput(out_var) for out_var in param_dict["output_variables"]]
    
    return FlorisWindFarmModel(name=name,
                               floris_model=floris_model,
                               ambient_variables=ambient_variables,
                               control_variables=control_variables,
                               output_variables=output_variables)