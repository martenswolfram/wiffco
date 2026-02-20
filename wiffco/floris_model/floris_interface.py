import pathlib
import csv
import floris
from wiffco.interface import (
    DataVariable,
    Ambient,
    Control)

def wifco2floris(data_var: DataVariable):
    match data_var:
        case Ambient.WIND_SPEED_MPS: return "wind_speeds"
        case Ambient.WIND_DIRECTION_DEG: return "wind_directions"
        case Ambient.TURBULENCE_INTENSITY: return "turbulence_intensities"
        case Control.YAW_ANGLE_DEG: return "yaw_angles"
    raise ValueError(f"Data variable {data_var} cannot be converted to Floris name.")

def floris2wifco(var_str: str):
    match var_str:
        case "wind_speeds": return Ambient.WIND_SPEED_MPS
        case "wind_directions": return Ambient.WIND_DIRECTION_DEG
        case "turbulence_intensities": return Ambient.TURBULENCE_INTENSITY
        case "yaw_angles": return Control.YAW_ANGLE_DEG
    raise ValueError(f" Floris string {var_str} cannot be converted to Floris name.")

def configure_floris_model(floris_config_path_str: str,
                           wind_farm_layout_path_str: str | None = None):
    floris_config_file = pathlib.Path(floris_config_path_str)
    floris_model = floris.FlorisModel(configuration=floris_config_file)
    if wind_farm_layout_path_str is not None:
        wind_farm_layout_file = pathlib.Path(wind_farm_layout_path_str)
        x = []
        y = []
        with open(wind_farm_layout_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=";")
            for row in reader:
                x.append(row['x'])
                y.append(row['y'])
        floris_model.set(layout_x=x, layout_y=y)
    return floris_model
