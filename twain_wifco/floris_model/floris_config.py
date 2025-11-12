import pathlib
import csv
import numpy as np
from floris import FlorisModel
from twain_wifco.twain_surrogate import (
    ANN_DEL_BladeRoot,
    ANN_DEL_Shaft,
    ANN_DEL_TowerBase,
    ANN_DEL_YawBearings,
    ANN_Power)
from twain_wifco.interface import (
    DataVariable,
    Ambient,
    Control)

def wifco2floris(data_var: DataVariable):
    match data_var:
        case Ambient.WIND_SPEED: return "wind_speeds"
        case Ambient.WIND_DIRECTION: return "wind_directions"
        case Ambient.TURBULENCE_INTENSITY: return "turbulence_intensities"
        case Control.YAW_ANGLE: return "yaw_angles"
    raise ValueError(f"Data variable {data_var} cannot be converted to Floris name.")

def floris2wifco(var_str: str):
    match var_str:
        case "wind_speeds": return Ambient.WIND_SPEED
        case "wind_directions": return Ambient.WIND_DIRECTION
        case "turbulence_intensities": return Ambient.TURBULENCE_INTENSITY
        case "yaw_angles": return Control.YAW_ANGLE
    raise ValueError(f" Floris string {var_str} cannot be converted to Floris name.")


def configure_floris_model(floris_config_path_str: str,
                           wind_farm_layout_path_str: str | None = None):
    floris_config_file = pathlib.Path(floris_config_path_str)
    floris_model = FlorisModel(configuration=floris_config_file)
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
import numpy as np

def get_load_channels(floris_model: FlorisModel):
    """
    """
    n_turbines = floris_model.n_turbines
    n_ac = floris_model.core.flow_field.u.shape[0]

    u_velocity = floris_model.core.flow_field.u
    # sa-quantities order is "up", "right", "down", "left"
    saws = np.stack(arrays=(u_velocity[:, :, :, 2].mean(axis=2),
                            u_velocity[:, :, 2, :].mean(axis=2),
                            u_velocity[:, :, :, 0].mean(axis=2),
                            u_velocity[:, :, 0, :].mean(axis=2)),
                     axis=0)
    ti = floris_model.core.flow_field.turbulence_intensities
    sati = ti[np.newaxis, :, np.newaxis] * np.ones(shape=(4, 1, n_turbines))
    yaw_angles = floris_model.core.farm.yaw_angles[np.newaxis, ...]
    # No power regulation for now
    power_demands = np.full_like(yaw_angles, fill_value=100)

    ann_input = np.concatenate((saws, sati, yaw_angles, power_demands))
    ann_input_flat = np.reshape(ann_input, shape=(10, n_turbines * n_ac))
    tower_base_del = ANN_DEL_TowerBase.ANN_DEL_TowerBase(x1=ann_input_flat)
    return tower_base_del