from twain_wifco.interface import (
    Ambient,
    DataVariable,
    Control
)

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
