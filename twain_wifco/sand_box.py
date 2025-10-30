from enum import Enum
from typing import TypeVar, Type

class Ambient(Enum):
    WIND_SPEED = "wind_speed"
    WIND_DIRECTION = "wind_direction"
    ELECTRICITY_PRICE = "electricity_price"

class Control(Enum):
    POWER_REGULATION = "power_regulation"
    yaw_angle = "yaw_angle"

class ModelOutput(Enum):
    ELECTRICAL_POWER = "electrical_power"
    DAMAGE_RATE = "damage_rate"

class Aggregated(Enum):
    REVENUE_RATE = "revenue_rate"
    DAMAGE_RATE = "damage_rate"
    
class AccumulatedMetric(Enum):
    REVENUE = "revenue"
    ACCRUED_DAMAGE = "accrued_damage"

DataTypes = (Ambient,
             Control,
             ModelOutput,
             Aggregated,
             AccumulatedMetric)

DataType = TypeVar("T",
            Ambient,
            Control,
            ModelOutput,
            Aggregated,
            AccumulatedMetric)

def create_enum_object(type: Type[DataType], value: str):
    return type(value)

electrical_power = create_enum_object(type=ModelOutput,
                                  value="electrical_power")