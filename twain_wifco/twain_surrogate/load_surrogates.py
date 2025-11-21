from typing import List
import numpy as np
import floris
from enum import Enum
from twain_wifco.twain_surrogate import (
    ANN_DEL_BladeRoot,
    ANN_DEL_Shaft,
    ANN_DEL_TowerBase,
    ANN_DEL_YawBearings)

class DamageComponent(Enum):
    BLADE_ROOT = "blade_root"
    SHAFT = "shaft"
    TOWER_BASE = "tower_base"
    YAW_BEARINGS = "yaw_bearings"

def dmg_equivalent_loads(floris_model: floris.FlorisModel,
                         damage_components: List[DamageComponent]):
    """
    """
    n_turbines = floris_model.n_turbines
    n_ac = floris_model.core.flow_field.u.shape[0]
    n_dc = len(damage_components)

    u_velocity = floris_model.core.flow_field.u
    # u_velocity has shape (n_ac, n_turbines, n_y=3, n_z=3)
    
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
    
    dels = np.empty(shape=(n_ac, n_turbines, n_dc))
    for i_dc, dmg_component in enumerate(damage_components):
        if dmg_component == DamageComponent.TOWER_BASE:
            tower_base_del_flat = ANN_DEL_TowerBase.ANN_DEL_TowerBase(x1=ann_input_flat)
            dels[..., i_dc] = np.reshape(tower_base_del_flat, shape=(n_ac, n_turbines))
        elif dmg_component == DamageComponent.BLADE_ROOT:
            blade_root_del_flat = ANN_DEL_BladeRoot.ANN_DEL_BladeRoot(x1=ann_input_flat)
            dels[..., i_dc] = np.reshape(blade_root_del_flat, shape=(n_ac, n_turbines))
        elif dmg_component == DamageComponent.SHAFT:
            shaft_del_flat = ANN_DEL_Shaft.ANN_DEL_Shaft(x1=ann_input_flat)
            dels[..., i_dc] = np.reshape(shaft_del_flat, shape=(n_ac, n_turbines))
        elif dmg_component == DamageComponent.YAW_BEARINGS:
            yaw_bearings_del_flat = ANN_DEL_YawBearings.ANN_DEL_YawBearings(x1=ann_input_flat)
            dels[..., i_dc] = np.reshape(yaw_bearings_del_flat, shape=(n_ac, n_turbines))
    
    return dels