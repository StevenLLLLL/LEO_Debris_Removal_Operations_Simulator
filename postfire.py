"""
C

When there is an effective solution, calculate the post-launch effects in the simulation, including recoil, orbital
changes, and whether the compensation budget is sufficient, and returns the post-launch report.
"""

"""Assess the immediate recoil in a simulated net release."""

from dataclasses import dataclass
from math import isclose, isfinite

import numpy as np

from data_source import SpaceObject


@dataclass(frozen=True)
class ReleaseRecoil:
    """Velocity vectors in the same coordinate frame as the input."""

    recoil_delta_v_mps: np.ndarray  # How much did the recoil change the satellite's speed
    servicer_velocity_after_mps: np.ndarray  # full speed of the satellite after recoil


def validate_postfire_inputs(
    servicer_before: SpaceObject,
    debris_before: SpaceObject,
    *, # include name for parameters under this
    net_speed_mps: float,
    release_direction: np.ndarray,
    direction_frame: str,
    servicer_mass_after_release_kg: float,
    net_mass_kg: float,
) -> None:
    """
    Raise ValueError when the release inputs violate the handoff contract.

    Both objects describe the same instant just before release. Positions are
    in metres and velocities are in metres per second. The release direction
    is a unit vector in the same coordinate frame as the two states.
    """
    if servicer_before.time.utcoffset() is None or debris_before.time.utcoffset() is None:
        raise ValueError("Both state times must include a time zone.")
    if servicer_before.time != debris_before.time:
        raise ValueError("Servicer and debris states must have the same time.")

    if servicer_before.frame != debris_before.frame:
        raise ValueError("Servicer and debris states must use the same coordinate frame.")
    if not servicer_before.frame or direction_frame != servicer_before.frame:
        raise ValueError("Release direction must use the same coordinate frame as the states.")

    for name, state in (("Servicer", servicer_before), ("Debris", debris_before)):
        for field_name, vector in (("position", state.position), ("velocity", state.velocity)):
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError(f"{name} {field_name} must contain three finite values.")

    if release_direction.shape != (3,) or not np.isfinite(release_direction).all():
        raise ValueError("Release direction must contain three finite values.")
    if not isclose(float(np.linalg.norm(release_direction)), 1.0, rel_tol=1e-6):
        raise ValueError("Release direction must be a unit vector.")

    if not isfinite(net_speed_mps) or net_speed_mps < 0:
        raise ValueError("Net speed must be finite and non-negative (m/s).")
    if not isfinite(servicer_mass_after_release_kg) or servicer_mass_after_release_kg <= 0:
        raise ValueError("Servicer mass after release must be finite and positive (kg).")
    if not isfinite(net_mass_kg) or net_mass_kg <= 0:
        raise ValueError("Net mass must be finite and positive (kg).")


def calculate_release_recoil(
    servicer_before: SpaceObject,
    debris_before: SpaceObject,
    *,
    net_speed_mps: float,
    release_direction: np.ndarray,
    direction_frame: str,
    servicer_mass_after_release_kg: float,
    net_mass_kg: float,
) -> ReleaseRecoil:
    """
    Compute the velocity change using an instantaneous separation model.

    Net speed is measured relative to the servicer immediately after release.
    The servicer and attached net share a velocity before separation.
    """
    validate_postfire_inputs(
        servicer_before,
        debris_before,
        net_speed_mps=net_speed_mps,
        release_direction=release_direction,
        direction_frame=direction_frame,
        servicer_mass_after_release_kg=servicer_mass_after_release_kg,
        net_mass_kg=net_mass_kg,
    )

    total_mass_kg = servicer_mass_after_release_kg + net_mass_kg
    relative_net_velocity = net_speed_mps * release_direction
    recoil_delta_v = -(net_mass_kg / total_mass_kg) * relative_net_velocity

    return ReleaseRecoil(
        recoil_delta_v_mps=recoil_delta_v,
        servicer_velocity_after_mps=servicer_before.velocity + recoil_delta_v,
    )
