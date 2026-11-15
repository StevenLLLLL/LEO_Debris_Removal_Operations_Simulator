"""
C

Assess recoil, orbital changes, and the velocity correction needed for
immediate compensation after a simulated net release.
"""

"""Assess the immediate recoil in a simulated net release."""

from dataclasses import dataclass
from math import isclose, isfinite

import numpy as np

from data_source import SpaceObject


EARTH_MU_M3_S2 = 3.986004418e14
EARTH_MEAN_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class ReleaseRecoil:
    """
    Velocity vectors in the same coordinate frame as the input.
    """

    recoil_delta_v_mps: np.ndarray  # How much did the recoil change the satellite's speed
    servicer_velocity_after_mps: np.ndarray  # full speed of the satellite after recoil


@dataclass(frozen=True)
class OrbitAltitudes:
    """
    Heights above the spherical Earth reference surface, in metres. Include both lowest and highest altitude.
    """

    perigee_altitude_m: float
    apogee_altitude_m: float


@dataclass(frozen=True)
class OrbitAltitudeChange:
    """
    Heights above the spherical Earth reference surface, in metres.
    Include both lowest and highest altitude, before and after shooting
    """
    before: OrbitAltitudes
    after: OrbitAltitudes

    @property
    def perigee_change_m(self) -> float:
        return self.after.perigee_altitude_m - self.before.perigee_altitude_m

    @property
    def apogee_change_m(self) -> float:
        return self.after.apogee_altitude_m - self.before.apogee_altitude_m


@dataclass(frozen=True)
class RecoilCompensation:
    """Immediate velocity correction in the recoil vector's coordinate frame."""

    correction_delta_v_mps: np.ndarray
    required_delta_v_mps: float


@dataclass(frozen=True)
class PostFireReport:
    """Immediate release effects and the correction that cancels the recoil."""

    recoil: ReleaseRecoil
    orbit_change: OrbitAltitudeChange
    compensation: RecoilCompensation


def validate_postfire_inputs(
    servicer_before: SpaceObject,  # state of us right before shoot
    debris_before: SpaceObject,  # debris right before we shoot
    *, # include name for parameters under this
    net_speed_mps: float,  # speed of net when shoot
    release_direction: np.ndarray,  # [x, y, z] direction of the net
    direction_frame: str,  # for release_direction
    servicer_mass_after_release_kg: float, # as is
    net_mass_kg: float,  # mass of the net we shoot
) -> None:
    """
    Raise ValueError when the release inputs violate the handoff contract.

    Both objects describe the same instant just before release. Positions are
    in metres and velocities are in metres per second. The release direction
    is a unit vector in the same coordinate frame as the two states.
    """
    # check if time satisfied requirements
    if servicer_before.time.utcoffset() is None or debris_before.time.utcoffset() is None:
        raise ValueError("Both state times must include a time zone.")
    if servicer_before.time != debris_before.time:
        raise ValueError("Servicer and debris states must have the same time.")

    # Check if coordinate and frame satisfied requirements
    if servicer_before.frame != debris_before.frame:
        raise ValueError("Servicer and debris states must use the same coordinate frame.")
    if not servicer_before.frame or direction_frame != servicer_before.frame:
        raise ValueError("Release direction must use the same coordinate frame as the states.")

    # check velocity and location of us and the debris
    for name, state in (("Servicer", servicer_before), ("Debris", debris_before)):
        for field_name, vector in (("position", state.position), ("velocity", state.velocity)):
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError(f"{name} {field_name} must contain three finite values.")

    # check direction of net,  provide only the direction
    if release_direction.shape != (3,) or not np.isfinite(release_direction).all():
        raise ValueError("Release direction must contain three finite values.")
    if not isclose(float(np.linalg.norm(release_direction)), 1.0, rel_tol=1e-6):
        raise ValueError("Release direction must be a unit vector.")

    # check velocity and mass, with net on and off
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

    # Make sure data is valid
    validate_postfire_inputs(
        servicer_before,
        debris_before,
        net_speed_mps=net_speed_mps,
        release_direction=release_direction,
        direction_frame=direction_frame,
        servicer_mass_after_release_kg=servicer_mass_after_release_kg,
        net_mass_kg=net_mass_kg,
    )

    total_mass_kg = servicer_mass_after_release_kg + net_mass_kg  # before shooting
    relative_net_velocity = net_speed_mps * release_direction
    recoil_delta_v = -(net_mass_kg / total_mass_kg) * relative_net_velocity  # conservation of momentum

    # return change in velocity and velocity after shooting
    return ReleaseRecoil(
        recoil_delta_v_mps=recoil_delta_v,
        servicer_velocity_after_mps=servicer_before.velocity + recoil_delta_v,
    )


def calculate_orbit_altitudes(
    position: np.ndarray,
    velocity: np.ndarray,
) -> OrbitAltitudes:
    """Estimate the two-body orbit's apsides from one Earth-centered state.

    Position is in metres and velocity is in metres per second, expressed in
    the same inertial coordinate frame. A negative altitude indicates that
    the modeled orbit intersects the spherical Earth.
    """
    for name, vector in (("Position", position), ("Velocity", velocity)):
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"{name} must contain three finite values.")

    radius_m = float(np.linalg.norm(position))
    if not isfinite(radius_m) or radius_m == 0:
        raise ValueError("Distance from Earth's centre must be finite and positive.")

    specific_energy = 0.5 * float(np.dot(velocity, velocity)) - EARTH_MU_M3_S2 / radius_m
    if not isfinite(specific_energy) or specific_energy >= 0:
        raise ValueError("A bound orbit is required to calculate a finite apogee.")

    angular_momentum = np.cross(position, velocity)
    angular_momentum_squared = float(np.dot(angular_momentum, angular_momentum))
    if not isfinite(angular_momentum_squared) or angular_momentum_squared <= 0:
        raise ValueError("A non-radial orbit is required to calculate the apsides.")

    eccentricity_vector = (
        np.cross(velocity, angular_momentum) / EARTH_MU_M3_S2
        - position / radius_m
    )
    eccentricity = float(np.linalg.norm(eccentricity_vector))
    semi_major_axis_m = -EARTH_MU_M3_S2 / (2 * specific_energy)
    semi_latus_rectum_m = angular_momentum_squared / EARTH_MU_M3_S2

    perigee_radius_m = semi_latus_rectum_m / (1 + eccentricity)
    apogee_radius_m = 2 * semi_major_axis_m - perigee_radius_m
    return OrbitAltitudes(
        perigee_altitude_m=perigee_radius_m - EARTH_MEAN_RADIUS_M,
        apogee_altitude_m=apogee_radius_m - EARTH_MEAN_RADIUS_M,
    )


def compare_release_orbit(
    servicer_before: SpaceObject,
    recoil: ReleaseRecoil,
) -> OrbitAltitudeChange:
    """Compare the servicer's orbit at the release instant, before and after recoil."""
    if servicer_before.frame != "TEME":
        raise ValueError("Release orbit comparison expects the servicer state in TEME.")

    return OrbitAltitudeChange(
        before=calculate_orbit_altitudes(
            servicer_before.position, servicer_before.velocity
        ),
        after=calculate_orbit_altitudes(
            servicer_before.position, recoil.servicer_velocity_after_mps
        ),
    )


def calculate_recoil_compensation(recoil: ReleaseRecoil) -> RecoilCompensation:
    """Return the change in velocity that immediately cancels recoil."""
    recoil_vector = recoil.recoil_delta_v_mps
    if recoil_vector.shape != (3,) or not np.isfinite(recoil_vector).all():
        raise ValueError("Recoil delta-v must contain three finite values.")

    correction_delta_v_mps = -recoil_vector
    required_delta_v_mps = float(np.linalg.norm(correction_delta_v_mps))
    if not isfinite(required_delta_v_mps):
        raise ValueError("Required delta-v must be finite (m/s).")

    return RecoilCompensation(
        correction_delta_v_mps=correction_delta_v_mps,  # correction direction
        required_delta_v_mps=required_delta_v_mps,  # length of the vector
    )


def evaluate_postfire(
    servicer_before: SpaceObject,
    debris_before: SpaceObject,
    *,
    net_speed_mps: float,
    release_direction: np.ndarray,
    direction_frame: str,
    servicer_mass_after_release_kg: float,
    net_mass_kg: float,
) -> PostFireReport:
    """Evaluate a simulated release from two pre-release TEME states.

    Positions are in metres and velocities are in metres per second. The
    release direction is a TEME unit vector, and net speed is relative to the
    servicer immediately after release. Orbit changes describe the state
    immediately after recoil, before compensation.
    """
    recoil = calculate_release_recoil(
        servicer_before,
        debris_before,
        net_speed_mps=net_speed_mps,
        release_direction=release_direction,
        direction_frame=direction_frame,
        servicer_mass_after_release_kg=servicer_mass_after_release_kg,
        net_mass_kg=net_mass_kg,
    )
    return PostFireReport(
        recoil=recoil,
        orbit_change=compare_release_orbit(servicer_before, recoil),
        compensation=calculate_recoil_compensation(recoil),
    )
