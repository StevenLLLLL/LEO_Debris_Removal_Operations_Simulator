"""
B

All of B's code goes here
"""

# Lets type hints such as list[...], tuple[...] and dict[...] work on older Python versions
from __future__ import annotations

# warnings: report a non-fatal problem (e.g. an eccentric target orbit) without stopping the run
import warnings

# dataclass: auto-generates the constructors (SensorFaults, NavFrame, FiringSolution);
# field(default_factory=...): gives each object its own new array instead of all objects sharing one array
from dataclasses import dataclass, field
# datetime / timedelta: fire time = epoch (UTC) + seconds since start
from datetime import datetime, timedelta
# Optional[X]: marks a value that may be None (e.g. no sensor reading during a dropout, or no epoch given);
# TYPE_CHECKING: True only for type checkers, so the import below costs nothing at run time
from typing import TYPE_CHECKING, Optional

# numpy: all matrix and vector calculations
import numpy as np

# Earth gravitational parameter μ (m^3/s^2), needed to get the orbital rate n from Part A's SpaceObject
from config import MU_EARTH

# Part A's SpaceObject, used only in type hints; Part B does not depend on data_source at run time
if TYPE_CHECKING:
    from data_source import SpaceObject


# ===========================================================================
# Part B ④: proximity relative navigation
#
# Purpose: in the last 10 m, turn unreliable sensor readings into a reliable relative position (and velocity).
#
# Coordinate convention (shared by the whole team):
#     Local RSW frame with the target debris at the origin
#         x = R radial: up / down (positive away from Earth)
#         y = S along-track: forward / backward (positive in the direction of flight)
#         z = W cross-track: left / right (orbit normal)
#     Relative state = our satellite relative to the target, [x, y, z, vx, vy, vz]
#     Units: metres, seconds. Hold point = (0, -10, 0), i.e. on the V-bar 10 m directly behind the target.
#
# This module contains:
#     1. CW state transition matrix Φ(t) (analytic solution, no integration error)
#     2. Sensor simulation (noise, dropout, bad data; controlled by fault-injection parameters)
#     3. Kalman filter (CW as the motion model, with a bad-data gate)
#     4. Proximity main loop run_proximity(): outputs a sequence of NavFrame (interface 4)
# ===========================================================================

# ---------------------------------------------------------------------------
# Default parameters (can also be overridden by C's config.py and passed in)
# ---------------------------------------------------------------------------
HOLD_POINT = np.array([0.0, -10.0, 0.0])  # m, V-bar hold point: 10 m directly behind the target

DEFAULT_DT = 1.0              # s, length of each time step (one sensor reading per step)
DEFAULT_STEPS = 120           # number of steps (120 s): the longest we wait at the hold point, not a fixed fire time
MAX_DROPOUT_S = 10.0          # s, continuous dropout longer than this → abort (dead reckoning no longer trusted)
MAX_SIGMA_M = 1.0             # m, position uncertainty still above this after settling → abort (filter divergence)
SETTLE_TIME_S = 15.0          # s, filter settling time after start; no uncertainty check before this

# Threshold for "this reading is clearly wrong": chi-square, 3 DOF, 99% critical value.
# A squared Mahalanobis distance of the innovation above it → bad data
GATE_CHI2_3DOF_99 = 11.345


# ---------------------------------------------------------------------------
# 1. CW state transition matrix
# ---------------------------------------------------------------------------
# Future state after t seconds: returns the 6x6 matrix that maps the current state to the state dt seconds later
def cw_stm(n: float, dt: float) -> np.ndarray:
    """
    Analytic solution of the Clohessy-Wiltshire (CW) equations as a 6x6 state transition matrix Φ(dt).

    n:  target orbital angular rate (rad/s)
    dt: propagation time (s)
    Returns Φ(dt) (6x6, mixed units); new state = Φ(dt) @ old state, state order (x, y, z, vx, vy, vz) in m and m/s.
    Called by cw_propagate(), CWKalmanFilter.predict(), and section ⑤ (targeting and miss verification).

    Valid when the target orbit is near-circular and the relative distance is much smaller than the orbit radius
    (very accurate within a few km).
    """
    # n: debris angular rate around Earth (rad/s), about 0.00113 rad/s at ISS altitude
    # dt: how many seconds to predict forward (s)
    # nt: angle the debris travels around Earth during dt (rad)
    nt = n * dt
    # c, s: cosine and sine of that angle, reused many times when filling the matrix
    c = np.cos(nt)
    s = np.sin(nt)

    # Create a 6x6 matrix of zeros; only non-zero entries are filled below, the rest stay 0
    phi = np.zeros((6, 6))

    # phi[row, col] = how many times the current value of col contributes to the future value of row;
    # rows and columns are both ordered (x, y, z, vx, vy, vz)

    # ---- position rows ----
    # x row: future up/down position
    # x(t) = (4-3c)x0 + (s/n)vx0 + (2(1-c)/n)vy0
    phi[0, 0] = 4.0 - 3.0 * c
    phi[0, 3] = s / n
    phi[0, 4] = 2.0 * (1.0 - c) / n
    # y row: future forward/backward position. Directly behind with zero velocity (x0 = vx0 = vy0 = 0) only
    # phi[1, 1] = 1 acts, so y stays constant. The lone nt terms grow with time: a height difference drifts along-track
    # y(t) = 6(s-nt)x0 + y0 - (2(1-c)/n)vx0 + ((4s-3nt)/n)vy0
    phi[1, 0] = 6.0 * (s - nt)
    phi[1, 1] = 1.0
    phi[1, 3] = -2.0 * (1.0 - c) / n
    phi[1, 4] = (4.0 * s - 3.0 * nt) / n
    # z row: left/right is independent, swinging back and forth like a spring, unrelated to x and y
    # z(t) = c z0 + (s/n) vz0
    phi[2, 2] = c
    phi[2, 5] = s / n

    # ---- velocity rows: time derivative of the matching position row above ----
    phi[3, 0] = 3.0 * n * s
    phi[3, 3] = c
    phi[3, 4] = 2.0 * s
    phi[4, 0] = -6.0 * n * (1.0 - c)
    phi[4, 3] = -2.0 * s
    phi[4, 4] = 4.0 * c - 3.0
    phi[5, 2] = -n * s
    phi[5, 5] = c

    return phi


def cw_propagate(state: np.ndarray, n: float, dt: float) -> np.ndarray:
    """
    Push the current state forward by dt seconds using the Φ table.
    CW relative propagation; not to be confused with the SGP4 module propagate.py (Part A).

    state: relative state [x, y, z, vx, vy, vz] (m, m/s)
    n:     target orbital angular rate (rad/s)
    dt:    propagation time (s)
    Returns the state dt seconds later (m, m/s).
    Called by run_proximity() to move the true position each step, and by the tests.
    """
    return cw_stm(n, dt) @ state


def cw_derivative(state: np.ndarray, n: float) -> np.ndarray:
    """
    The raw CW differential equations, for tests only.

    state: relative state [x, y, z, vx, vy, vz] (m, m/s)
    n:     target orbital angular rate (rad/s)
    Returns the time derivative [vx, vy, vz, ax, ay, az] (m/s, m/s²).
    Called only by the tests: integrating it numerically (RK4) and comparing with Φ proves the table has no typo.
        ẍ = 3n²x + 2nẏ
        ÿ = -2nẋ
        z̈ = -n²z
    """
    x, y, z, vx, vy, vz = state
    # up/down: a height difference gets amplified, and forward speed pushes upward
    ax = 3.0 * n**2 * x + 2.0 * n * vy
    # forward/backward: moving upward gets pulled backward
    ay = -2.0 * n * vx
    # left/right: acts like a spring pulling back to the orbit plane
    az = -(n**2) * z
    return np.array([vx, vy, vz, ax, ay, az])


# ---------------------------------------------------------------------------
# 2. Sensor simulation
# ---------------------------------------------------------------------------
@dataclass
class SensorFaults:
    """
    Fault-injection parameters (the ones agreed in the interface spec).
    All defaults mean "no fault" (only normal sensor noise). A's UI buttons change these values,
    which are then passed to run_proximity().
    """
    noise_sigma: float = 0.05            # m, Gaussian noise std dev of normal readings (LiDAR, cm level); default 0.05
    dropout_rate: float = 0.0            # 0-1, probability of a random dropout at each step; default 0 (never)
    outlier_rate: float = 0.0            # 0-1, probability of bad data at each step; default 0 (never)
    outlier_magnitude: float = 3.0       # m, size of the bad-data offset; default 3.0 (only used if outlier_rate > 0)
    force_dropout_seconds: float = 0.0   # s, duration of the forced continuous dropout; default 0 (off)
    force_dropout_start: float = 30.0    # s, time at which the forced dropout starts; default 30


def simulate_sensor(true_pos: np.ndarray, t: float, faults: SensorFaults,
                    rng: np.random.Generator) -> tuple[Optional[np.ndarray], bool]:
    """
    Simulate one reading of the relative navigation sensor (position only, no velocity, like a real LiDAR / camera).

    We simulate the sensor because the program knows the true answer, so we can check how accurate the filter is.
    true_pos: true relative position (m)
    t:        time since start (s), used for the forced dropout window
    faults:   fault-injection parameters
    rng:      seeded random generator, so the same seed always gives the same readings (reproducible demo and tests)
    Returns (reading in m, or None during a dropout; True if bad data was injected this time).
    Called once per step by run_proximity().
    """
    # Check 1: dropout? Either inside the forced dropout window...
    in_forced = (faults.force_dropout_seconds > 0 and
                 faults.force_dropout_start <= t <
                 faults.force_dropout_start + faults.force_dropout_seconds)
    # ...or a random dropout this step; either way there is no reading
    if in_forced or rng.random() < faults.dropout_rate:
        return None, False

    # Check 2: normal reading = true position + Gaussian noise (m)
    reading = true_pos + rng.normal(0.0, faults.noise_sigma, size=3)

    # Check 3: turn it into bad data? Add a large offset in a random direction (e.g. a false reflection)
    if rng.random() < faults.outlier_rate:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        return reading + faults.outlier_magnitude * direction, True

    return reading, False


# ---------------------------------------------------------------------------
# 3. Kalman filter (CW motion model)
# ---------------------------------------------------------------------------
class CWKalmanFilter:
    """
    Kalman filter that combines the physics prediction (CW) and the sensor reading,
    weighting each by how trustworthy it is.

    State X = [x, y, z, vx, vy, vz] (m, m/s), covariance P (6x6, how uncertain the estimate is).
    Each step:
        predict(dt): X ← ΦX, P ← ΦPΦᵀ + Q        (done every step)
        update(z)  : only when a reading is available; must pass the bad-data gate first
    During a dropout only predict() runs, so the uncertainty keeps growing.
    Created and driven by run_proximity(); also used directly in the tests.
    """

    # Measurement matrix: picks the position out of the 6-D state, because the sensor observes position only
    H = np.hstack([np.eye(3), np.zeros((3, 3))])

    def __init__(self, n: float, x0: np.ndarray, meas_sigma: float,
                 p0_pos: float = 1.0, p0_vel: float = 0.1,
                 q_accel: float = 1e-6, gate: float = GATE_CHI2_3DOF_99):
        """
        n:          target orbital angular rate (rad/s)
        x0:         initial best guess [x, y, z, vx, vy, vz] (m, m/s)
        meas_sigma: sensor noise std dev (m)
        p0_pos, p0_vel: initial uncertainty of position (m) and velocity (m/s)
        q_accel:    process noise intensity (m²/s³), how imperfect the CW model is
        gate:       bad-data threshold on the squared Mahalanobis distance (default 11.345)
        """
        self.n = n
        # x: best guess of the relative state (m, m/s)
        self.x = np.asarray(x0, dtype=float).copy()
        # P: how uncertain that guess is; starts at p0_pos m in position and p0_vel m/s in velocity
        self.P = np.diag([p0_pos**2] * 3 + [p0_vel**2] * 3)
        # R: sensor noise covariance (m²)
        self.R = np.eye(3) * meas_sigma**2
        # q: how imperfect the model is (small unmodelled accelerations)
        self.q = q_accel
        self.gate = gate

    def _process_noise(self, dt: float) -> np.ndarray:
        """
        Process noise Q for one step of dt seconds, from a discrete white-noise acceleration model:
            Q = q * [[dt³/3 I, dt²/2 I],
                     [dt²/2 I, dt   I]]
        Represents disturbances not included in the CW model (perturbations, small thruster leaks, etc.).
        Returns a 6x6 matrix (m², m²/s, m²/s²). Called by predict().
        """
        i3 = np.eye(3)
        return self.q * np.block([[dt**3 / 3 * i3, dt**2 / 2 * i3],
                                  [dt**2 / 2 * i3, dt * i3]])

    def predict(self, dt: float) -> None:
        """
        Use the physics (CW) to push the guess forward by dt seconds (s); the uncertainty P grows.
        Returns nothing (updates self.x and self.P). Called every step by run_proximity().
        """
        phi = cw_stm(self.n, dt)
        self.x = phi @ self.x
        self.P = phi @ self.P @ phi.T + self._process_noise(dt)

    def update(self, z: np.ndarray) -> str:
        """
        Correct the guess with one position reading.

        z: sensor reading of the relative position (m)
        Returns "ok" (reading used) or "rejected" (flagged as bad data by the gate, guess unchanged).
        Called by run_proximity() whenever a reading is available.
        """
        H = self.H
        # Innovation: how far the reading is from where we predicted (m)
        nu = z - H @ self.x
        # S: how big the innovation is expected to be (prediction uncertainty + sensor noise, m²)
        S = H @ self.P @ H.T + self.R
        S_inv = np.linalg.inv(S)
        # Squared Mahalanobis distance: the innovation measured in units of its expected size
        d2 = float(nu @ S_inv @ nu)

        # Above 11.345, a reading this far off happens less than 1% of the time by noise alone → treat it as bad data
        if d2 > self.gate:
            return "rejected"                   # bad data: no update

        # Kalman gain K: the larger K, the more we trust the reading over the prediction
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ nu
        # Joseph form keeps P symmetric and positive definite despite rounding errors, so it is more stable
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return "ok"

    @property
    def pos(self) -> np.ndarray:
        """Estimated relative position (m), a copy so callers cannot change the filter state."""
        return self.x[:3].copy()

    @property
    def vel(self) -> np.ndarray:
        """Estimated relative velocity (m/s), a copy so callers cannot change the filter state."""
        return self.x[3:].copy()

    @property
    def sigma(self) -> float:
        """Total position uncertainty (m) = sqrt(trace of the position block of P); C uses it to decide Go / No-Go."""
        return float(np.sqrt(np.trace(self.P[:3, :3])))


# ---------------------------------------------------------------------------
# 4. Proximity main loop
# ---------------------------------------------------------------------------
@dataclass
class NavFrame:
    """Interface 4: one report per time step, produced by run_proximity() and read by C and the UI."""
    t: float                                  # s, time since start
    true_rel: np.ndarray                      # m, true relative position (for plotting and verification only)
    measured_rel: Optional[np.ndarray]        # m, sensor reading; None during dropout
    est_pos: np.ndarray                       # m, filter's estimated relative position
    est_vel: np.ndarray                       # m/s, filter's estimated relative velocity
    est_sigma: float                          # m, total position uncertainty of the estimate
    sensor_status: str                        # "ok" / "dropout" / "rejected"
    dropout_duration: float                   # s, how long there has been no valid reading so far
    abort: bool                               # True once an abort has been triggered (stays True)
    true_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))   # m/s, true relative velocity (extra field, for miss verification; C can ignore)
    abort_reason: str = ""                    # why the abort was triggered; empty if not aborted (extra field)


def run_proximity(n: float,
                  steps: int = DEFAULT_STEPS,
                  dt: float = DEFAULT_DT,
                  hold_point: np.ndarray = HOLD_POINT,
                  initial_pos_error: float = 0.05,
                  initial_vel_error: float = 0.002,
                  faults: Optional[SensorFaults] = None,
                  seed: int = 42,
                  max_dropout_s: float = MAX_DROPOUT_S,
                  max_sigma_m: float = MAX_SIGMA_M,
                  settle_time_s: float = SETTLE_TIME_S) -> list[NavFrame]:
    """
    Main simulation function for the proximity phase (public interface).

    n: target orbital angular rate (rad/s), e.g. from mean_motion_from_space_object()
    steps, dt: number of steps and step length (s); steps * dt is the longest wait at the hold point
    hold_point: hold point in RSW (m)
    initial_pos_error / initial_vel_error: true offset of the satellite when it arrives at the hold point (m, m/s),
        so the "true" motion is not perfectly stationary, closer to reality.
    faults: fault-injection parameters; None means no faults.
    seed: random seed. Fixed seed → same result every time (reproducible demo, tests can hard-code expected values).
    max_dropout_s (s), max_sigma_m (m), settle_time_s (s): abort thresholds, see below.
    Abort rules (once triggered, the abort stays on):
        ① continuous dropout longer than max_dropout_s seconds
        ② after the settling period settle_time_s, position uncertainty still above max_sigma_m
    Note: "is 3σ smaller than the net radius" is a pre-fire check, decided by C's Go / No-Go; it does not abort here.

    Returns a list of NavFrame, one report per step; element 0 is the state at t=0.
    C should scan it from the start and pick the first time that meets the firing conditions.
    Called by run_proximity_for_target(), firing.monte_carlo(), C's pipeline and the tests.
    """
    faults = faults or SensorFaults()
    rng = np.random.default_rng(seed)

    # ---- true state: hold point + small offset, because a real arrival is never perfect ----
    true_state = np.concatenate([
        hold_point + rng.normal(0.0, initial_pos_error, 3),
        rng.normal(0.0, initial_vel_error, 3),
    ])

    # ---- filter initial guess: exactly at the hold point with zero velocity,
    # because GNC reports "arrived at hold point" and that is all we know before the first reading ----
    kf = CWKalmanFilter(n, np.concatenate([hold_point, np.zeros(3)]),
                        meas_sigma=max(faults.noise_sigma, 1e-4))

    frames: list[NavFrame] = []
    # time of the last accepted reading (s), used to measure how long the dropout has lasted
    last_valid_t = 0.0
    aborted = False
    abort_reason = ""

    for k in range(steps + 1):
        t = k * dt

        # Step 1: move the true position and the filter forward by one step (skip at t = 0)
        if k > 0:
            true_state = cw_propagate(true_state, n, dt)   # true motion
            kf.predict(dt)                              # filter prediction

        # Step 2: take a sensor reading of the true position
        z, _injected = simulate_sensor(true_state[:3], t, faults, rng)

        # Step 3: no reading → record a dropout; otherwise update the filter (it may reject bad data)
        if z is None:
            status = "dropout"
        else:
            status = kf.update(z)
            if status == "ok":
                last_valid_t = t

        # Step 4: how long since the last valid reading (s)
        dropout_duration = t - last_valid_t

        # Step 5: abort check. Once aborted it stays aborted, so a lucky reading cannot make an unsafe run look safe
        if not aborted:
            # Rule ①: no valid reading for too long, the prediction alone can no longer be trusted
            if dropout_duration > max_dropout_s:
                aborted, abort_reason = True, f"No valid measurement for {dropout_duration:.0f} s"
            # Rule ②: uncertainty still too large after settling
            elif t >= settle_time_s and kf.sigma > max_sigma_m:
                # uncertainty is naturally large right after start, so give the filter settle_time_s seconds to converge;
                # still over the limit after that means the filter diverged or the sensor quality is too poor
                aborted, abort_reason = True, f"Position uncertainty {kf.sigma:.2f} m exceeds limit"

        # Step 6: record this step's report. .copy() everywhere: numpy arrays are references, so without a copy
        # later changes would also change the frames already stored
        frames.append(NavFrame(
            t=t,
            true_rel=true_state[:3].copy(),
            measured_rel=None if z is None else z.copy(),
            est_pos=kf.pos,
            est_vel=kf.vel,
            est_sigma=kf.sigma,
            sensor_status=status,
            dropout_duration=dropout_duration,
            abort=aborted,
            true_vel=true_state[3:].copy(),
            abort_reason=abort_reason,
        ))

    return frames


def mean_motion_from_space_object(space_object) -> float:
    """
    Orbital angular rate n of the target, computed from Part A's SpaceObject.

    space_object: Part A's SpaceObject; position in km, velocity in km/s (geocentric TEME frame).
    Returns n in rad/s.
    Called by run_proximity_for_target() (and by the tests).
    Raises ValueError if the input looks like metres or the orbit is not a plausible closed Earth orbit.
    Warns (UserWarning) if the eccentricity is above 0.03, because CW then becomes less accurate.
    """
    # Unit guard: an Earth orbit in km has |r| below about 1e5; larger means A already sent metres
    if np.linalg.norm(space_object.position) > 1e5:
        raise ValueError("SpaceObject position looks like metres; expected km "
                         "(Part A outputs km, Part B converts to m)")

    # Part A gives km and km/s; Part B converts to m and m/s here. Everything inside Part B uses SI units.
    r = np.asarray(space_object.position, dtype=float) * 1000.0   # km -> m
    v = np.asarray(space_object.velocity, dtype=float) * 1000.0   # km/s -> m/s
    r_norm = float(np.linalg.norm(r))
    v_norm = float(np.linalg.norm(v))

    # Sanity check: |r| must be between just above Earth's surface (6.5e6 m) and 5e7 m
    if not 6.5e6 <= r_norm <= 5e7:
        raise ValueError(f"Orbit radius {r_norm:.3e} m is outside the plausible range 6.5e6 to 5e7 m")

    # Vis-viva equation: a = 1 / (2/|r| - |v|²/μ); a non-positive denominator means a <= 0 (not a closed orbit)
    inv_a = 2.0 / r_norm - v_norm**2 / MU_EARTH
    if inv_a <= 0:
        raise ValueError(f"Semi-major axis is not positive (speed {v_norm:.1f} m/s is at or above "
                         f"escape speed); not a closed orbit")
    a = 1.0 / inv_a

    # Eccentricity vector e_vec = ((|v|² - μ/|r|) r - (r·v) v) / μ; e = |e_vec| (unitless)
    e_vec = ((v_norm**2 - MU_EARTH / r_norm) * r - float(np.dot(r, v)) * v) / MU_EARTH
    e = float(np.linalg.norm(e_vec))
    # CW assumes a circular target orbit; above e = 0.03 results are less accurate, so warn but keep running
    if e > 0.03:
        warnings.warn(f"Target eccentricity {e:.3f} > 0.03; CW model assumes a near-circular orbit, "
                      f"accuracy reduced")

    # Mean motion: n = sqrt(μ / a³), rad/s
    return float(np.sqrt(MU_EARTH / a**3))


def run_proximity_for_target(space_object, **kwargs) -> list[NavFrame]:
    """
    Entry point for C: take Part A's SpaceObject, compute n from it, then call run_proximity().

    space_object: Part A's SpaceObject (position in km, velocity in km/s)
    **kwargs:     any other run_proximity() argument (steps, faults, seed, ...)
    Returns the same list of NavFrame as run_proximity() (SI units).
    Called by C's pipeline.
    """
    return run_proximity(mean_motion_from_space_object(space_object), **kwargs)


# ===========================================================================
# Part B ⑤: firing solution + miss verification
#
# Core idea:
#     Once the net leaves the satellite it is a free-flying object, so in the RSW frame centred on the target
#     the net obeys the CW equations just like our satellite. The target is always at the origin.
#
#     The problem becomes: starting from the current position r0, the net must reach the origin exactly
#     T seconds later; what should its initial velocity v0 (relative to the target) be?
#
#     Split Φ(T) into 3x3 blocks:
#         r(T) = Φ_rr(T) r0 + Φ_rv(T) v0 = 0
#     ⇒  v0 = -Φ_rv(T)⁻¹ Φ_rr(T) r0
#
#     The net inherits the satellite's own relative velocity v_sat, so the launcher must add:
#         v_net = v0 - v_sat
#
#     The net speed |v_net| must equal a given value, so we solve a 1-D root-finding problem in T (bisection).
#
# All inputs and outputs are in SI units (m, m/s, s, rad); fire_time is a UTC datetime.
# ===========================================================================

# ---- Net and debris parameters (assumed values; to be moved to config.py once the team agrees) ----
# Net mass (kg), assumed
NET_MASS_KG = 2.0
# Net radius when fully deployed (m), assumed; a capture needs the net to cover the debris
NET_RADIUS_M = 2.0
# Time for the net to open fully after launch (s), assumed; a shorter flight hits with a half-open net
NET_DEPLOY_TIME_S = 0.3
# Kinetic energy the launcher gives the net (J), assumed; together with the mass it sets the net speed
LAUNCHER_ENERGY_J = 400.0
# Debris radius (m), assumed, because CelesTrak orbit data has no object size
DEBRIS_RADIUS_M = 0.5


def net_speed_from_energy(mass_kg: float, energy_j: float) -> float:
    """
    Net launch speed from the launcher energy: E = ½ m v²  ⇒  v = sqrt(2E / m).

    mass_kg:  net mass (kg), must be > 0
    energy_j: kinetic energy given by the launcher (J), must be > 0
    Returns the net speed relative to the satellite (m/s).
    Called at import time to set DEFAULT_NET_SPEED, and by the tests.
    """
    if mass_kg <= 0 or energy_j <= 0:
        raise ValueError(f"Net mass and launcher energy must be positive (got {mass_kg} kg, {energy_j} J)")
    # A heavier net flies slower for the same energy. Mass does not change the aim, because in vacuum
    # gravity gives every object the same acceleration, so the CW motion of the net does not depend on its mass
    return float(np.sqrt(2.0 * energy_j / mass_kg))


DEFAULT_NET_SPEED = net_speed_from_energy(NET_MASS_KG, LAUNCHER_ENERGY_J)   # m/s, net launch speed (20 m/s)
T_MIN, T_MAX = 1e-3, 5.0     # s, search range for the net's time of flight (no solution if it needs longer)


@dataclass
class FiringSolution:
    """Interface 5: the firing solution, produced by solve_firing() and read by C and the UI."""
    fire_time: Optional[datetime]   # UTC fire time (epoch + frame.t); None if no epoch was given
    direction_rsw: np.ndarray       # firing direction unit vector (RSW, unitless)
    azimuth: float                  # rad, measured from along-track, positive towards cross-track
    elevation: float                # rad, positive towards radial (away from Earth)
    net_speed: float                # m/s, net muzzle speed relative to the satellite
    time_of_flight: float           # s, time for the net to reach the target
    predicted_miss: float           # m, miss distance from the simulation's true state (not available in a real mission)
    converged: bool                 # True if a time of flight was found
    net_velocity_rsw: Optional[np.ndarray] = None   # m/s, net velocity relative to the satellite (extra field)
    miss_uncertainty: float = 0.0         # m, 3σ position uncertainty (extra field); real Go / No-Go must use this
    message: str = ""                     # explanation when there is no solution or the net would not open (extra field)
    net_mass: float = 0.0                 # kg, net mass used for this solution
    net_momentum_rsw: Optional[np.ndarray] = None   # N*s, net momentum = mass x velocity relative to satellite; C uses it for recoil: recoil dv = -momentum / satellite mass
    capture_margin: float = float("nan")  # m, = NET_RADIUS_M - DEBRIS_RADIUS_M - predicted_miss; > 0 means the net covers the debris
    deploy_ok: bool = False               # True if time_of_flight >= NET_DEPLOY_TIME_S (net fully open on arrival)


# ---------------------------------------------------------------------------
# CW targeting
# ---------------------------------------------------------------------------
def cw_targeting_velocity(r0: np.ndarray, n: float, T: float) -> np.ndarray:
    """
    Relative velocity needed to start from r0 and reach the target (origin) exactly T seconds later.

    r0: start position relative to the target (m)
    n:  target orbital angular rate (rad/s)
    T:  time of flight (s)
    Returns v0, the required velocity relative to the target (m/s).
    Called by _speed_error() and solve_firing().
    """
    # Split Φ(T) into four 3x3 blocks; only the top two matter because we only need the final position:
    # Φ_rr = how the start position moves the end position, Φ_rv = how the start velocity moves it
    phi = cw_stm(n, T)
    phi_rr = phi[:3, :3]
    phi_rv = phi[:3, 3:]
    # End position Φ_rr r0 + Φ_rv v0 = 0  ⇒  v0 = -Φ_rv⁻¹ Φ_rr r0.
    # solve() gives the same result as an explicit inverse but is faster and more accurate
    return -np.linalg.solve(phi_rv, phi_rr @ r0)


def _speed_error(T: float, r0, v_sat, n, net_speed) -> float:
    """
    How far the required launch speed for time of flight T is from the launcher's speed.

    T: time of flight (s); r0: start position (m); v_sat: satellite relative velocity (m/s);
    n: orbital rate (rad/s); net_speed: launcher speed (m/s)
    Returns f(T) = |v0(T) - v_sat| - net_speed (m/s). Smaller T needs a larger speed, so f decreases monotonically.
    Called by solve_time_of_flight().
    """
    return float(np.linalg.norm(cw_targeting_velocity(r0, n, T) - v_sat) - net_speed)


def solve_time_of_flight(r0, v_sat, n, net_speed,
                         t_min=T_MIN, t_max=T_MAX, tol=1e-9, max_iter=100):
    """
    Find the time of flight T for which the required launch speed equals the launcher's net_speed.

    We must search for T because the launcher has a fixed speed: any T gives a valid aim, but only one T
    matches that speed. f(T) decreases monotonically, so bisection always finds it if it exists.
    r0: start position (m); v_sat: satellite relative velocity (m/s); n: orbital rate (rad/s);
    net_speed: launcher speed (m/s); t_min, t_max: search range (s); tol: stop tolerance (s); max_iter: iteration cap
    Returns (T in s, converged). converged = False if no root exists in [t_min, t_max],
    i.e. even t_max is too short for this net speed (net too slow or target too far).
    Called by solve_firing() and the tests.
    """
    f_lo = _speed_error(t_min, r0, v_sat, n, net_speed)
    f_hi = _speed_error(t_max, r0, v_sat, n, net_speed)
    # A root exists only if f changes sign: f(t_min) >= 0 (too fast) and f(t_max) <= 0 (slow enough)
    if f_lo < 0 or f_hi > 0:
        return float("nan"), False   # no root in the interval

    lo, hi = t_min, t_max
    # Each step halves the interval [lo, hi] that contains the root
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if _speed_error(mid, r0, v_sat, n, net_speed) > 0:
            lo = mid          # speed still too high → need a longer time of flight
        else:
            hi = mid          # speed already low enough → the root is at or before mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi), True


# ---------------------------------------------------------------------------
# Miss verification
# ---------------------------------------------------------------------------
def miss_distance(sat_pos: np.ndarray, sat_vel: np.ndarray, v_net: np.ndarray,
                  n: float, T: float, samples: int = 300) -> float:
    """
    Closest distance between the net and the target (origin) after launch.

    The net starts at sat_pos with velocity sat_vel + v_net and moves under CW up to 1.5T.
    solve_firing() passes the TRUE state here while the aim used the ESTIMATED state,
    so the result is the real miss caused by the estimation error.
    sat_pos (m), sat_vel (m/s): satellite relative position and velocity; v_net (m/s): net velocity relative
    to the satellite; n (rad/s): orbital rate; T (s): planned time of flight; samples: number of coarse samples
    Returns the minimum distance (m).
    Called by solve_firing() and the tests.
    """
    # Net state right after launch: it inherits the satellite's velocity plus the launch velocity
    net_state = np.concatenate([sat_pos, sat_vel + v_net])

    # Distance from the net to the target t seconds after launch (m)
    def dist(t: float) -> float:
        return float(np.linalg.norm((cw_stm(n, t) @ net_state)[:3]))

    # Step 1: coarse sampling finds roughly where the closest approach is (avoids missing it with local search alone)
    times = np.linspace(0.0, 1.5 * T, samples)
    dists = [dist(t) for t in times]
    i = int(np.argmin(dists))
    lo = times[max(i - 1, 0)]
    hi = times[min(i + 1, samples - 1)]

    # Step 2: ternary search refines it inside that small interval, where the distance has a single minimum
    for _ in range(60):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if dist(m1) < dist(m2):
            hi = m2
        else:
            lo = m1
    return min(dist(0.5 * (lo + hi)), min(dists))


# ---------------------------------------------------------------------------
# Public main function
# ---------------------------------------------------------------------------
def direction_to_angles(d: np.ndarray) -> tuple[float, float]:
    """
    Convert a firing direction into pointing angles.

    d: firing direction unit vector in RSW (unitless)
    Returns (azimuth, elevation) in rad:
        azimuth   = angle in the S-W plane, measured from along-track (S), positive towards cross-track (W)
        elevation = angle above the S-W plane, positive towards radial (R, away from Earth)
    Called by solve_firing().
    """
    # elevation from the radial component (clip guards against rounding just above 1)
    elevation = float(np.arcsin(np.clip(d[0], -1.0, 1.0)))
    # azimuth from the along-track and cross-track components, turning from S towards W
    azimuth = float(np.arctan2(d[2], d[1]))
    return azimuth, elevation


def solve_firing(frame: NavFrame, n: float,
                 epoch: Optional[datetime] = None,
                 net_speed: float = DEFAULT_NET_SPEED,
                 net_mass: float = NET_MASS_KG,
                 net_radius_m: float = NET_RADIUS_M,
                 debris_radius_m: float = DEBRIS_RADIUS_M,
                 deploy_time_s: float = NET_DEPLOY_TIME_S) -> FiringSolution:
    """
    Compute the firing solution from one navigation frame (public interface).

    frame:     the NavFrame to fire from (usually the first one that passes C's Go / No-Go)
    n:         target orbital angular rate (rad/s)
    epoch:     UTC start time of the proximity run; can be Part A's SpaceObject.time. fire time = epoch + frame.t
    net_speed: launcher speed (m/s)
    net_mass:  net mass (kg); only used for the momentum, it does not change the aim
    net_radius_m, debris_radius_m: net and debris radii (m), used for capture_margin
    deploy_time_s: time the net needs to open fully (s), used for deploy_ok
    Returns a FiringSolution (SI units). Aiming uses the estimated state (est_pos, est_vel);
    verification uses the true state (true_rel, true_vel).
    Called by C's pipeline, monte_carlo() and the tests.
    """
    # Step 1: aim from the filter's estimate (the only thing a real satellite knows)
    r0 = frame.est_pos
    v_sat = frame.est_vel
    fire_time = epoch + timedelta(seconds=frame.t) if epoch else None

    # Step 2: find the time of flight that matches the launcher speed; report "no solution" if there is none
    T, ok = solve_time_of_flight(r0, v_sat, n, net_speed)
    if not ok:
        return FiringSolution(
            fire_time=fire_time, direction_rsw=np.zeros(3), azimuth=float("nan"),
            elevation=float("nan"), net_speed=net_speed, time_of_flight=float("nan"),
            predicted_miss=float("inf"), converged=False,
            net_velocity_rsw=np.zeros(3), miss_uncertainty=3 * frame.est_sigma,
            message="No solution within the time-of-flight search range (net speed too low or distance too large)",
            net_mass=net_mass, net_momentum_rsw=np.zeros(3),
        )

    # Step 3: velocity needed relative to the target, minus the satellite's own velocity,
    # because the net inherits the satellite's velocity and the launcher only adds the difference
    v0 = cw_targeting_velocity(r0, n, T)
    v_net = v0 - v_sat
    # Step 4: firing direction as a unit vector and as pointing angles
    d = v_net / np.linalg.norm(v_net)
    az, el = direction_to_angles(d)

    # Step 5: check the shot against the true state to get the real miss distance (simulation only)
    miss = miss_distance(frame.true_rel, frame.true_vel, v_net, n, T)

    # Step 6: net size and mass checks
    # Momentum the net carries away (N*s); the satellite gets the opposite, which C turns into the recoil dv
    net_momentum = net_mass * v_net
    # Capture margin (m): how much room is left after the miss; > 0 means the net still covers the debris
    capture_margin = net_radius_m - debris_radius_m - miss
    # The net must be fully open on arrival, so the flight must last at least the deploy time
    deploy_ok = T >= deploy_time_s
    message = "" if deploy_ok else (
        f"Time of flight {T:.2f} s is shorter than net deploy time {deploy_time_s} s; net would not be fully open. "
        f"Increase stand-off distance or lower net speed.")

    return FiringSolution(
        fire_time=fire_time,
        direction_rsw=d,
        azimuth=az,
        elevation=el,
        net_speed=float(np.linalg.norm(v_net)),
        time_of_flight=T,
        predicted_miss=miss,
        converged=True,
        net_velocity_rsw=v_net,
        miss_uncertainty=3 * frame.est_sigma,
        message=message,
        net_mass=net_mass,
        net_momentum_rsw=net_momentum,
        capture_margin=capture_margin,
        deploy_ok=deploy_ok,
    )


# ---------------------------------------------------------------------------
# Choosing when to fire, and the single entry point for Part A's output
# ---------------------------------------------------------------------------
def first_firing_opportunity(frames: list[NavFrame], n: float,
                             epoch: Optional[datetime] = None,
                             net_speed: float = DEFAULT_NET_SPEED) -> tuple[Optional[int], Optional[FiringSolution]]:
    """
    Scan the navigation frames from the start and return the first one that is safe to fire from.

    frames:    NavFrame list from run_proximity() (one per second)
    n:         target orbital angular rate (rad/s)
    epoch:     UTC start time of the run (Part A's SpaceObject.time); fire time = epoch + frame.t
    net_speed: launcher speed (m/s)
    A frame qualifies when: not aborted, converged, deploy_ok, and NET_RADIUS_M - DEBRIS_RADIUS_M - miss_uncertainty > 0.
    miss_uncertainty (3σ) is used instead of predicted_miss, because predicted_miss needs the simulation's
    true state, which a real satellite does not have.
    Returns (frame index, FiringSolution), or (None, None) if an abort happens first or no frame qualifies.
    Called by run_part_b().
    """
    for i, frame in enumerate(frames):
        # Once the run is aborted it stays aborted (see run_proximity), so stop scanning
        if frame.abort:
            return None, None
        sol = solve_firing(frame, n, epoch=epoch, net_speed=net_speed)
        # Real-mission capture check: the 3σ position error must still fit inside the net around the debris
        if sol.converged and sol.deploy_ok and NET_RADIUS_M - DEBRIS_RADIUS_M - sol.miss_uncertainty > 0:
            return i, sol
    return None, None


@dataclass
class PartBResult:
    """Everything Part B produces for one target, returned by run_part_b() and read by C and the UI."""
    norad_id: int                             # NORAD catalogue number, copied from Part A's SpaceObject
    name: str                                 # object name, copied from Part A's SpaceObject
    epoch: datetime                           # UTC start time of the run (Part A's SpaceObject.time)
    n: float                                  # rad/s, target orbital angular rate derived from Part A's state
    frames: list[NavFrame]                    # one NavFrame per second of the proximity phase (SI units)
    fire_index: Optional[int]                 # index in frames of the first firing opportunity; None if none
    solution: Optional[FiringSolution]        # firing solution at fire_index; None if no opportunity
    abort_reason: str                         # why there is no firing opportunity; empty if one was found


def run_part_b(space_object: SpaceObject, faults: Optional[SensorFaults] = None,
               seed: int = 42) -> PartBResult:
    """
    Part B's single public entry point: from Part A's target state to a firing decision input for C.

    space_object: Part A's SpaceObject, e.g. from app.calculate_state(); position in km, velocity in km/s,
                  frame "TEME", time in UTC. Part B converts to m and m/s at this entry point.
    faults:       sensor fault-injection parameters; None means no faults
    seed:         random seed for a reproducible run
    Returns a PartBResult; all its numbers are in SI units (m, m/s, s, rad), fire_time is UTC.
    Raises ValueError if the frame is not TEME or the state is not a plausible closed Earth orbit.
    Called by C's pipeline.
    """
    # The vis-viva step needs a geocentric inertial state; TEME is what Part A's SGP4 output uses
    if space_object.frame != "TEME":
        raise ValueError(f"Part B needs a TEME state from Part A; got frame '{space_object.frame}'")
    # space_object.orientation is not used: the relative navigation only needs position and velocity

    # Step 1: orbital rate n (rad/s) from Part A's km / km/s state (unit conversion happens inside)
    n = mean_motion_from_space_object(space_object)
    # Step 2: simulate the proximity phase at the hold point
    frames = run_proximity_for_target(space_object, faults=faults, seed=seed)
    # Step 3: first frame that is safe to fire from; fire time is measured from Part A's state time
    fire_index, solution = first_firing_opportunity(frames, n, epoch=space_object.time)

    if fire_index is not None:
        abort_reason = ""
    else:
        abort_reason = frames[-1].abort_reason or "No frame met the firing criteria within the time window"

    return PartBResult(
        norad_id=space_object.norad_id,
        name=space_object.name,
        epoch=space_object.time,
        n=n,
        frames=frames,
        fire_index=fire_index,
        solution=solution,
        abort_reason=abort_reason,
    )


# ---------------------------------------------------------------------------
# Monte Carlo: noise → capture success rate
# ---------------------------------------------------------------------------
def monte_carlo(n: float, noise_sigmas, runs: int = 200, steps: int = 60,
                net_speed: float = DEFAULT_NET_SPEED,
                net_radius_m: float = NET_RADIUS_M,
                debris_radius_m: float = DEBRIS_RADIUS_M,
                enforce_abort: bool = True) -> dict[float, dict]:
    """
    Monte Carlo study: how sensor noise affects the capture success rate.

    n: orbital rate (rad/s); noise_sigmas: sensor noise levels to test (m); runs: runs per noise level
    (a different random seed each time); steps: steps per run (1 s each); net_speed (m/s);
    net_radius_m, debris_radius_m: net and debris radii (m);
    enforce_abort: True = respect the abort rules; False = ignore them and always fire,
        used to plot the "pure physics" miss curve.
    Returns {noise σ: {"hit_rate", "abort_rate", "mean_miss"}}:
        hit_rate   —— fraction of runs that fired and captured (capture_margin > 0 and the net fully deployed)
        abort_rate —— fraction of runs where the system judged it unsafe and aborted
        mean_miss  —— mean miss distance (m) over the runs that actually fired
    Called by the UI / analysis scripts (not used in the main pipeline).
    """
    results = {}
    for sigma in noise_sigmas:
        hits = aborts = 0
        misses = []
        for seed in range(runs):
            frames = run_proximity(n, steps=steps, seed=seed,
                                   faults=SensorFaults(noise_sigma=sigma))
            # fire from the last frame of each run
            last = frames[-1]
            if last.abort and enforce_abort:
                aborts += 1
                continue
            sol = solve_firing(last, n, net_speed=net_speed,
                               net_radius_m=net_radius_m, debris_radius_m=debris_radius_m)
            if not sol.converged:
                continue
            misses.append(sol.predicted_miss)
            # A capture needs the net to cover the debris AND to be fully open on arrival
            if sol.capture_margin > 0 and sol.deploy_ok:
                hits += 1
        results[float(sigma)] = {
            "hit_rate": hits / runs,
            "abort_rate": aborts / runs,
            "mean_miss": float(np.mean(misses)) if misses else float("nan"),
        }
    return results
