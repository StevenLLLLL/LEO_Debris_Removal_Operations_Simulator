"""
relnav.py  —  Part B ④: proximity relative navigation

Coordinate convention (shared by the whole team):
    Local RSW frame with the target debris at the origin
        x = R radial (positive away from Earth)
        y = S along-track (positive in the direction of flight)
        z = W cross-track (orbit normal)
    Relative state = our satellite relative to the target, [x, y, z, vx, vy, vz]
    Units: metres, seconds. Hold point = (0, -10, 0), i.e. on the V-bar 10 m directly behind the target.

This module contains:
    1. CW state transition matrix Φ(t) (analytic solution, no integration error)
    2. Sensor simulation (noise, dropout, bad data; controlled by fault-injection parameters)
    3. Kalman filter (CW as the motion model, with a bad-data gate)
    4. Proximity main loop run_proximity(): outputs a sequence of NavFrame (interface 4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Default parameters (can also be overridden by C's config.py and passed in)
# ---------------------------------------------------------------------------
HOLD_POINT = np.array([0.0, -10.0, 0.0])  # m, V-bar hold point

DEFAULT_DT = 1.0              # s, length of each time step
DEFAULT_STEPS = 120           # number of steps (120 s)
MAX_DROPOUT_S = 10.0          # s, continuous dropout longer than this → abort
MAX_SIGMA_M = 1.0             # m, position uncertainty still above this after settling → abort (filter divergence)
SETTLE_TIME_S = 15.0          # s, filter settling time after start; no uncertainty check before this

# Chi-square critical value, 3 DOF, 99%. Squared Mahalanobis distance of the innovation above this → bad data
GATE_CHI2_3DOF_99 = 11.345


# ---------------------------------------------------------------------------
# 1. CW state transition matrix
# ---------------------------------------------------------------------------
def cw_stm(n: float, dt: float) -> np.ndarray:
    """
    Analytic solution of the Clohessy-Wiltshire equations, written as a 6x6 state transition matrix Φ(dt).
    new state = Φ(dt) @ old state.
    n: target orbital angular rate (rad/s), dt: propagation time (s).

    Valid when: the target orbit is near-circular and the relative distance is much smaller than the orbit radius (very accurate within a few km).
    """
    nt = n * dt
    c = np.cos(nt)
    s = np.sin(nt)

    phi = np.zeros((6, 6))

    # ---- position rows ----
    # x(t) = (4-3c)x0 + (s/n)vx0 + (2(1-c)/n)vy0
    phi[0, 0] = 4.0 - 3.0 * c
    phi[0, 3] = s / n
    phi[0, 4] = 2.0 * (1.0 - c) / n
    # y(t) = 6(s-nt)x0 + y0 - (2(1-c)/n)vx0 + ((4s-3nt)/n)vy0
    phi[1, 0] = 6.0 * (s - nt)
    phi[1, 1] = 1.0
    phi[1, 3] = -2.0 * (1.0 - c) / n
    phi[1, 4] = (4.0 * s - 3.0 * nt) / n
    # z(t) = c z0 + (s/n) vz0      (cross-track is an independent harmonic oscillation)
    phi[2, 2] = c
    phi[2, 5] = s / n

    # ---- velocity rows (time derivative of the position rows) ----
    phi[3, 0] = 3.0 * n * s
    phi[3, 3] = c
    phi[3, 4] = 2.0 * s
    phi[4, 0] = -6.0 * n * (1.0 - c)
    phi[4, 3] = -2.0 * s
    phi[4, 4] = 4.0 * c - 3.0
    phi[5, 2] = -n * s
    phi[5, 5] = c

    return phi


def propagate(state: np.ndarray, n: float, dt: float) -> np.ndarray:
    """Propagate the 6-D relative state forward by dt seconds using Φ."""
    return cw_stm(n, dt) @ state


def cw_derivative(state: np.ndarray, n: float) -> np.ndarray:
    """
    The CW differential equations themselves (test use only: compared against numerical integration to verify Φ).
        ẍ = 3n²x + 2nẏ
        ÿ = -2nẋ
        z̈ = -n²z
    """
    x, y, z, vx, vy, vz = state
    ax = 3.0 * n**2 * x + 2.0 * n * vy
    ay = -2.0 * n * vx
    az = -(n**2) * z
    return np.array([vx, vy, vz, ax, ay, az])


# ---------------------------------------------------------------------------
# 2. Sensor simulation
# ---------------------------------------------------------------------------
@dataclass
class SensorFaults:
    """
    Fault-injection parameters (the ones agreed in the interface spec). Defaults = no faults.
    A's UI buttons change these values, which are then passed to run_proximity().
    """
    noise_sigma: float = 0.05            # m, Gaussian noise std dev of normal readings (LiDAR, cm level)
    dropout_rate: float = 0.0            # probability of a random dropout at each step
    outlier_rate: float = 0.0            # probability of bad data at each step
    outlier_magnitude: float = 3.0       # m, size of the bad-data offset
    force_dropout_seconds: float = 0.0   # duration of the forced continuous dropout
    force_dropout_start: float = 30.0    # time (s) at which the forced dropout starts


def simulate_sensor(true_pos: np.ndarray, t: float, faults: SensorFaults,
                    rng: np.random.Generator) -> tuple[Optional[np.ndarray], bool]:
    """
    Simulate the relative navigation sensor (measures position only, not velocity, like a real LiDAR / camera).
    Returns (reading or None, whether bad data was injected this time).
    """
    # forced dropout window
    in_forced = (faults.force_dropout_seconds > 0 and
                 faults.force_dropout_start <= t <
                 faults.force_dropout_start + faults.force_dropout_seconds)
    # random dropout
    if in_forced or rng.random() < faults.dropout_rate:
        return None, False

    reading = true_pos + rng.normal(0.0, faults.noise_sigma, size=3)

    # bad data: add a large offset in a random direction (e.g. a false reflection)
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
    State X = [x, y, z, vx, vy, vz], covariance P (6x6, how uncertain the estimate is).

    Each step:
        predict(dt): X ← ΦX, P ← ΦPΦᵀ + Q        (done every step)
        update(z)  : only when a reading is available; must pass the bad-data gate first
    """

    H = np.hstack([np.eye(3), np.zeros((3, 3))])  # observe position only

    def __init__(self, n: float, x0: np.ndarray, meas_sigma: float,
                 p0_pos: float = 1.0, p0_vel: float = 0.1,
                 q_accel: float = 1e-6, gate: float = GATE_CHI2_3DOF_99):
        self.n = n
        self.x = np.asarray(x0, dtype=float).copy()
        # initial covariance: uncertainty of p0_pos m in position and p0_vel m/s in velocity
        self.P = np.diag([p0_pos**2] * 3 + [p0_vel**2] * 3)
        self.R = np.eye(3) * meas_sigma**2       # sensor noise covariance
        self.q = q_accel                          # process noise intensity (small unmodelled accelerations)
        self.gate = gate

    def _process_noise(self, dt: float) -> np.ndarray:
        """
        Discrete white-noise acceleration model:
            Q = q * [[dt³/3 I, dt²/2 I],
                     [dt²/2 I, dt   I]]
        Represents disturbances not included in the CW model (perturbations, small thruster leaks, etc.).
        """
        i3 = np.eye(3)
        return self.q * np.block([[dt**3 / 3 * i3, dt**2 / 2 * i3],
                                  [dt**2 / 2 * i3, dt * i3]])

    def predict(self, dt: float) -> None:
        phi = cw_stm(self.n, dt)
        self.x = phi @ self.x
        self.P = phi @ self.P @ phi.T + self._process_noise(dt)

    def update(self, z: np.ndarray) -> str:
        """
        Update with one position reading. Returns "ok" or "rejected" (flagged as bad data by the gate).
        """
        H = self.H
        nu = z - H @ self.x                     # innovation: reading minus prediction
        S = H @ self.P @ H.T + self.R           # innovation covariance
        S_inv = np.linalg.inv(S)
        d2 = float(nu @ S_inv @ nu)             # squared Mahalanobis distance

        if d2 > self.gate:
            return "rejected"                   # bad data: no update

        K = self.P @ H.T @ S_inv                # Kalman gain
        self.x = self.x + K @ nu
        # Joseph-form covariance update, numerically more stable (keeps P symmetric positive definite)
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return "ok"

    @property
    def pos(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def vel(self) -> np.ndarray:
        return self.x[3:].copy()

    @property
    def sigma(self) -> float:
        """Total position uncertainty (m) = sqrt(trace of the position block of P)."""
        return float(np.sqrt(np.trace(self.P[:3, :3])))


# ---------------------------------------------------------------------------
# 4. Proximity main loop
# ---------------------------------------------------------------------------
@dataclass
class NavFrame:
    """Interface 4: one per time step."""
    t: float                                  # s, time since start
    true_rel: np.ndarray                      # m, true relative position (for plotting and verification only)
    measured_rel: Optional[np.ndarray]        # m, sensor reading; None during dropout
    est_pos: np.ndarray                       # m, estimated position
    est_vel: np.ndarray                       # m/s, estimated velocity
    est_sigma: float                          # m, estimate uncertainty
    sensor_status: str                        # "ok" / "dropout" / "rejected"
    dropout_duration: float                   # s, how long there has been no valid reading so far
    abort: bool                               # whether an abort has been triggered
    true_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # ↑ extra field beyond the interface: miss verification needs the true velocity. C can ignore it.
    abort_reason: str = ""


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

    n: target orbital angular rate (from A's TargetState.n)
    initial_pos_error / initial_vel_error: true offset of the satellite when it arrives at the hold point (m, m/s),
        so the "true" motion is not perfectly stationary, closer to reality.
    faults: fault-injection parameters; None means no faults.
    seed: random seed. Fixed seed → same result every time (reproducible demo, tests can hard-code expected values).
    Abort rules (once triggered, the abort stays on):
        ① continuous dropout longer than max_dropout_s seconds
        ② after the settling period settle_time_s, position uncertainty still above max_sigma_m
    Note: "is 3σ smaller than the net radius" is a pre-fire check, decided by C's Go / No-Go; it does not abort here.

    Returns a list of NavFrame; element 0 is the state at t=0.
    """
    faults = faults or SensorFaults()
    rng = np.random.default_rng(seed)

    # ---- true state: hold point + small offset ----
    true_state = np.concatenate([
        hold_point + rng.normal(0.0, initial_pos_error, 3),
        rng.normal(0.0, initial_vel_error, 3),
    ])

    # ---- filter initial value: GNC reports "arrived at hold point", so prior = hold point, zero velocity ----
    kf = CWKalmanFilter(n, np.concatenate([hold_point, np.zeros(3)]),
                        meas_sigma=max(faults.noise_sigma, 1e-4))

    frames: list[NavFrame] = []
    last_valid_t = 0.0
    aborted = False
    abort_reason = ""

    for k in range(steps + 1):
        t = k * dt

        if k > 0:
            true_state = propagate(true_state, n, dt)   # true motion
            kf.predict(dt)                              # filter prediction

        z, _injected = simulate_sensor(true_state[:3], t, faults, rng)

        if z is None:
            status = "dropout"
        else:
            status = kf.update(z)
            if status == "ok":
                last_valid_t = t

        dropout_duration = t - last_valid_t

        # ---- abort check (once triggered, stays aborted) ----
        if not aborted:
            if dropout_duration > max_dropout_s:
                aborted, abort_reason = True, f"No valid measurement for {dropout_duration:.0f} s"
            elif t >= settle_time_s and kf.sigma > max_sigma_m:
                # uncertainty is naturally large right after start, so give the filter settle_time_s seconds to converge;
                # still over the limit after that means the filter diverged or the sensor quality is too poor
                aborted, abort_reason = True, f"Position uncertainty {kf.sigma:.2f} m exceeds limit"

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


def run_proximity_for_target(target_state, **kwargs) -> list[NavFrame]:
    """Convenience wrapper for C: pass A's TargetState directly (only its n is used)."""
    return run_proximity(target_state.n, **kwargs)
