"""
Location_of_us.py  —  Part B ④: proximity relative navigation

Purpose: in the last 10 m, turn unreliable sensor readings into a reliable relative position (and velocity).

Coordinate convention (shared by the whole team):
    Local RSW frame with the target debris at the origin
        x = R radial: up / down (positive away from Earth)
        y = S along-track: forward / backward (positive in the direction of flight)
        z = W cross-track: left / right (orbit normal)
    Relative state = our satellite relative to the target, [x, y, z, vx, vy, vz]
    Units: metres, seconds. Hold point = (0, -10, 0), i.e. on the V-bar 10 m directly behind the target.

This module contains:
    1. CW state transition matrix Φ(t) (analytic solution, no integration error)
    2. Sensor simulation (noise, dropout, bad data; controlled by fault-injection parameters)
    3. Kalman filter (CW as the motion model, with a bad-data gate)
    4. Proximity main loop run_proximity(): outputs a sequence of NavFrame (interface 4)
"""

# Lets type hints such as list[...] and tuple[...] work on older Python versions
from __future__ import annotations

from dataclasses import dataclass, field
# Optional[X]: marks a value that may be None (e.g. no sensor reading during a dropout)
from typing import Optional

# numpy: all matrix and vector calculations
import numpy as np

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
    Called by propagate(), CWKalmanFilter.predict(), and firing.py (targeting and miss verification).

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


def propagate(state: np.ndarray, n: float, dt: float) -> np.ndarray:
    """
    Push the current state forward by dt seconds using the Φ table.

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
            true_state = propagate(true_state, n, dt)   # true motion
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


def run_proximity_for_target(target_state, **kwargs) -> list[NavFrame]:
    """Convenience wrapper for C: pass A's TargetState directly (only its n is used)."""
    return run_proximity(target_state.n, **kwargs)
