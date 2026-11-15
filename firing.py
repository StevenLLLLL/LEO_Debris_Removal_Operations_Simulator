"""
firing.py  —  Part B ⑤: firing solution + miss verification

Core idea:
    Once the net leaves the satellite it is a free-flying object, so in the RSW frame centred on the target
    the net obeys the CW equations just like our satellite. The target is always at the origin.

    The problem becomes: starting from the current position r0, the net must reach the origin exactly
    T seconds later; what should its initial velocity v0 (relative to the target) be?

    Split Φ(T) into 3x3 blocks:
        r(T) = Φ_rr(T) r0 + Φ_rv(T) v0 = 0
    ⇒  v0 = -Φ_rv(T)⁻¹ Φ_rr(T) r0

    The net inherits the satellite's own relative velocity v_sat, so the launcher must add:
        v_net = v0 - v_sat

    The net speed |v_net| must equal a given value, so we solve a 1-D root-finding problem in T (bisection).

All inputs and outputs are in SI units (m, m/s, s, rad); fire_time is a UTC datetime.
"""

# Lets type hints such as tuple[...] and dict[...] work on older Python versions
from __future__ import annotations

# dataclass: auto-generates the FiringSolution constructor
from dataclasses import dataclass
# datetime / timedelta: fire time = epoch (UTC) + seconds since start
from datetime import datetime, timedelta
# Optional[X]: marks a value that may be None (e.g. no epoch given)
from typing import Optional

# numpy: all matrix and vector calculations
import numpy as np

# Part B ④: navigation frames, sensor faults, the CW matrix and the proximity simulation
from Location_of_us import NavFrame, SensorFaults, cw_stm, run_proximity

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
