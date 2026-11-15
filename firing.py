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
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from relnav import NavFrame, SensorFaults, cw_stm, run_proximity

DEFAULT_NET_SPEED = 20.0     # m/s, desired net speed
DEFAULT_NET_RADIUS = 2.0     # m, effective radius of the deployed net (parameter, adjust to the actual net spec)
T_MIN, T_MAX = 1e-3, 5.0     # s, search range for time of flight


@dataclass
class FiringSolution:
    """Interface 5."""
    fire_time: Optional[datetime]   # UTC
    direction_rsw: np.ndarray       # firing direction unit vector (RSW)
    azimuth: float                  # rad, measured from along-track, positive towards cross-track
    elevation: float                # rad, positive towards radial (away from Earth)
    net_speed: float                # m/s, net muzzle speed relative to the satellite
    time_of_flight: float           # s
    predicted_miss: float           # m, miss distance checked against the "true state" (simulation verification)
    converged: bool                 # whether root finding succeeded
    # ---- extra information beyond the interface; C may choose to display it ----
    net_velocity_rsw: np.ndarray = None   # m/s, net velocity vector relative to the satellite (C needs it for recoil)
    miss_uncertainty: float = 0.0         # m, 3σ estimate error; in a real mission only this can be used for the decision
    message: str = ""


# ---------------------------------------------------------------------------
# CW targeting
# ---------------------------------------------------------------------------
def cw_targeting_velocity(r0: np.ndarray, n: float, T: float) -> np.ndarray:
    """Relative velocity v0 needed to start from r0 and reach the origin after T seconds."""
    phi = cw_stm(n, T)
    phi_rr = phi[:3, :3]
    phi_rv = phi[:3, 3:]
    # use solve instead of an explicit inverse: numerically more stable, same result
    return -np.linalg.solve(phi_rv, phi_rr @ r0)


def _speed_error(T: float, r0, v_sat, n, net_speed) -> float:
    """f(T) = |v0(T) - v_sat| - desired net speed. Smaller T needs a larger speed, so f is monotonically decreasing."""
    return float(np.linalg.norm(cw_targeting_velocity(r0, n, T) - v_sat) - net_speed)


def solve_time_of_flight(r0, v_sat, n, net_speed,
                         t_min=T_MIN, t_max=T_MAX, tol=1e-9, max_iter=100):
    """
    Bisection for T such that the net speed equals net_speed exactly.
    Returns (T, converged). If the target cannot be reached within t_max (net too slow), converged = False.
    """
    f_lo = _speed_error(t_min, r0, v_sat, n, net_speed)
    f_hi = _speed_error(t_max, r0, v_sat, n, net_speed)
    if f_lo < 0 or f_hi > 0:
        return float("nan"), False   # no root in the interval

    lo, hi = t_min, t_max
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if _speed_error(mid, r0, v_sat, n, net_speed) > 0:
            lo = mid          # speed still too high → need a longer time of flight
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi), True


# ---------------------------------------------------------------------------
# Miss verification
# ---------------------------------------------------------------------------
def miss_distance(sat_pos: np.ndarray, sat_vel: np.ndarray, v_net: np.ndarray,
                  n: float, T: float, samples: int = 300) -> float:
    """
    Launch the net from the satellite position with (satellite velocity + v_net), propagate with CW up to 1.5T,
    and find the minimum distance from the net to the target (origin).

    When the "true" state is passed in, the result is the actual offset caused by the estimation error.
    """
    net_state = np.concatenate([sat_pos, sat_vel + v_net])

    def dist(t: float) -> float:
        return float(np.linalg.norm((cw_stm(n, t) @ net_state)[:3]))

    # step 1: coarse sampling to find roughly which interval contains the closest approach
    times = np.linspace(0.0, 1.5 * T, samples)
    dists = [dist(t) for t in times]
    i = int(np.argmin(dists))
    lo = times[max(i - 1, 0)]
    hi = times[min(i + 1, samples - 1)]

    # step 2: refine within this small interval using ternary search (distance is unimodal near the closest approach)
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
    """RSW unit vector → (azimuth, elevation), in rad."""
    elevation = float(np.arcsin(np.clip(d[0], -1.0, 1.0)))   # radial component
    azimuth = float(np.arctan2(d[2], d[1]))                  # from S towards W
    return azimuth, elevation


def solve_firing(frame: NavFrame, n: float,
                 epoch: Optional[datetime] = None,
                 net_speed: float = DEFAULT_NET_SPEED) -> FiringSolution:
    """
    Compute the firing solution from the latest navigation frame.
    Targeting uses the estimated state (est_pos, est_vel); verification uses the true state (true_rel, true_vel).
    epoch: epoch of the target state (A's TargetState.epoch); fire time = epoch + frame.t.
    """
    r0 = frame.est_pos
    v_sat = frame.est_vel
    fire_time = epoch + timedelta(seconds=frame.t) if epoch else None

    T, ok = solve_time_of_flight(r0, v_sat, n, net_speed)
    if not ok:
        return FiringSolution(
            fire_time=fire_time, direction_rsw=np.zeros(3), azimuth=float("nan"),
            elevation=float("nan"), net_speed=net_speed, time_of_flight=float("nan"),
            predicted_miss=float("inf"), converged=False,
            net_velocity_rsw=np.zeros(3), miss_uncertainty=3 * frame.est_sigma,
            message="No solution within the time-of-flight search range (net speed too low or distance too large)",
        )

    v0 = cw_targeting_velocity(r0, n, T)
    v_net = v0 - v_sat
    d = v_net / np.linalg.norm(v_net)
    az, el = direction_to_angles(d)

    miss = miss_distance(frame.true_rel, frame.true_vel, v_net, n, T)

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
    )


# ---------------------------------------------------------------------------
# Monte Carlo: noise → capture success rate
# ---------------------------------------------------------------------------
def monte_carlo(n: float, noise_sigmas, runs: int = 200, steps: int = 60,
                net_speed: float = DEFAULT_NET_SPEED,
                net_radius: float = DEFAULT_NET_RADIUS,
                enforce_abort: bool = True) -> dict[float, dict]:
    """
    Run `runs` times for each noise level (a different random seed each time).
    Returns {noise σ: {"hit_rate", "abort_rate", "mean_miss"}}:
        hit_rate   —— fraction of runs that fired and hit (miss distance < net radius)
        abort_rate —— fraction of runs where the system judged it unsafe and aborted
        mean_miss  —— mean miss distance (m) over the runs that actually fired
    With enforce_abort=False the abort rules are ignored and it always fires, used to plot the "pure physics" miss curve.
    """
    results = {}
    for sigma in noise_sigmas:
        hits = aborts = 0
        misses = []
        for seed in range(runs):
            frames = run_proximity(n, steps=steps, seed=seed,
                                   faults=SensorFaults(noise_sigma=sigma))
            last = frames[-1]
            if last.abort and enforce_abort:
                aborts += 1
                continue
            sol = solve_firing(last, n, net_speed=net_speed)
            if not sol.converged:
                continue
            misses.append(sol.predicted_miss)
            if sol.predicted_miss < net_radius:
                hits += 1
        results[float(sigma)] = {
            "hit_rate": hits / runs,
            "abort_rate": aborts / runs,
            "mean_miss": float(np.mean(misses)) if misses else float("nan"),
        }
    return results
