"""Part B tests: run  pytest -q  (from the project root)"""

import numpy as np
import pytest

from Location_of_us import (HOLD_POINT, CWKalmanFilter, SensorFaults, cw_derivative,
                            cw_stm, propagate, run_proximity)
from firing import (cw_targeting_velocity, miss_distance, solve_firing,
                    solve_time_of_flight)

# rad/s, orbital angular rate near ISS altitude (period ~92.6 min);
# inside the range of our Cosmos 2251 debris data (0.00096-0.00116 rad/s)
N_ISS = 0.001131


# ---------------- Group 1: CW state transition matrix ----------------

# Checks: after 0 seconds, PHI is the identity matrix.
# Why: with no time passing, the state must not change.
def test_stm_identity_at_zero():
    assert np.allclose(cw_stm(N_ISS, 0.0), np.eye(6))


# Checks: a point 10 m behind the target on the V-bar with zero velocity stays put for 1 s, 100 s and 3000 s.
# Why: this is the reason we chose the V-bar as the firing position (no fuel needed to hold it).
def test_vbar_point_is_stationary():
    # state = (x, y, z, vx, vy, vz): 10 m behind the target, not moving
    s0 = np.array([0.0, -10.0, 0.0, 0.0, 0.0, 0.0])
    for t in (1.0, 100.0, 3000.0):
        assert np.allclose(propagate(s0, N_ISS, t), s0)


# Checks: with vy0 = -2*n*x0 the relative orbit is a closed ellipse and returns to the start after one period.
# Why: reproduces a known textbook result of CW theory.
def test_closed_ellipse_returns_after_one_period():
    x0 = 5.0                                             # m, start 5 m above the target
    s0 = np.array([x0, 0.0, 1.0, 0.0, -2 * N_ISS * x0, 0.0])
    period = 2 * np.pi / N_ISS                           # s, one full orbit
    assert np.allclose(propagate(s0, N_ISS, period), s0, atol=1e-6)


# Checks: PHI(30 s) equals PHI(10 s) @ PHI(20 s).
# Why: real dynamics must compose this way; a single wrong matrix entry usually breaks it.
def test_stm_composition():
    assert np.allclose(cw_stm(N_ISS, 30.0), cw_stm(N_ISS, 10.0) @ cw_stm(N_ISS, 20.0))


# Checks: the analytic PHI and step-by-step RK4 integration of the raw CW equations agree after 600 s.
# Why: two independent methods give the same answer; all six state components are non-zero,
#      so every entry of PHI is exercised.
def test_stm_matches_rk4_integration():
    s = np.array([3.0, -10.0, 2.0, 0.01, -0.02, 0.005])  # m and m/s, deliberately all non-zero
    s_rk = s.copy()
    h, total = 0.5, 600.0                                # s, RK4 step size and total time
    for _ in range(int(total / h)):
        # RK4: four slope estimates per step, weighted average gives the next state
        k1 = cw_derivative(s_rk, N_ISS)
        k2 = cw_derivative(s_rk + h / 2 * k1, N_ISS)
        k3 = cw_derivative(s_rk + h / 2 * k2, N_ISS)
        k4 = cw_derivative(s_rk + h * k3, N_ISS)
        s_rk = s_rk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert np.allclose(propagate(s, N_ISS, total), s_rk, atol=1e-6)


# ---------------- Group 2: Kalman filter and proximity loop ----------------

# Checks: with no faults, the estimate is within 10 cm of the truth after 120 s and no abort is triggered.
# Why: the filter's basic job is to track the true position.
def test_filter_tracks_truth_without_faults():
    frames = run_proximity(N_ISS, steps=120, seed=1)
    err = np.linalg.norm(frames[-1].est_pos - frames[-1].true_rel)  # m, estimate error
    assert err < 0.1
    assert not frames[-1].abort


# Checks: a reading 5 m off is rejected, a reading 2 cm off is accepted.
# Why: tests both directions, so the gate neither accepts bad data nor discards good data.
def test_outlier_is_rejected():
    # Small initial uncertainty so a 5 m jump clearly exceeds the gate
    kf = CWKalmanFilter(N_ISS, np.concatenate([HOLD_POINT, np.zeros(3)]), meas_sigma=0.05,
                        p0_pos=0.05, p0_vel=0.001)
    assert kf.update(HOLD_POINT + np.array([5.0, 0.0, 0.0])) == "rejected"
    assert kf.update(HOLD_POINT + np.array([0.02, 0.0, 0.0])) == "ok"


# Checks: with 20% bad readings, some are marked "rejected" and the estimate stays within 20 cm.
# Why: proves bad data is filtered out during a full simulation, not just in isolation.
def test_outliers_flagged_in_run():
    frames = run_proximity(N_ISS, steps=120, seed=3,
                           faults=SensorFaults(outlier_rate=0.2))
    statuses = [f.sensor_status for f in frames]
    assert "rejected" in statuses
    err = np.linalg.norm(frames[-1].est_pos - frames[-1].true_rel)
    assert err < 0.2   # bad data did not drag the estimate away


# Checks: during a forced dropout (20-28 s), status is "dropout" and sigma at 27 s is larger than at 19 s.
# Why: the system must know it is becoming less certain when it has no readings.
def test_sigma_grows_during_dropout():
    frames = run_proximity(N_ISS, steps=60, seed=2,
                           faults=SensorFaults(force_dropout_start=20, force_dropout_seconds=8))
    assert frames[27].sensor_status == "dropout"
    assert frames[27].est_sigma > frames[19].est_sigma


# Checks: a 30 s dropout (over the 10 s limit) triggers an abort, and the reason mentions the missing measurements.
# Why: confirms the abort happens for the right reason, not by coincidence.
def test_long_dropout_aborts():
    frames = run_proximity(N_ISS, steps=60, seed=2,
                           faults=SensorFaults(force_dropout_start=20, force_dropout_seconds=30))
    assert frames[-1].abort
    assert "No valid measurement" in frames[-1].abort_reason


# Checks: two runs with the same seed give identical results.
# Why: guarantees the demo behaves the same way every time.
def test_same_seed_is_reproducible():
    a = run_proximity(N_ISS, steps=30, seed=7)
    b = run_proximity(N_ISS, steps=30, seed=7)
    assert np.allclose(a[-1].est_pos, b[-1].est_pos)


# ---------------- Firing solution ----------------
def test_targeting_hits_origin():
    r0 = np.array([0.1, -10.0, 0.05])                    # m, slightly off the V-bar
    T = 0.5                                              # s, chosen time of flight
    v0 = cw_targeting_velocity(r0, N_ISS, T)
    end = cw_stm(N_ISS, T) @ np.concatenate([r0, v0])    # net state after T seconds
    assert np.linalg.norm(end[:3]) < 1e-9


# Checks: 10 m at 20 m/s gives a time of flight of about 0.5 s.
# Why: matches the simple estimate 10 / 20, so the bisection found the right root.
def test_time_of_flight_gives_requested_speed():
    r0 = np.array([0.0, -10.0, 0.0])
    T, ok = solve_time_of_flight(r0, np.zeros(3), N_ISS, 20.0)
    assert ok
    assert T == pytest.approx(0.5, rel=1e-3)   # 10 m / 20 m/s


# Checks: a 0.5 m/s net cannot reach 10 m within 5 s, so the solver reports no solution.
# Why: the system must refuse to produce a firing plan it cannot achieve.
def test_too_slow_net_does_not_converge():
    r0 = np.array([0.0, -10.0, 0.0])
    _, ok = solve_time_of_flight(r0, np.zeros(3), N_ISS, 0.5)   # cannot arrive within 5 s
    assert not ok


# Checks: when the estimate equals the truth, the miss is below 1 mm and the direction points forward (+S).
# Why: aiming uses the estimate and verification uses the truth; zero miss here proves both parts agree.
# Note: regression test; it exposed the coarse-sampling error that led us to add ternary refinement.
def test_perfect_knowledge_gives_zero_miss():
    frames = run_proximity(N_ISS, steps=1, seed=0, faults=SensorFaults(noise_sigma=1e-6))
    f = frames[-1]
    # Force the estimate to equal the truth
    f.est_pos, f.est_vel = f.true_rel.copy(), f.true_vel.copy()
    sol = solve_firing(f, N_ISS)
    assert sol.converged
    assert sol.predicted_miss < 1e-3
    # Fired from directly behind, so the direction should point almost exactly along +S (forward)
    assert sol.direction_rsw[1] > 0.99


# Checks: the full pipeline (120 s simulation then firing) converges, hits within the net radius, and uses 20 m/s.
# Why: end-to-end test that all parts work together.
def test_nominal_run_hits_within_net_radius():
    frames = run_proximity(N_ISS, steps=120, seed=5)
    sol = solve_firing(frames[-1], N_ISS)
    assert sol.converged
    assert sol.predicted_miss < 2.0
    assert sol.net_speed == pytest.approx(20.0, rel=1e-6)


def test_startup_transient_does_not_abort():
    """Large uncertainty right after start is normal and must not falsely trigger an abort."""
    frames = run_proximity(N_ISS, steps=60, seed=4, faults=SensorFaults(noise_sigma=0.4))
    assert not frames[-1].abort


def test_very_noisy_sensor_aborts_after_settle():
    frames = run_proximity(N_ISS, steps=60, seed=4, faults=SensorFaults(noise_sigma=3.0))
    assert frames[-1].abort
    assert "uncertainty" in frames[-1].abort_reason
