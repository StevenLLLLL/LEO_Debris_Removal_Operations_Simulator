"""Part B tests: run  pytest -q  (from the project root)"""

import numpy as np
import pytest

from Location_of_us import (HOLD_POINT, CWKalmanFilter, SensorFaults, cw_derivative,
                            cw_stm, propagate, run_proximity)
from firing import (cw_targeting_velocity, miss_distance, solve_firing,
                    solve_time_of_flight)

N_ISS = 0.001131  # rad/s, orbital angular rate at roughly ISS altitude (period about 92.6 min)


# ---------------- CW transition matrix ----------------
def test_stm_identity_at_zero():
    assert np.allclose(cw_stm(N_ISS, 0.0), np.eye(6))


def test_vbar_point_is_stationary():
    """A point on the V-bar with zero relative velocity stays put —— the reason it is chosen as the hold point."""
    s0 = np.array([0.0, -10.0, 0.0, 0.0, 0.0, 0.0])
    for t in (1.0, 100.0, 3000.0):
        assert np.allclose(propagate(s0, N_ISS, t), s0)


def test_closed_ellipse_returns_after_one_period():
    """With ẏ0 = -2n·x0 the relative orbit is closed and returns to the start after one period."""
    x0 = 5.0
    s0 = np.array([x0, 0.0, 1.0, 0.0, -2 * N_ISS * x0, 0.0])
    period = 2 * np.pi / N_ISS
    assert np.allclose(propagate(s0, N_ISS, period), s0, atol=1e-6)


def test_stm_composition():
    """Φ(t1+t2) = Φ(t2)Φ(t1)."""
    assert np.allclose(cw_stm(N_ISS, 30.0), cw_stm(N_ISS, 10.0) @ cw_stm(N_ISS, 20.0))


def test_stm_matches_rk4_integration():
    """Analytic solution vs numerical integration (RK4) —— shows every term of Φ is correct."""
    s = np.array([3.0, -10.0, 2.0, 0.01, -0.02, 0.005])
    s_rk = s.copy()
    h, total = 0.5, 600.0
    for _ in range(int(total / h)):
        k1 = cw_derivative(s_rk, N_ISS)
        k2 = cw_derivative(s_rk + h / 2 * k1, N_ISS)
        k3 = cw_derivative(s_rk + h / 2 * k2, N_ISS)
        k4 = cw_derivative(s_rk + h * k3, N_ISS)
        s_rk = s_rk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert np.allclose(propagate(s, N_ISS, total), s_rk, atol=1e-6)


# ---------------- Filter ----------------
def test_filter_tracks_truth_without_faults():
    frames = run_proximity(N_ISS, steps=120, seed=1)
    err = np.linalg.norm(frames[-1].est_pos - frames[-1].true_rel)
    assert err < 0.1
    assert not frames[-1].abort


def test_outlier_is_rejected():
    kf = CWKalmanFilter(N_ISS, np.concatenate([HOLD_POINT, np.zeros(3)]), meas_sigma=0.05,
                        p0_pos=0.05, p0_vel=0.001)
    assert kf.update(HOLD_POINT + np.array([5.0, 0.0, 0.0])) == "rejected"
    assert kf.update(HOLD_POINT + np.array([0.02, 0.0, 0.0])) == "ok"


def test_outliers_flagged_in_run():
    frames = run_proximity(N_ISS, steps=120, seed=3,
                           faults=SensorFaults(outlier_rate=0.2))
    statuses = [f.sensor_status for f in frames]
    assert "rejected" in statuses
    err = np.linalg.norm(frames[-1].est_pos - frames[-1].true_rel)
    assert err < 0.2   # bad data did not pull the estimate off


def test_sigma_grows_during_dropout():
    frames = run_proximity(N_ISS, steps=60, seed=2,
                           faults=SensorFaults(force_dropout_start=20, force_dropout_seconds=8))
    assert frames[27].sensor_status == "dropout"
    assert frames[27].est_sigma > frames[19].est_sigma


def test_long_dropout_aborts():
    frames = run_proximity(N_ISS, steps=60, seed=2,
                           faults=SensorFaults(force_dropout_start=20, force_dropout_seconds=30))
    assert frames[-1].abort
    assert "No valid measurement" in frames[-1].abort_reason


def test_same_seed_is_reproducible():
    a = run_proximity(N_ISS, steps=30, seed=7)
    b = run_proximity(N_ISS, steps=30, seed=7)
    assert np.allclose(a[-1].est_pos, b[-1].est_pos)


# ---------------- Firing solution ----------------
def test_targeting_hits_origin():
    r0 = np.array([0.1, -10.0, 0.05])
    T = 0.5
    v0 = cw_targeting_velocity(r0, N_ISS, T)
    end = cw_stm(N_ISS, T) @ np.concatenate([r0, v0])
    assert np.linalg.norm(end[:3]) < 1e-9


def test_time_of_flight_gives_requested_speed():
    r0 = np.array([0.0, -10.0, 0.0])
    T, ok = solve_time_of_flight(r0, np.zeros(3), N_ISS, 20.0)
    assert ok
    assert T == pytest.approx(0.5, rel=1e-3)   # 10 m / 20 m/s


def test_too_slow_net_does_not_converge():
    r0 = np.array([0.0, -10.0, 0.0])
    _, ok = solve_time_of_flight(r0, np.zeros(3), N_ISS, 0.5)   # cannot arrive within 5 s
    assert not ok


def test_perfect_knowledge_gives_zero_miss():
    """When estimate = truth, the miss distance should be close to 0 (targeting and verification logic agree)."""
    frames = run_proximity(N_ISS, steps=1, seed=0, faults=SensorFaults(noise_sigma=1e-6))
    f = frames[-1]
    f.est_pos, f.est_vel = f.true_rel.copy(), f.true_vel.copy()
    sol = solve_firing(f, N_ISS)
    assert sol.converged
    assert sol.predicted_miss < 1e-3
    # fired from directly behind, so the direction should be essentially +S (forward along-track)
    assert sol.direction_rsw[1] > 0.99


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
