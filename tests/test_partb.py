"""
Part B tests. Run from the repository root with:  pytest -q

Four groups:
  1. CW dynamics       - prove the state transition matrix (PHI) is correct
  2. Navigation        - prove the filter tracks, rejects bad data, and aborts correctly
  3. Firing            - prove the aiming formula and end-to-end capture work
  4. Part A input/net  - prove km-to-m conversion, orbit checks, and net mass/size handling
"""

# Standard library: datetime / timezone build a UTC time for Part A's SpaceObject
from datetime import datetime, timezone
# csv / math / warnings: read every record of Part A's debris file, 2π for rev/day → rad/s, silence expected warnings
import csv
import math
import warnings

# Third-party: numpy for arrays and numerical comparison (np.allclose tolerates rounding error);
# pytest is the test runner, pytest.approx compares floats with a tolerance
import numpy as np
import pytest

# Project modules: Part A's SpaceObject (real input type) and the Part B functions under test
from data_source import SpaceObject
from firing import (DEBRIS_RADIUS_M, HOLD_POINT, NET_RADIUS_M, CWKalmanFilter, SensorFaults,
                    cw_derivative, cw_stm, cw_targeting_velocity, mean_motion_from_space_object,
                    miss_distance, net_speed_from_energy, cw_propagate, run_proximity,
                    run_part_b, run_proximity_for_target, solve_firing, solve_time_of_flight)

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
        assert np.allclose(cw_propagate(s0, N_ISS, t), s0)


# Checks: with vy0 = -2*n*x0 the relative orbit is a closed ellipse and returns to the start after one period.
# Why: reproduces a known textbook result of CW theory.
def test_closed_ellipse_returns_after_one_period():
    x0 = 5.0                                             # m, start 5 m above the target
    s0 = np.array([x0, 0.0, 1.0, 0.0, -2 * N_ISS * x0, 0.0])
    period = 2 * np.pi / N_ISS                           # s, one full orbit
    assert np.allclose(cw_propagate(s0, N_ISS, period), s0, atol=1e-6)


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
    assert np.allclose(cw_propagate(s, N_ISS, total), s_rk, atol=1e-6)


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


# Checks: with 0.4 m noise, the large uncertainty at start-up does not cause a false abort.
# Why: regression test for a real bug we fixed (aborting at t = 0 before the filter settled).
def test_startup_transient_does_not_abort():
    frames = run_proximity(N_ISS, steps=60, seed=4, faults=SensorFaults(noise_sigma=0.4))
    assert not frames[-1].abort


# Checks: with 3 m noise, the system aborts after the settling period because uncertainty stays too high.
# Why: paired with the previous test, it shows the threshold stops when it should, and only then.
def test_very_noisy_sensor_aborts_after_settle():
    frames = run_proximity(N_ISS, steps=60, seed=4, faults=SensorFaults(noise_sigma=3.0))
    assert frames[-1].abort
    assert "uncertainty" in frames[-1].abort_reason


# ---------------- Group 3: Firing solution ----------------

# Checks: firing with the computed v0 brings the net exactly to the target (error < 1e-9 m) after 0.5 s.
# Why: directly verifies the formula v0 = -inv(PHI_rv) @ PHI_rr @ r0; the start point is deliberately off-axis.
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


# ---------------- Group 4: Part A input and net parameters ----------------

def _space_object(position_km, velocity_km_s, norad_id=99999):
    """Build a Part A SpaceObject at a fixed UTC time; position in km, velocity in km/s; 99999 = local test number."""
    return SpaceObject(norad_id=norad_id, name="TEST", time=datetime(2026, 11, 15, tzinfo=timezone.utc),
                       position=np.asarray(position_km, dtype=float),
                       velocity=np.asarray(velocity_km_s, dtype=float), orientation=None)


# Checks: an ISS-like circular orbit given in km and km/s (6778 km, 7.67 km/s) gives n ≈ 0.00113 rad/s.
# Why: matches the known ISS orbital rate, so the km -> m conversion and the vis-viva formula are both right.
def test_mean_motion_from_iss_like_space_object():
    obj = _space_object([6778.0, 0.0, 0.0], [0.0, 7.67, 0.0])
    assert mean_motion_from_space_object(obj) == pytest.approx(0.00113, rel=1e-2)


# Checks: a position already in metres raises ValueError.
# Why: if Part A ever switches to metres, the x1000 conversion would be applied twice; this guard catches it.
def test_space_object_in_metres_is_rejected():
    obj = _space_object([6.778e6, 0.0, 0.0], [0.0, 7670.0, 0.0])
    with pytest.raises(ValueError, match="looks like metres"):
        mean_motion_from_space_object(obj)


# Checks: a speed above escape speed (15 km/s at 6778 km) gives a <= 0 and raises ValueError.
# Why: an open orbit has no orbital rate, so returning any n here would be wrong.
def test_unbound_orbit_is_rejected():
    obj = _space_object([6778.0, 0.0, 0.0], [0.0, 15.0, 0.0])
    with pytest.raises(ValueError, match="not a closed orbit"):
        mean_motion_from_space_object(obj)


# Checks: 10% faster than circular speed (e ≈ 0.21) makes the entry point warn, but it still returns frames.
# Why: CW assumes a near-circular orbit, so the user must be told accuracy is reduced without stopping the run.
def test_eccentric_orbit_warns():
    obj = _space_object([6778.0, 0.0, 0.0], [0.0, 7.67 * 1.1, 0.0])
    with pytest.warns(UserWarning, match="eccentricity"):
        frames = run_proximity_for_target(obj, steps=1)
    assert len(frames) == 2


def _hold_point_frame():
    """One frame at the hold point (10 m behind the target) from a short fault-free run."""
    return run_proximity(N_ISS, steps=1, seed=0)[-1]


# Checks: 400 J on a 2 kg net gives 20 m/s, and a zero mass raises ValueError.
# Why: matches E = 1/2 m v^2 worked by hand, and a bad mass is rejected instead of dividing by zero.
def test_net_speed_from_energy():
    assert net_speed_from_energy(2.0, 400.0) == pytest.approx(20.0)
    with pytest.raises(ValueError):
        net_speed_from_energy(0.0, 400.0)


# Checks: the size of net_momentum_rsw equals net_mass * net_speed.
# Why: C computes the recoil as dv = -momentum / satellite mass, so this number must be exactly p = m v.
def test_net_momentum_is_mass_times_speed():
    sol = solve_firing(_hold_point_frame(), N_ISS, net_mass=3.0)
    assert np.linalg.norm(sol.net_momentum_rsw) == pytest.approx(3.0 * sol.net_speed)


# Checks: with the same 400 J, a 4 kg net is slower than a 2 kg net.
# Why: v = sqrt(2E / m) falls as m grows, which is the physical trade-off of a heavier net.
def test_heavier_net_is_slower():
    assert net_speed_from_energy(4.0, 400.0) < net_speed_from_energy(2.0, 400.0)


# Checks: a 1 kg and a 10 kg net give exactly the same direction_rsw.
# Why: in vacuum gravity accelerates every mass equally, so mass must not change the aim.
def test_net_mass_does_not_change_direction():
    f = _hold_point_frame()
    light = solve_firing(f, N_ISS, net_mass=1.0)
    heavy = solve_firing(f, N_ISS, net_mass=10.0)
    assert np.array_equal(light.direction_rsw, heavy.direction_rsw)


# Checks: 10 m at 20 m/s (about 0.49 s) passes the 0.3 s deploy time; with a 1.0 s deploy time deploy_ok is False with a message.
# Why: tests both sides of the deploy rule, so a half-open net is flagged and a normal shot is not.
def test_deploy_time_check():
    f = _hold_point_frame()
    sol = solve_firing(f, N_ISS)
    assert sol.time_of_flight == pytest.approx(0.49, abs=0.02)
    assert sol.deploy_ok
    slow = solve_firing(f, N_ISS, deploy_time_s=1.0)
    assert not slow.deploy_ok
    assert slow.message != ""


# Checks: capture_margin = NET_RADIUS_M - DEBRIS_RADIUS_M - predicted_miss.
# Why: this is the number Go / No-Go checks (> 0 means the net covers the debris), so its formula must be exact.
def test_capture_margin_formula():
    sol = solve_firing(_hold_point_frame(), N_ISS)
    assert sol.capture_margin == pytest.approx(NET_RADIUS_M - DEBRIS_RADIUS_M - sol.predicted_miss)


# ---------------- Group 5: Part A output to Part B ----------------

def _part_a_state(norad_id=33757):
    """Real Part A output: the CSV record and app.calculate_state() at the record's EPOCH (UTC); skipped without sgp4."""
    pytest.importorskip("sgp4")
    import app
    record = app.find_omm_record(app.CSV_PATH, norad_id)
    epoch = datetime.fromisoformat(record["EPOCH"]).replace(tzinfo=timezone.utc)
    return record, app.calculate_state(record, epoch)


# Checks: for debris 33757, run_part_b's n is within 1% of the catalogue MEAN_MOTION × 2π / 86400, and norad_id / name pass through.
# Why: an independent value from the CSV confirms the km -> m conversion and vis-viva on real SGP4 output.
def test_part_a_state_gives_catalogue_mean_motion():
    record, obj = _part_a_state()
    result = run_part_b(obj)
    n_catalogue = float(record["MEAN_MOTION"]) * 2 * math.pi / 86400
    assert result.n == pytest.approx(n_catalogue, rel=0.01)
    assert result.norad_id == 33757
    assert result.name == record["OBJECT_NAME"]


# Checks: with a fault-free sensor, run_part_b finds a firing opportunity with capture_margin > 0, deploy_ok and a UTC fire time.
# Why: proves the whole chain from Part A's real state to a usable firing solution works end to end.
def test_part_a_state_gives_firing_opportunity():
    _, obj = _part_a_state()
    result = run_part_b(obj)
    assert result.fire_index is not None
    assert result.solution.capture_margin > 0
    assert result.solution.deploy_ok
    assert result.solution.fire_time.tzinfo is not None


# Checks: a SpaceObject whose frame is not "TEME" is rejected with ValueError.
# Why: vis-viva needs a geocentric inertial state; silently accepting another frame would give a wrong n.
def test_non_teme_state_is_rejected():
    obj = _space_object([6778.0, 0.0, 0.0], [0.0, 7.67, 0.0])
    obj.frame = "RSW"
    with pytest.raises(ValueError, match="TEME"):
        run_part_b(obj)


# Checks: a 30 s forced dropout from t = 0 gives no firing opportunity and a non-empty abort_reason.
# Why: C must receive "do not fire" together with the reason when the sensor fails. The dropout starts at 0 s
#      because with a working sensor the very first frame already qualifies, so a later dropout would never be reached.
def test_long_dropout_gives_no_opportunity():
    _, obj = _part_a_state()
    result = run_part_b(obj, faults=SensorFaults(force_dropout_start=0, force_dropout_seconds=30))
    assert result.fire_index is None
    assert result.solution is None
    assert result.abort_reason != ""


# Checks: every debris record in Part A's CSV goes through app.calculate_state() and run_part_b() without errors.
# Why: real catalogue data includes eccentric and decaying orbits; the entry point must handle all of them (warnings allowed).
def test_all_catalogue_debris_run_through_part_b():
    pytest.importorskip("sgp4")
    import app
    with app.CSV_PATH.open(encoding="utf-8-sig", newline="") as file:
        records = list(csv.DictReader(file))
    assert records
    for record in records:
        epoch = datetime.fromisoformat(record["EPOCH"]).replace(tzinfo=timezone.utc)
        obj = app.calculate_state(record, epoch)
        # Eccentricity warnings are expected for some debris and are not failures here
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = run_part_b(obj)
        assert result.norad_id == int(record["NORAD_CAT_ID"])
