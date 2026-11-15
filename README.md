# LEO Debris Removal Operations Simulator

## Part B: Proximity Navigation & Firing Solution

All Part B code lives in firing.py, as specified by the team's file structure.

### Files

- `firing.py`, section ④ — proximity relative navigation: Clohessy-Wiltshire (CW) state transition matrix, sensor simulation with fault injection (noise, dropout, outliers), a Kalman filter with an outlier gate, and the proximity loop with abort rules (dropout too long, uncertainty too large after settling).
- `firing.py`, section ⑤ — firing solution: CW targeting (velocity needed for the net to reach the target), bisection on time of flight so the net speed matches the launcher speed, miss-distance verification, and a Monte Carlo helper (noise vs capture rate).

### Conventions

- Target-centred RSW frame: `x` radial (away from Earth), `y` along-track, `z` cross-track.
- Units: metres and seconds (angles in radians). Vectors are length-3 numpy arrays; times are UTC `datetime`.
- Hold point: `(0, -10, 0)` m, on the V-bar 10 m behind the target.
- Input SpaceObject from Part A is in km and km/s; Part B converts to m and m/s at the entry point. All Part B outputs (NavFrame, FiringSolution) are in SI units (m, m/s, s, rad).

### Public interface

```python
from firing import run_part_b
result = run_part_b(space_object)   # space_object: output of Part A's calculate_state()
```

`run_part_b(space_object, faults=None, seed=42)` is Part B's single entry point. Input: Part A's `SpaceObject` (km, km/s, frame `TEME`, UTC time; `orientation` is not used). Part B converts units at the entry point and raises `ValueError` if the frame is not `TEME`. It simulates the proximity phase, then scans the frames from the start and picks the first one that is not aborted, converged, `deploy_ok`, and has `NET_RADIUS_M - DEBRIS_RADIUS_M - miss_uncertainty > 0`.

`PartBResult` fields:

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `norad_id` | int | — | NORAD number, copied from Part A |
| `name` | str | — | object name, copied from Part A |
| `epoch` | datetime | UTC | Part A's `SpaceObject.time`; fire time = `epoch + frame.t` |
| `n` | float | rad/s | target orbital angular rate |
| `frames` | list[NavFrame] | SI | one frame per second of the proximity phase |
| `fire_index` | int or None | — | index in `frames` of the first firing opportunity |
| `solution` | FiringSolution or None | SI | firing solution at `fire_index` |
| `abort_reason` | str | — | why there is no opportunity (last frame's abort reason, or "No frame met the firing criteria within the time window"); empty if one was found |

### Lower-level functions

- `run_proximity(n, ...)` → `list[NavFrame]` (one frame per time step; `n` is the target's orbital rate in rad/s, e.g. from `mean_motion_from_space_object()`).
- `run_proximity_for_target(space_object, ...)` → `list[NavFrame]`: same as `run_proximity`, but takes Part A's `SpaceObject` and computes `n` from its position and velocity (vis-viva, `MU_EARTH` from `config.py`).
- `solve_firing(frame, n, epoch=None, net_speed=20.0)` → `FiringSolution`. `epoch` can be `SpaceObject.time` (UTC); the fire time is `epoch + frame.t`.

### Notes for Part C

- `NavFrame` has two extra fields beyond the interface: `true_vel` (true relative velocity, used for miss verification) and `abort_reason` (text, empty if not aborted).
- `FiringSolution` has extra fields beyond the interface:
  - `net_velocity_rsw` — net velocity relative to the satellite (m/s, RSW).
  - `miss_uncertainty` — 3σ position uncertainty (m). `predicted_miss` is computed from the simulation's true state, which a real mission would not have.
  - `message` — explanation when no solution is found or the net would not be fully open.
  - `net_mass` (kg), `net_momentum_rsw` (N·s, RSW), `capture_margin` (m, `NET_RADIUS_M - DEBRIS_RADIUS_M - predicted_miss`), `deploy_ok` (`time_of_flight >= NET_DEPLOY_TIME_S`).
- Recoil: net_velocity_rsw is the net velocity relative to the satellite's pre-release velocity.
  Exact recoil by momentum conservation: dv_sat = -net_momentum_rsw / M_dry, where M_dry is the satellite mass after release (excluding the net).
  Do not substitute net_velocity_rsw into -(m/(M+m))*u; that formula expects u = post-separation relative speed and would be about m/M off (about 2% for a 2 kg net and 100 kg satellite).
- Go/No-Go should require capture_margin > 0 (using miss_uncertainty in place of predicted_miss for real operations) and deploy_ok.

### Net parameters

Defined in `config.py` under "Part B: Firing solution and net parameters" (the navigation constants are under "Part B: Location of us"); `DEFAULT_NET_SPEED` stays in `firing.py`. The net speed is derived from mass and energy: `v = sqrt(2E / m)`.

| Name | Value | Unit | Note |
|---|---|---|---|
| `NET_MASS_KG` | 2.0 | kg | assumed |
| `NET_RADIUS_M` | 2.0 | m | assumed |
| `NET_DEPLOY_TIME_S` | 0.3 | s | assumed |
| `LAUNCHER_ENERGY_J` | 400.0 | J | assumed |
| `DEBRIS_RADIUS_M` | 0.5 | m | assumed (CelesTrak data has no size) |
| `DEFAULT_NET_SPEED` | 20.0 | m/s | derived from `NET_MASS_KG` and `LAUNCHER_ENERGY_J` |

### Running the tests

```bash
pip install -r requirements.txt
python -m pytest -q
```
