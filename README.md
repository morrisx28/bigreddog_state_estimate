# Bigreddog_state_estimate

A **pure-Python, ROS-free** port of the proprioceptive legged-robot
odometry estimator from
[CAPO-LeggedRobotOdometry](https://github.com/ShineMinxing/CAPO-LeggedRobotOdometry)
(the `FusionEstimator/` C/C++ core).

It estimates full 6-DoF body odometry — position, velocity, acceleration,
orientation, angular velocity/acceleration — using only **IMU** (orientation
quaternion + gyroscope + accelerometer) and **joint motor** data (position,
velocity, torque). No cameras, no LiDAR, no ROS. The only dependency is
`numpy`.

This is a faithful translation of the original algorithm: same Kalman
state-space model, same forward kinematics, contact detection, contact
anchoring, stair-height stabilization, slope estimation, and optional
kinematic yaw correction.

## Install

```bash
pip install -r requirements.txt   # just numpy
```

Then import `capo_estimator` from this directory (or add it to your
`PYTHONPATH` / install as a package).

## Quick start

```python
from capo_estimator import FusionEstimatorCore, LowlevelState, ConfigIndex as CI

est = FusionEstimatorCore()          # defaults to the Go2 point-foot preset

# (optional) configure parameters via the same status-array API as the C++ core
status = [0.0] * 100
status[CI.IndexInOrOut]                  = 1     # 1 = write parameters
status[CI.IndexStatusOK]                 = 1
status[CI.IndexIMUAccEnable]             = 1
status[CI.IndexIMUQuaternionEnable]      = 1
status[CI.IndexIMUGyroEnable]            = 1
status[CI.IndexJointsXYZEnable]          = 1
status[CI.IndexJointsVelocityXYZEnable]  = 1
status[CI.IndexJointsRPYEnable]          = 0     # 1 = kinematic yaw correction
status[CI.IndexLegFootForceThreshold]    = -1.0  # Go2 point-foot torque threshold
status[CI.IndexLegMinStairHeight]        = 0.08
status[CI.IndexStairHeightFogotten]      = 1200.0
status[CI.IndexSlopeEstimationEnable]    = 1
est.fusion_estimator_status(status)

# per control loop:
st = LowlevelState()
st.imu.timestamp     = t_ms                      # milliseconds
st.imu.quaternion    = [w, x, y, z]
st.imu.gyroscope     = [wx, wy, wz]
st.imu.accelerometer = [ax, ay, az]
for i in range(16):
    st.motorState[i].q      = q[i]
    st.motorState[i].dq     = dq[i]
    st.motorState[i].tauEst = tau[i]

odom = est.fusion_estimator(st)
print(odom.XPos, odom.YPos, odom.ZPos, odom.YawRad)
```

## Motor slot layout

16 motor slots, ordered (matching the original repo):

```
0  FL_Hip   1  FL_Thigh   2  FL_Calf   3  FL_Wheel
4  FR_Hip   5  FR_Thigh   6  FR_Calf   7  FR_Wheel
8  RL_Hip   9  RL_Thigh  10  RL_Calf  11  RL_Wheel
12 RR_Hip  13  RR_Thigh  14  RR_Calf  15  RR_Wheel
```

For **point-foot** robots leave the wheel slots (3, 7, 11, 15) at 0.
If you have a **contact sensor** instead of joint torque, put the contact
value into the calf torque slot and set a positive `IndexLegFootForceThreshold`
(see the original repo's notes).

## Kinematic presets

Switch the robot model by sending a mode code through
`fusion_estimator_status` (`status[IndexInOrOut] = code`):

| Code | Method     | Robot                         |
|-----:|------------|-------------------------------|
| 99   | `UseGo2P`  | Unitree Go2, point foot (default) |
| 98   | `UseGo2W`  | Unitree Go2-W, wheel foot     |
| 4    | `UseMP`    | MP point-foot platform        |
| 7    | `UseSP`    | SP point-foot platform        |
| 5    | `UseLW`    | LW wheel-foot platform        |
| 6    | `UseMW`    | MW wheel-foot platform        |

You can also call e.g. `est.legs_pos.UseGo2W()` directly.

## Configuration indices (`ConfigIndex`)

| Index | Name                              | Meaning |
|------:|-----------------------------------|---------|
| 0 | `IndexInOrOut`                | 1 write / 2 read / 3 reset / 4-7,98,99 preset switch |
| 1 | `IndexStatusOK`               | status counter |
| 2 | `IndexIMUAccEnable`           | use accelerometer |
| 3 | `IndexIMUQuaternionEnable`    | use IMU orientation |
| 4 | `IndexIMUGyroEnable`          | use gyroscope |
| 5 | `IndexJointsXYZEnable`        | leg position kinematics |
| 6 | `IndexJointsVelocityXYZEnable`| leg velocity kinematics |
| 7 | `IndexJointsRPYEnable`        | kinematic yaw correction |
| 8 | `IndexSlopeModeTimeThreshold` | slope-mode contact dwell time |
| 9 | `IndexSlopeModeAngleThreshold`| slope-mode angle threshold (rad) |
| 10| `IndexLegFootForceThreshold`  | contact threshold (negative = torque) |
| 11| `IndexLegMinStairHeight`      | min stair-height hypothesis |
| 12| `IndexStairHeightFogotten`    | stored-step fade time |
| 13| `IndexLegOrientationInitialWeight` | yaw-correction initial weight |
| 14| `IndexLegOrientationTimeWeight`    | yaw-correction growth weight |
| 15| `IndexSlopeEstimationEnable`  | enable slope estimation |

Sending `status[IndexInOrOut] = 3` resets the estimated XY position to zero
and re-aligns the yaw correction (the C++ `SportCmd` reset).

## Output (`Odometer`)

`fusion_estimator` returns an `Odometer` with: `XPos/YPos/ZPos`,
`XVel/YVel/ZVel`, `XAcc/YAcc/ZAcc`, `RollRad/PitchRad/YawRad`,
`RollVel/PitchVel/YawVel`, `RollAcc/PitchAcc/YawAcc`,
`FootfallAverageX/Y/Yaw`, per-leg `FLFootLanded/FRFootLanded/RLFootLanded/RRFootLanded`
(contact probability 0..1) and `LoadedWeight`.

## Examples

```bash
# synthetic demo (no data needed): stand, then turn in place
python3 examples/demo_synthetic.py

# replay a CSV log and write an odometry CSV
python3 examples/run_csv.py input.csv --preset go2p --out odom.csv
python3 examples/run_csv.py input.csv --preset go2w --foot-thr -30 --leg-ori
```

See the header of [`examples/run_csv.py`](examples/run_csv.py) for the
expected input CSV columns. Test datasets for the original project are
published at the upstream
[release page](https://github.com/ShineMinxing/CAPO-LeggedRobotOdometry/releases/tag/DataForTest).

## Layout

```
capo_estimator/
  array_utils.py      quaternion / Euler / vector helpers (matrix.c array_*)
  lowlevel_state.py   IMU / MotorState / Odometer / LowlevelState
  kalman.py           9-state linear Kalman filter + state-space model
  sensor_base.py      Sensors base + observation-frame corrections + shared state
  sensor_imu.py       IMU acceleration / orientation+gyro sensors
  sensor_legs.py      leg kinematics, contact, anchoring, slope, yaw correction
  fusion_estimator.py FusionEstimatorCore + ConfigIndex (top-level API)
examples/
  demo_synthetic.py   self-contained synthetic demo
  run_csv.py          offline CSV replay runner
```

## Big Reddog velocity estimation

[`BigReddogStateEstimator`](capo_estimator/big_reddog.py) wraps the core for
the Big Reddog quadruped: it loads a URDF-derived point-foot preset, maps
the 12 SDK motors (FL, FR, RL, RR x hip/thigh/calf) onto the 16-slot joint
message, and returns world- and body-frame base linear velocity.

Raw stance-kinematics velocity is spiky (contact transitions + encoder
noise) and systematically under-scales forward speed. The output is
conditioned in the **body frame** (forward/lateral/vertical, where the
error characteristics are heading-independent) with a causal median plus a
**per-axis** first-order low-pass and a forward-axis gain:

- `vel_filter_tau=(0.08, 0.15, 0.03)` — forward moderate, lateral heavy
  (near-zero noisy signal), vertical light (preserves real landing dynamics)
- `vel_scale=(1.23, 1.0, 1.0)` — corrects the forward under-estimate
  (scale ~0.82 measured consistently across test logs)

On a MuJoCo trot log this took body-frame RMSE from
`vx/vy/vz = 0.110 / 0.053 / 0.068` to `0.064 / 0.043 / 0.056` m/s
(vx −42%). Each result also carries `lin_vel_world_raw` /
`lin_vel_body_raw`. Set `vel_filter_tau=0.0`, `vel_median_window=1` and
`vel_scale=(1,1,1)` to recover the raw output.

> The `1.23` forward gain is an empirical calibration that reproduced
> across two logs of similar gait. Re-check it if your gait/terrain differs;
> the principled fix (weighting each leg's velocity by contact probability
> in the averaging) is a worthwhile next step.

The estimator runs at **200 Hz** by default (`update_rate_hz=200`), which
sets the Kalman prediction time step to 0.005 s. The controller throttles
the call to that rate with `time.perf_counter()`, so it stays at 200 Hz
even when `rt/lowstate` arrives faster on hardware. Change the rate in one
place via `update_rate_hz`.

## Notes on fidelity

- Two coupled 9-state linear Kalman filters are used, exactly as in the
  original: one for linear `[pos, vel, acc]` per axis, one for angular
  `[angle, rate, accel]` per axis.
- The empirical gravity term (`9.43 m/s^2`) and the fixed `1.7°` IMU pitch
  mounting correction from the source are preserved.
- The C++ "signal availability" gate is reproduced: byte-identical
  consecutive IMU/joint frames are skipped (so feeding the same sample
  twice updates nothing), matching upstream behavior.
- Function-local `static` state in the C++ (height map, ring buffers, angle
  unwrap counters, timestamp anchors) is held per-instance, so multiple
  `FusionEstimatorCore` objects are fully independent.

## Credit

Algorithm and original C/C++ implementation by Minxing Sun (Institute of
Optics and Electronics, CAS). Paper: *Contact-Anchored Proprioceptive
Odometry for Quadruped Robots* (arXiv:2602.17393). This package only
re-implements the estimator core in Python.
