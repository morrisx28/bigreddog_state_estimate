#!/usr/bin/env python3
"""Offline CSV replay runner for the CAPO proprioceptive estimator.

Reads a CSV of IMU + joint samples, runs the estimator frame by frame, and
writes an odometry CSV (and prints a short summary).

Expected input columns (header row, case-insensitive; missing optional
columns are treated as zero):

  timestamp_ms                      sample time in milliseconds (required)
  qw, qx, qy, qz                    IMU orientation quaternion
  gx, gy, gz                        IMU gyroscope (rad/s)
  ax, ay, az                        IMU accelerometer (m/s^2)
  q0  .. q15                        16 motor positions (rad)
  dq0 .. dq15                       16 motor velocities (rad/s)
  tau0 .. tau15                     16 motor torques (N*m) or contact values

Motor slot order matches the original repo:
  FL_Hip, FL_Thigh, FL_Calf, FL_Wheel,
  FR_Hip, FR_Thigh, FR_Calf, FR_Wheel,
  RL_..., RR_...
For point-foot robots leave the wheel slots (q3/q7/q11/q15, etc.) at 0.

Usage:
  python3 examples/run_csv.py input.csv --preset go2p --out odom.csv
  python3 examples/run_csv.py input.csv --preset go2w --foot-thr -30 --leg-ori
"""

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capo_estimator import FusionEstimatorCore, LowlevelState, ConfigIndex as CI

# preset name -> fusion_estimator_status mode code
PRESET_MODE = {
    "go2p": 99, "mp": 4, "sp": 7, "lw": 5, "mw": 6, "go2w": 98,
}
# sensible default torque/contact thresholds per preset
PRESET_FOOT_THR = {
    "go2p": -1.0, "mp": -80.0, "sp": -40.0, "lw": -125.0, "mw": -85.0, "go2w": -30.0,
}


def configure(est, foot_thr, leg_ori, slope):
    s = [0.0] * 100
    s[CI.IndexInOrOut] = 1
    s[CI.IndexStatusOK] = 1
    s[CI.IndexIMUAccEnable] = 1
    s[CI.IndexIMUQuaternionEnable] = 1
    s[CI.IndexIMUGyroEnable] = 1
    s[CI.IndexJointsXYZEnable] = 1
    s[CI.IndexJointsVelocityXYZEnable] = 1
    s[CI.IndexJointsRPYEnable] = 1 if leg_ori else 0
    s[CI.IndexSlopeModeTimeThreshold] = 1.0
    s[CI.IndexSlopeModeAngleThreshold] = 5.0 * math.pi / 180.0
    s[CI.IndexLegFootForceThreshold] = foot_thr
    s[CI.IndexLegMinStairHeight] = 0.08
    s[CI.IndexStairHeightFogotten] = 1200.0
    s[CI.IndexLegOrientationInitialWeight] = 0.001
    s[CI.IndexLegOrientationTimeWeight] = 1000.0
    s[CI.IndexSlopeEstimationEnable] = 1 if slope else 0
    est.fusion_estimator_status(s)


def _get(row, key, default=0.0):
    v = row.get(key)
    if v is None or v == "":
        return default
    return float(v)


def row_to_state(row):
    st = LowlevelState()
    st.imu.timestamp = int(round(_get(row, "timestamp_ms")))
    st.imu.quaternion = [_get(row, "qw", 1.0), _get(row, "qx"),
                         _get(row, "qy"), _get(row, "qz")]
    st.imu.gyroscope = [_get(row, "gx"), _get(row, "gy"), _get(row, "gz")]
    st.imu.accelerometer = [_get(row, "ax"), _get(row, "ay"), _get(row, "az")]
    for i in range(16):
        st.motorState[i].q = _get(row, "q%d" % i)
        st.motorState[i].dq = _get(row, "dq%d" % i)
        st.motorState[i].tauEst = _get(row, "tau%d" % i)
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input CSV path")
    ap.add_argument("--preset", default="go2p", choices=sorted(PRESET_MODE),
                    help="robot kinematic preset (default: go2p)")
    ap.add_argument("--foot-thr", type=float, default=None,
                    help="foot force/torque contact threshold (overrides preset default)")
    ap.add_argument("--leg-ori", action="store_true",
                    help="enable kinematic yaw correction")
    ap.add_argument("--no-slope", action="store_true",
                    help="disable slope estimation")
    ap.add_argument("--out", default=None, help="output odometry CSV path")
    args = ap.parse_args()

    est = FusionEstimatorCore()
    mode = PRESET_MODE[args.preset]
    m = [0.0] * 100
    m[CI.IndexInOrOut] = mode
    est.fusion_estimator_status(m)

    foot_thr = args.foot_thr if args.foot_thr is not None else PRESET_FOOT_THR[args.preset]
    configure(est, foot_thr, args.leg_ori, not args.no_slope)

    out_f = open(args.out, "w", newline="") if args.out else None
    writer = None
    if out_f:
        writer = csv.writer(out_f)
        writer.writerow(["timestamp_ms", "x", "y", "z", "vx", "vy", "vz",
                         "roll", "pitch", "yaw",
                         "ff_x", "ff_y", "ff_yaw", "loaded_weight",
                         "fl", "fr", "rl", "rr"])

    n = 0
    last = None
    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        # normalize header names to lower case
        reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]
        for raw in reader:
            row = {k.strip().lower(): v for k, v in raw.items()}
            st = row_to_state(row)
            odom = est.fusion_estimator(st)
            last = odom
            n += 1
            if writer:
                fp = est.legs_pos.FootfallProbability
                fa = est.legs_pos.FootfallAveragePosition
                writer.writerow([st.imu.timestamp,
                                 odom.XPos, odom.YPos, odom.ZPos,
                                 odom.XVel, odom.YVel, odom.ZVel,
                                 odom.RollRad, odom.PitchRad, odom.YawRad,
                                 fa[0], fa[1], fa[2], odom.LoadedWeight,
                                 fp[0], fp[1], fp[2], fp[3]])

    if out_f:
        out_f.close()

    print("Processed %d frames." % n)
    if last is not None:
        print("Final %s" % last)
    if args.out:
        print("Odometry written to %s" % args.out)


if __name__ == "__main__":
    main()
