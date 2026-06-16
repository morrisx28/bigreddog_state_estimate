#!/usr/bin/env python3
"""Synthetic demo: a Go2 point-foot robot standing then turning in place.

Runs the estimator on procedurally generated IMU + joint data so you can
see it working without any dataset. Prints odometry every few steps.

    python3 examples/demo_synthetic.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capo_estimator import FusionEstimatorCore, LowlevelState, ConfigIndex as CI


def configure(est):
    """Apply a config.yaml-like parameter set (Go2 point-foot)."""
    s = [0.0] * 100
    s[CI.IndexInOrOut] = 1
    s[CI.IndexStatusOK] = 1
    s[CI.IndexIMUAccEnable] = 1
    s[CI.IndexIMUQuaternionEnable] = 1
    s[CI.IndexIMUGyroEnable] = 1
    s[CI.IndexJointsXYZEnable] = 1
    s[CI.IndexJointsVelocityXYZEnable] = 1
    s[CI.IndexJointsRPYEnable] = 0          # set 1 to enable kinematic yaw correction
    s[CI.IndexSlopeModeTimeThreshold] = 1.0
    s[CI.IndexSlopeModeAngleThreshold] = 5.0 * math.pi / 180.0
    s[CI.IndexLegFootForceThreshold] = -1.0  # Go2 point-foot torque threshold
    s[CI.IndexLegMinStairHeight] = 0.08
    s[CI.IndexStairHeightFogotten] = 1200.0
    s[CI.IndexLegOrientationInitialWeight] = 0.001
    s[CI.IndexLegOrientationTimeWeight] = 1000.0
    s[CI.IndexSlopeEstimationEnable] = 1
    est.fusion_estimator_status(s)


def main():
    est = FusionEstimatorCore()       # defaults to the Go2 point-foot preset
    configure(est)

    dt = 0.004                        # 250 Hz
    n_steps = 1500

    print("step     X       Y       Z      yaw   FL FR RL RR")
    for k in range(n_steps):
        t = k * dt
        st = LowlevelState()
        st.imu.timestamp = int(k * 4)  # milliseconds

        # Stand for 2 s, then yaw back and forth.
        yaw = 0.0 if t < 2.0 else 0.4 * math.sin(1.5 * (t - 2.0))
        st.imu.quaternion = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        st.imu.gyroscope = [0.0, 0.0,
                            0.0 if t < 2.0 else 0.4 * 1.5 * math.cos(1.5 * (t - 2.0))]
        st.imu.accelerometer = [0.0, 0.0, 9.81]

        # Nominal standing posture with a little joint jitter so the
        # signal-availability gate keeps accepting frames.
        for leg in range(4):
            b = leg * 4
            st.motorState[b + 0].q = 0.0 + 0.01 * math.sin(t)
            st.motorState[b + 1].q = 0.9 + 0.01 * math.sin(2 * t)
            st.motorState[b + 2].q = -1.8 + 0.01 * math.cos(t)
            # Torques that yield a downward foot reaction (foot in contact).
            st.motorState[b + 1].tauEst = -8.0
            st.motorState[b + 2].tauEst = 4.0

        odom = est.fusion_estimator(st)

        if k % 150 == 0 or k == n_steps - 1:
            fp = est.legs_pos.FootfallProbability
            print("%4d  %6.3f  %6.3f  %6.3f  %6.3f  %s" % (
                k, odom.XPos, odom.YPos, odom.ZPos, odom.YawRad,
                " ".join("%2.0f" % p for p in fp)))


if __name__ == "__main__":
    main()
