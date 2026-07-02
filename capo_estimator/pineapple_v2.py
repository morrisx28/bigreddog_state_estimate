"""High-level proprioceptive velocity estimator for the Pineapple V2 biped.

Thin convenience wrapper around :class:`FusionEstimatorCore` for the Pineapple
V2 **wheeled biped** (two legs, each hip/thigh/calf + a driven wheel). It:

  * loads the Pineapple V2 two-chain wheel-foot kinematic preset
    (:meth:`SensorLegsPos.UsePineappleV2`),
  * maps the 8 SDK motors onto the estimator's 16-slot joint message
    (left leg -> slots 0-3, right leg -> slots 4-7),
  * applies the same tuned causal output filter to the base linear velocity
    as the quadruped preset, and
  * returns both world-frame and body-frame base linear velocity (filtered
    and raw).

Unlike a point-foot quadruped, the base linear velocity for this robot comes
mostly from **wheel odometry**: both wheels stay loaded almost continuously,
so the estimator averages each contacting wheel's rolling velocity
(``wheel_radius * wheel_speed``, shank-pitch compensated) with the stance-leg
kinematics. Good wheel joint velocity and a correctly tuned contact threshold
(``foot_force_threshold``) therefore matter most.

Joint order
-----------
The 8 joints must be supplied in the controller's xml order::

    [L_hip, L_thigh, L_calf, L_wheel, R_hip, R_thigh, R_calf, R_wheel]

which maps directly onto estimator slots 0-7.

Example
-------
>>> est = PineappleV2StateEstimator()
>>> vx, vy, vz = est.update(quat, gyro, accel, q8, dq8, tau8, t_ms).lin_vel_body
"""

import math
from collections import deque

import numpy as np

from . import array_utils as au
from .fusion_estimator import FusionEstimatorCore, ConfigIndex as CI
from .lowlevel_state import LowlevelState


class PineappleV2Vel:
    """Result of one estimator update.

    Attributes
    ----------
    lin_vel_world, lin_vel_body : tuple(3)
        Filtered base linear velocity in world / body frame.
    lin_vel_world_raw, lin_vel_body_raw : tuple(3)
        Unfiltered estimate (straight from the Kalman filter).
    pos_world, rpy, wheel_contact, odom :
        Position, orientation, per-wheel (left, right) contact probability and
        the full :class:`Odometer`.
    """

    __slots__ = ("lin_vel_world", "lin_vel_body",
                 "lin_vel_world_raw", "lin_vel_body_raw",
                 "pos_world", "rpy", "wheel_contact", "odom")

    def __init__(self, odom, vel_world_filt, vel_body_filt,
                 vel_world_raw, vel_body_raw):
        self.odom = odom
        self.pos_world = (odom.XPos, odom.YPos, odom.ZPos)
        self.rpy = (odom.RollRad, odom.PitchRad, odom.YawRad)
        # Only the first two contact chains exist (left, right wheel).
        self.wheel_contact = (odom.FLFootLanded, odom.FRFootLanded)
        self.lin_vel_world = tuple(float(v) for v in vel_world_filt)
        self.lin_vel_body = tuple(float(v) for v in vel_body_filt)
        self.lin_vel_world_raw = tuple(float(v) for v in vel_world_raw)
        self.lin_vel_body_raw = tuple(float(v) for v in vel_body_raw)


class PineappleV2StateEstimator:
    """Proprioceptive odometry/velocity estimator preset for Pineapple V2.

    Parameters
    ----------
    foot_force_threshold : float
        Contact threshold on the estimated wheel vertical force (negative,
        torque-derived). A wheel counts as on the ground when its estimated
        vertical force drops below this value. Tune on hardware: for a wheeled
        biped both wheels are usually loaded, so this should be small enough in
        magnitude that contact (hence wheel odometry) is detected continuously,
        but not so small that swing/lift-off is missed.
    enable_slope : bool
        Enable slope/height stabilization.
    vel_filter_tau : float or sequence of 3 floats
        First-order low-pass time constant(s) (s) for the BODY-frame velocity
        (forward, lateral, vertical). A scalar applies to all axes; 0 on an
        axis disables its low-pass.
    vel_median_window : int
        Causal median window (samples) applied before the low-pass to reject
        contact-transition spikes. 1 disables it. Defaults to 1: wheel velocity
        is smooth (no foot touchdown/liftoff like a stepping robot), so the
        median only adds lag here -- a MuJoCo sweep showed it strictly hurt
        body-velocity RMSE. Raise it (e.g. 3) only if real wheel encoders show
        spiky velocity.
    vel_scale : sequence of 3 floats
        Per-axis BODY-frame velocity gain applied before filtering. Defaults to
        ``(1, 1, 1)``: wheel odometry does not have the systematic forward
        under-estimate that point-foot stance kinematics does (a forward-scale
        sweep confirmed 1.0 is optimal -- any gain made vx worse).
    update_rate_hz : float
        Rate at which :meth:`update` is called. Sets the Kalman prediction
        time step (1/rate) and the low-pass fallback period.

    Notes
    -----
    The filter defaults below were tuned against MuJoCo ground truth on the
    scripted command sequence (accel/decel, height changes, forward+yaw): the
    chosen ``vel_filter_tau=(0.03, 0.10, 0.03)`` with ``vel_median_window=1``
    took body-frame velocity RMSE from vx/vy/vz = 0.119/0.017/0.044 to
    0.104/0.017/0.032 m/s. The lateral axis keeps a heavier low-pass (0.10):
    its true signal is small and noise-dominated, so less smoothing only added
    noise. The residual vx error is the core filter converging during hard
    accel/decel and is not reachable by output filtering.
    """

    NUM_MOTORS = 8

    def __init__(self, foot_force_threshold=-15.0, enable_slope=True,
                 vel_filter_tau=(0.03, 0.10, 0.03), vel_median_window=1,
                 vel_scale=(1.0, 1.0, 1.0), update_rate_hz=200.0):
        nominal_dt = 1.0 / float(update_rate_hz)
        self.update_rate_hz = float(update_rate_hz)
        self.core = FusionEstimatorCore(dt=nominal_dt)
        self.core.legs_pos.UsePineappleV2()

        status = [0.0] * 100
        status[CI.IndexInOrOut] = 1
        status[CI.IndexStatusOK] = 1
        status[CI.IndexIMUAccEnable] = 1
        status[CI.IndexIMUQuaternionEnable] = 1
        status[CI.IndexIMUGyroEnable] = 1
        status[CI.IndexJointsXYZEnable] = 1
        status[CI.IndexJointsVelocityXYZEnable] = 1
        status[CI.IndexJointsRPYEnable] = 0  # leg-yaw needs 4 chains; off here
        status[CI.IndexSlopeModeTimeThreshold] = 1.0
        status[CI.IndexSlopeModeAngleThreshold] = 5.0 * math.pi / 180.0
        status[CI.IndexLegFootForceThreshold] = foot_force_threshold
        status[CI.IndexLegMinStairHeight] = 0.05
        status[CI.IndexStairHeightFogotten] = 1200.0
        status[CI.IndexLegOrientationInitialWeight] = 0.001
        status[CI.IndexLegOrientationTimeWeight] = 1000.0
        status[CI.IndexSlopeEstimationEnable] = 1 if enable_slope else 0
        self.core.fusion_estimator_status(status)

        self._st = LowlevelState()

        # ---- output velocity filter state (body frame, per axis) ----
        tau = np.atleast_1d(np.asarray(vel_filter_tau, dtype=float))
        if tau.size == 1:
            tau = np.repeat(tau, 3)
        self.vel_filter_tau = tau                      # (3,)
        self.vel_median_window = max(1, int(vel_median_window))
        self.vel_scale = np.asarray(vel_scale, dtype=float)
        self.nominal_dt = float(nominal_dt)
        self._vel_buf = deque(maxlen=self.vel_median_window)
        self._vel_ema = None
        self._last_t_ms = None

    def reset_position(self):
        """Zero the estimated XY position and re-align the yaw correction."""
        status = [0.0] * 100
        status[CI.IndexInOrOut] = 3
        status[CI.IndexStatusOK] = 1
        self.core.fusion_estimator_status(status)

    def reset_filter(self):
        """Clear the output-velocity filter memory."""
        self._vel_buf.clear()
        self._vel_ema = None
        self._last_t_ms = None

    def _filter_velocity(self, vel_body_raw, t_ms):
        """Causal scale -> median -> per-axis low-pass on body velocity."""
        scaled = np.asarray(vel_body_raw, dtype=float) * self.vel_scale

        # median spike rejection over the recent window
        self._vel_buf.append(scaled)
        med = np.median(np.asarray(self._vel_buf), axis=0)

        if self._last_t_ms is not None:
            dt = (t_ms - self._last_t_ms) / 1000.0
        else:
            dt = self.nominal_dt
        if not (0.0 < dt <= 0.1):
            dt = self.nominal_dt
        self._last_t_ms = t_ms

        # per-axis first-order low-pass; tau<=0 => pass median through
        alpha = np.where(self.vel_filter_tau > 0.0,
                         dt / (self.vel_filter_tau + dt), 1.0)
        if self._vel_ema is None:
            self._vel_ema = med
        else:
            self._vel_ema = alpha * med + (1.0 - alpha) * self._vel_ema
        return self._vel_ema

    def update(self, quat, gyro, accel, q8, dq8, tau8, timestamp_ms):
        """Run one estimation step.

        Parameters
        ----------
        quat : sequence of 4 floats
            IMU orientation ``[w, x, y, z]``.
        gyro : sequence of 3 floats
            Body angular velocity ``[wx, wy, wz]`` (rad/s).
        accel : sequence of 3 floats
            Body linear acceleration ``[ax, ay, az]`` (m/s^2).
        q8, dq8, tau8 : sequence of 8 floats
            Joint position / velocity / torque in xml order
            ``[L_hip, L_thigh, L_calf, L_wheel, R_hip, R_thigh, R_calf, R_wheel]``.
        timestamp_ms : int
            Monotonic time stamp in milliseconds.

        Returns
        -------
        PineappleV2Vel
        """
        st = self._st
        st.imu.timestamp = int(timestamp_ms)
        st.imu.quaternion = [float(quat[0]), float(quat[1]),
                             float(quat[2]), float(quat[3])]
        st.imu.gyroscope = [float(gyro[0]), float(gyro[1]), float(gyro[2])]
        st.imu.accelerometer = [float(accel[0]), float(accel[1]), float(accel[2])]

        # Map the 8 SDK motors directly onto estimator slots 0-7
        # (left leg -> 0-3, right leg -> 4-7); slots 8-15 stay at zero.
        for i in range(self.NUM_MOTORS):
            m = st.motorState[i]
            m.q = float(q8[i])
            m.dq = float(dq8[i])
            m.tauEst = float(tau8[i])
        for i in range(self.NUM_MOTORS, 16):
            m = st.motorState[i]
            m.q = 0.0
            m.dq = 0.0
            m.tauEst = 0.0

        odom = self.core.fusion_estimator(st)

        # Condition the velocity in the body frame (scale/noise are body-frame
        # properties), then rotate the filtered result back to world.
        q = au.quaternion_normalize(st.imu.quaternion)
        q_inv = au.quaternion_conjugate(q)
        vel_world_raw = [odom.XVel, odom.YVel, odom.ZVel]
        vel_body_raw = au.quaternion_rotate_vector(q_inv, vel_world_raw)

        vel_body_filt = self._filter_velocity(vel_body_raw, int(timestamp_ms))
        vel_world_filt = au.quaternion_rotate_vector(q, list(vel_body_filt))

        return PineappleV2Vel(odom, vel_world_filt, vel_body_filt,
                              vel_world_raw, vel_body_raw)
