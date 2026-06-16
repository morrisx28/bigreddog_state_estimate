"""High-level proprioceptive velocity estimator for the Big Reddog quadruped.

Thin convenience wrapper around :class:`FusionEstimatorCore` that:

  * loads the Big Reddog point-foot kinematic preset,
  * maps the 12 SDK motors (FL, FR, RL, RR x hip/thigh/calf) onto the
    estimator's 16-slot joint message,
  * applies a tuned causal output filter to the base linear velocity, and
  * returns both world-frame and body-frame base linear velocity (filtered
    and raw).

The estimator is proprioceptive: it fuses IMU orientation/gyro/accel with
leg forward kinematics and contact anchoring. Linear velocity comes mainly
from the stance-leg kinematics, so good joint torque (for contact
detection) and joint velocity are what matter most.

Output velocity filter
----------------------
Raw stance-kinematics velocity is spiky: every contact touchdown/liftoff
changes the averaged support set, and encoder velocity noise feeds straight
through. The output is conditioned in the **body frame** (forward / lateral
/ vertical), where the error characteristics are consistent regardless of
heading:

  * a causal median + per-axis first-order low-pass removes the spikes;
  * the forward axis gets a scale gain, because proprioceptive odometry
    systematically UNDER-estimates forward speed.

Two MuJoCo trot logs both showed a forward-velocity scale of ~0.82 (i.e. a
~1.23 correction) and that the vertical axis carries real fast dynamics
(so it wants little smoothing) while the lateral axis is near-zero and
noise-dominated (so it wants more). The tuned per-axis defaults
(``vel_filter_tau=(0.08, 0.15, 0.03)``, ``vel_scale=(1.23, 1.0, 1.0)``)
took body-frame RMSE from vx/vy/vz = 0.110 / 0.053 / 0.068 to
0.064 / 0.043 / 0.056 m/s. Set ``vel_filter_tau=0.0``,
``vel_median_window=1`` and ``vel_scale=(1,1,1)`` to recover the raw output.

Example
-------
>>> est = BigReddogStateEstimator()
>>> vx, vy, vz = est.update(quat, gyro, accel, q12, dq12, tau12, t_ms).lin_vel_body
"""

import math
from collections import deque

import numpy as np

from . import array_utils as au
from .fusion_estimator import FusionEstimatorCore, ConfigIndex as CI
from .lowlevel_state import LowlevelState


class BigReddogVel:
    """Result of one estimator update.

    Attributes
    ----------
    lin_vel_world, lin_vel_body : tuple(3)
        Filtered base linear velocity in world / body frame.
    lin_vel_world_raw, lin_vel_body_raw : tuple(3)
        Unfiltered estimate (straight from the Kalman filter).
    pos_world, rpy, foot_contact, odom :
        Position, orientation, per-leg contact probability and the full
        :class:`Odometer`.
    """

    __slots__ = ("lin_vel_world", "lin_vel_body",
                 "lin_vel_world_raw", "lin_vel_body_raw",
                 "pos_world", "rpy", "foot_contact", "odom")

    def __init__(self, odom, vel_world_filt, vel_body_filt,
                 vel_world_raw, vel_body_raw):
        self.odom = odom
        self.pos_world = (odom.XPos, odom.YPos, odom.ZPos)
        self.rpy = (odom.RollRad, odom.PitchRad, odom.YawRad)
        self.foot_contact = (odom.FLFootLanded, odom.FRFootLanded,
                             odom.RLFootLanded, odom.RRFootLanded)
        self.lin_vel_world = tuple(float(v) for v in vel_world_filt)
        self.lin_vel_body = tuple(float(v) for v in vel_body_filt)
        self.lin_vel_world_raw = tuple(float(v) for v in vel_world_raw)
        self.lin_vel_body_raw = tuple(float(v) for v in vel_body_raw)


class BigReddogStateEstimator:
    """Proprioceptive odometry/velocity estimator preset for Big Reddog.

    Parameters
    ----------
    foot_force_threshold : float
        Contact threshold on the estimated foot vertical force (negative,
        torque-derived). A foot counts as on the ground when its estimated
        vertical force drops below this value. Tune on hardware: too small
        in magnitude => spurious contacts; too large => stance never
        detected and velocity freezes.
    enable_leg_yaw : bool
        Enable kinematics-based yaw drift correction (off by default; it can
        slightly reduce translational accuracy during motion).
    enable_slope : bool
        Enable slope/height stabilization.
    vel_filter_tau : float or sequence of 3 floats
        First-order low-pass time constant(s) (s) for the BODY-frame velocity
        (forward, lateral, vertical). A scalar applies to all axes; 0 on an
        axis disables its low-pass. Default ``(0.08, 0.15, 0.03)``: forward
        moderate, lateral heavy (near-zero noisy signal), vertical light
        (preserves real landing dynamics).
    vel_median_window : int
        Causal median window (samples) applied before the low-pass to reject
        contact-transition spikes. 1 disables it.
    vel_scale : sequence of 3 floats
        Per-axis BODY-frame velocity gain applied before filtering. Default
        ``(1.23, 1.0, 1.0)`` corrects the systematic forward-speed
        under-estimate (scale ~0.82 seen across test logs). Set
        ``(1.0, 1.0, 1.0)`` for the uncalibrated estimate.
    update_rate_hz : float
        Rate at which :meth:`update` is called. Sets the Kalman prediction
        time step (1/rate) and the low-pass fallback period. Default 200 Hz.
    """

    def __init__(self, foot_force_threshold=-30.0, enable_leg_yaw=False,
                 enable_slope=True, vel_filter_tau=(0.08, 0.15, 0.03),
                 vel_median_window=5, vel_scale=(1.23, 1.0, 1.0),
                 update_rate_hz=100.0):
        nominal_dt = 1.0 / float(update_rate_hz)
        self.update_rate_hz = float(update_rate_hz)
        self.core = FusionEstimatorCore(dt=nominal_dt)
        self.core.legs_pos.UseBigReddog()

        status = [0.0] * 100
        status[CI.IndexInOrOut] = 1
        status[CI.IndexStatusOK] = 1
        status[CI.IndexIMUAccEnable] = 1
        status[CI.IndexIMUQuaternionEnable] = 1
        status[CI.IndexIMUGyroEnable] = 1
        status[CI.IndexJointsXYZEnable] = 1
        status[CI.IndexJointsVelocityXYZEnable] = 1
        status[CI.IndexJointsRPYEnable] = 1 if enable_leg_yaw else 0
        status[CI.IndexSlopeModeTimeThreshold] = 1.0
        status[CI.IndexSlopeModeAngleThreshold] = 5.0 * math.pi / 180.0
        status[CI.IndexLegFootForceThreshold] = foot_force_threshold
        status[CI.IndexLegMinStairHeight] = 0.08
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

    def update(self, quat, gyro, accel, q12, dq12, tau12, timestamp_ms):
        """Run one estimation step.

        Parameters
        ----------
        quat : sequence of 4 floats
            IMU orientation ``[w, x, y, z]``.
        gyro : sequence of 3 floats
            Body angular velocity ``[wx, wy, wz]`` (rad/s).
        accel : sequence of 3 floats
            Body linear acceleration ``[ax, ay, az]`` (m/s^2).
        q12, dq12, tau12 : sequence of 12 floats
            Joint position / velocity / torque in SDK order
            (FL, FR, RL, RR x hip, thigh, calf).
        timestamp_ms : int
            Monotonic time stamp in milliseconds.

        Returns
        -------
        BigReddogVel
        """
        st = self._st
        st.imu.timestamp = int(timestamp_ms)
        st.imu.quaternion = [float(quat[0]), float(quat[1]),
                             float(quat[2]), float(quat[3])]
        st.imu.gyroscope = [float(gyro[0]), float(gyro[1]), float(gyro[2])]
        st.imu.accelerometer = [float(accel[0]), float(accel[1]), float(accel[2])]

        # Map 12 SDK motors -> 16-slot estimator layout (leg L joint j -> 4*L+j).
        for L in range(4):
            for j in range(3):
                m = st.motorState[4 * L + j]
                m.q = float(q12[3 * L + j])
                m.dq = float(dq12[3 * L + j])
                m.tauEst = float(tau12[3 * L + j])
            # wheel slot (4*L+3) stays at zero for the point-foot robot
            w = st.motorState[4 * L + 3]
            w.q = 0.0
            w.dq = 0.0
            w.tauEst = 0.0

        odom = self.core.fusion_estimator(st)

        # Condition the velocity in the body frame (scale/noise are body-frame
        # properties), then rotate the filtered result back to world.
        q = au.quaternion_normalize(st.imu.quaternion)
        q_inv = au.quaternion_conjugate(q)
        vel_world_raw = [odom.XVel, odom.YVel, odom.ZVel]
        vel_body_raw = au.quaternion_rotate_vector(q_inv, vel_world_raw)

        vel_body_filt = self._filter_velocity(vel_body_raw, int(timestamp_ms))
        vel_world_filt = au.quaternion_rotate_vector(q, list(vel_body_filt))

        return BigReddogVel(odom, vel_world_filt, vel_body_filt,
                            vel_world_raw, vel_body_raw)
