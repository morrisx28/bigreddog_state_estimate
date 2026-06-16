"""Top-level proprioceptive odometry estimator.

Python port of ``FusionEstimator/fusion_estimator.h`` (the
``FusionEstimatorCore`` class plus the ``ConfigIndex`` enum).

Typical use::

    from capo_estimator import FusionEstimatorCore, LowlevelState

    est = FusionEstimatorCore()
    st = LowlevelState()
    # ... fill st.imu and st.motorState from your sensors ...
    odom = est.fusion_estimator(st)

Runtime configuration / reset is done through :meth:`fusion_estimator_status`
using a 100-element list indexed by :class:`ConfigIndex`, exactly as in the
C++ API.
"""

from . import array_utils as au
from .kalman import make_go2_estimator
from .lowlevel_state import LowlevelState, Odometer
from .sensor_base import SharedState
from .sensor_imu import SensorIMUAcc, SensorIMUMagGyro
from .sensor_legs import SensorLegsPos, SensorLegsOri


class ConfigIndex:
    IndexInOrOut = 0
    IndexStatusOK = 1
    IndexIMUAccEnable = 2
    IndexIMUQuaternionEnable = 3
    IndexIMUGyroEnable = 4
    IndexJointsXYZEnable = 5
    IndexJointsVelocityXYZEnable = 6
    IndexJointsRPYEnable = 7
    IndexSlopeModeTimeThreshold = 8
    IndexSlopeModeAngleThreshold = 9
    IndexLegFootForceThreshold = 10
    IndexLegMinStairHeight = 11
    IndexStairHeightFogotten = 12
    IndexLegOrientationInitialWeight = 13
    IndexLegOrientationTimeWeight = 14
    IndexSlopeEstimationEnable = 15


CI = ConfigIndex


class FusionEstimatorCore:
    """Two coupled linear Kalman filters fed by IMU + leg kinematics.

    ``sensors[0]`` estimates linear position/velocity/acceleration;
    ``sensors[1]`` estimates orientation/angular-rate. The IMU and leg
    sensors push observations into these filters each ``fusion_estimator``
    call.
    """

    def __init__(self, dt=0.004):
        """``dt`` is the prediction time step (s) for both Kalman filters;
        set it to match the update rate (1/200 = 0.005 for a 200 Hz loop).
        Defaults to the original 0.004 s (250 Hz)."""
        self.shared = SharedState()
        self.sensors = [make_go2_estimator(interval=dt),
                        make_go2_estimator(interval=dt)]

        self.imu_acc = SensorIMUAcc(self.sensors[0], self.shared)
        self.imu_gyro = SensorIMUMagGyro(self.sensors[1], self.shared)
        self.legs_pos = SensorLegsPos(self.sensors[0], self.shared)
        self.legs_ori = SensorLegsOri(self.sensors[1], self.shared)
        self.legs_ori.SetLegsPosRef(self.legs_pos)

        self.yaw_correct = 0.0

        # statics from fusion_estimator()
        self._last_used_timestamp = 0.0
        self._start_timestamp = 0.0

        # static last[3][48] from Signal_Available_Check
        self._sig_last = [[0.0] * 48 for _ in range(3)]
        self._sig_number = [9, 9, 48]

    # ------------------------------------------------------------------
    def fusion_estimator_status(self, status):
        """Read/modify estimator parameters or trigger control actions.

        ``status`` is a mutable list of length >= 100, indexed by
        :class:`ConfigIndex`. The dispatch on ``status[IndexInOrOut]``
        matches the C++ implementation.
        """
        mode = status[CI.IndexInOrOut]

        if mode == 1:
            status[CI.IndexInOrOut] = 0
            status[CI.IndexStatusOK] = status[CI.IndexStatusOK] + 1

            if not (status[CI.IndexIMUAccEnable] or status[CI.IndexIMUQuaternionEnable]
                    or status[CI.IndexIMUGyroEnable] or status[CI.IndexJointsXYZEnable]
                    or status[CI.IndexJointsRPYEnable]):
                status[CI.IndexStatusOK] = -999

            self.imu_acc.IMUAccEnable = bool(status[CI.IndexIMUAccEnable])
            self.imu_gyro.IMUQuaternionEnable = bool(status[CI.IndexIMUQuaternionEnable])
            self.imu_gyro.IMUGyroEnable = bool(status[CI.IndexIMUGyroEnable])
            self.legs_pos.JointsXYZEnable = bool(status[CI.IndexJointsXYZEnable])
            self.legs_pos.JointsXYZVelocityEnable = bool(status[CI.IndexJointsVelocityXYZEnable])
            self.legs_ori.JointsRPYEnable = bool(status[CI.IndexJointsRPYEnable])

            self.legs_pos.SlopeModeTimeThreshold = status[CI.IndexSlopeModeTimeThreshold]
            self.legs_pos.SlopeModeAngleThreshold = status[CI.IndexSlopeModeAngleThreshold]
            self.legs_pos.FootEffortThreshold = status[CI.IndexLegFootForceThreshold]
            self.legs_pos.Environement_Height_Scope = status[CI.IndexLegMinStairHeight]
            self.legs_pos.Data_Fading_Time = status[CI.IndexStairHeightFogotten]

            self.legs_ori.legori_init_weight = status[CI.IndexLegOrientationInitialWeight]
            self.legs_ori.legori_time_weight = status[CI.IndexLegOrientationTimeWeight]

            self.legs_pos.SlopeModeEnable = bool(status[CI.IndexSlopeEstimationEnable])

        elif mode == 2:
            status[CI.IndexInOrOut] = 0
            status[CI.IndexStatusOK] = status[CI.IndexStatusOK] + 10
            if status[CI.IndexStatusOK] > 999:
                status[CI.IndexStatusOK] = 1

            status[CI.IndexIMUAccEnable] = self.imu_acc.IMUAccEnable
            status[CI.IndexIMUQuaternionEnable] = self.imu_gyro.IMUQuaternionEnable
            status[CI.IndexIMUGyroEnable] = self.imu_gyro.IMUGyroEnable
            status[CI.IndexJointsXYZEnable] = self.legs_pos.JointsXYZEnable
            status[CI.IndexJointsVelocityXYZEnable] = self.legs_pos.JointsXYZVelocityEnable
            status[CI.IndexJointsRPYEnable] = self.legs_ori.JointsRPYEnable

            status[CI.IndexSlopeModeTimeThreshold] = self.legs_pos.SlopeModeTimeThreshold
            status[CI.IndexSlopeModeAngleThreshold] = self.legs_pos.SlopeModeAngleThreshold
            status[CI.IndexLegFootForceThreshold] = self.legs_pos.FootEffortThreshold
            status[CI.IndexLegMinStairHeight] = self.legs_pos.Environement_Height_Scope
            status[CI.IndexStairHeightFogotten] = self.legs_pos.Data_Fading_Time

            status[CI.IndexLegOrientationInitialWeight] = self.legs_ori.legori_init_weight
            status[CI.IndexLegOrientationTimeWeight] = self.legs_ori.legori_time_weight

            status[CI.IndexSlopeEstimationEnable] = self.legs_pos.SlopeModeEnable

            for i in range(80):
                status[20 + i] = self.sensors[0].Double_Par[i]

        elif mode == 3:
            # reset position to zero & re-align yaw correction
            status[CI.IndexInOrOut] = 0
            status[CI.IndexStatusOK] = status[CI.IndexStatusOK] + 20
            if status[CI.IndexStatusOK] > 999:
                status[CI.IndexStatusOK] = 1
            self.sensors[0].EstimatedState[0] = 0.0
            self.sensors[0].EstimatedState[3] = 0.0
            self.yaw_correct = self.yaw_correct - self.sensors[1].EstimatedState[6]
            for i in range(4):
                self.legs_pos.FootfallPositionRecordIsInitiated[i] = False

        elif mode == 4:
            status[CI.IndexInOrOut] = 0
            status[CI.IndexStatusOK] = self._bump(status[CI.IndexStatusOK], 40)
            self.legs_pos.UseMP()
        elif mode == 5:
            status[CI.IndexInOrOut] = 0
            status[CI.IndexStatusOK] = self._bump(status[CI.IndexStatusOK], 60)
            self.legs_pos.UseLW()
        elif mode == 6:
            status[CI.IndexInOrOut] = 0
            status[CI.IndexStatusOK] = self._bump(status[CI.IndexStatusOK], 80)
            self.legs_pos.UseMW()
        elif mode == 7:
            status[CI.IndexInOrOut] = 0
            status[CI.IndexStatusOK] = self._bump(status[CI.IndexStatusOK], 100)
            self.legs_pos.UseSP()
        elif mode == 98:
            status[CI.IndexInOrOut] = 0
            status[CI.IndexStatusOK] = self._bump(status[CI.IndexStatusOK], 98)
            self.legs_pos.UseGo2W()
        elif mode == 99:
            status[CI.IndexInOrOut] = 0
            status[CI.IndexStatusOK] = self._bump(status[CI.IndexStatusOK], 99)
            self.legs_pos.UseGo2P()

    @staticmethod
    def _bump(value, inc):
        value = value + inc
        if value > 999:
            value = 1
        return value

    # ------------------------------------------------------------------
    def _signal_available_check(self, signal, type_):
        last = self._sig_last[type_]
        number = self._sig_number[type_]
        diff = False
        for i in range(number):
            if not (signal[i] < 9999.0 and signal[i] > -9999.0):
                return False
            if signal[i] != last[i]:
                diff = True
        if not diff:
            return False
        for i in range(number):
            last[i] = signal[i]
        return True

    # ------------------------------------------------------------------
    def fusion_estimator(self, st):
        """Run one estimation step and return an :class:`Odometer`."""
        odom = Odometer()

        q = [float(st.imu.quaternion[i]) for i in range(4)]

        is_ok = au.quaternion_check(q)
        if (not is_ok) or (not self.imu_gyro.IMUQuaternionEnable):
            if not self.legs_ori.JointsRPYEnable:
                return odom
            q = [1.0, 0.0, 0.0, 0.0]
        else:
            q = au.quaternion_normalize(q)

        current_timestamp = 1e-3 * float(st.imu.timestamp)

        delta = current_timestamp - self._start_timestamp - self._last_used_timestamp
        if not (delta < 1) or not (delta > 0):
            self._start_timestamp = current_timestamp - self._last_used_timestamp

        used_timestamp = current_timestamp - self._start_timestamp
        self._last_used_timestamp = used_timestamp

        # --- IMU acceleration ---
        if self.imu_acc.IMUAccEnable:
            msg_acc = [0.0] * 9
            msg_acc[3 * 0 + 2] = float(st.imu.accelerometer[0])
            msg_acc[3 * 1 + 2] = float(st.imu.accelerometer[1])
            msg_acc[3 * 2 + 2] = float(st.imu.accelerometer[2])
            if self._signal_available_check(msg_acc, 0):
                self.imu_acc.SensorDataHandle(msg_acc, used_timestamp)

        # --- IMU orientation + gyro ---
        euler = au.quaternion_to_eulerZYX(q)
        roll, pitch, yaw = euler[0], euler[1], euler[2]

        msg_rpy = [0.0] * 9
        msg_rpy[3 * 0] = roll
        msg_rpy[3 * 1] = pitch
        msg_rpy[3 * 2] = yaw + self.yaw_correct

        if self.imu_gyro.IMUGyroEnable:
            msg_rpy[3 * 0 + 1] = float(st.imu.gyroscope[0])
            msg_rpy[3 * 1 + 1] = float(st.imu.gyroscope[1])
            msg_rpy[3 * 2 + 1] = float(st.imu.gyroscope[2])

        if self._signal_available_check(msg_rpy, 1):
            self.imu_gyro.SensorDataHandle(msg_rpy, used_timestamp)

        # --- Leg kinematics ---
        if self.legs_pos.JointsXYZEnable or self.legs_pos.JointsXYZVelocityEnable:
            joint = [0.0] * 48
            for i in range(16):
                m = st.motorState[i]
                joint[0 + i] = float(m.q)
                joint[16 + i] = float(m.dq)
                joint[32 + i] = float(m.tauEst)

            if self._signal_available_check(joint, 2) or self.legs_pos.CalculateWeightEnable:
                self.legs_pos.SensorDataHandle(joint, used_timestamp)
                self.legs_pos.LoadedWeightCheck(joint, used_timestamp)

                if self.legs_ori.JointsRPYEnable:
                    last_yaw = self.sensors[1].EstimatedState[6] - self.yaw_correct
                    self.legs_ori.SensorDataHandle(joint, used_timestamp)
                    self.yaw_correct = self.legs_ori.legori_correct - last_yaw

        # --- Pack output ---
        s0 = self.sensors[0].EstimatedState
        s1 = self.sensors[1].EstimatedState

        odom.XPos, odom.YPos, odom.ZPos = s0[0], s0[3], s0[6]
        odom.XVel, odom.YVel, odom.ZVel = s0[1], s0[4], s0[7]
        odom.XAcc, odom.YAcc, odom.ZAcc = s0[2], s0[5], s0[8]

        odom.RollRad, odom.PitchRad, odom.YawRad = s1[0], s1[3], s1[6]
        odom.RollVel, odom.PitchVel, odom.YawVel = s1[1], s1[4], s1[7]
        odom.RollAcc, odom.PitchAcc, odom.YawAcc = s1[2], s1[5], s1[8]

        fa = self.legs_pos.FootfallAveragePosition
        odom.FootfallAverageX, odom.FootfallAverageY, odom.FootfallAverageYaw = fa[0], fa[1], fa[2]

        odom.LoadedWeight = self.legs_pos.TimelyWeight

        fp = self.legs_pos.FootfallProbability
        odom.FLFootLanded, odom.FRFootLanded = fp[0], fp[1]
        odom.RLFootLanded, odom.RRFootLanded = fp[2], fp[3]

        return odom


def CreateRobot_Estimation(dt=0.004):
    """Factory mirroring the C++ ``CreateRobot_Estimation()``."""
    return FusionEstimatorCore(dt=dt)
