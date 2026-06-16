"""IMU sensors.

Python port of ``FusionEstimator/Sensor_IMU.{h,cpp}``.

  * :class:`SensorIMUAcc`     feeds body acceleration into the *linear*
    filter (after rotating to world frame and removing gravity).
  * :class:`SensorIMUMagGyro` feeds orientation (quaternion) and gyro
    angular rate into the *angular* filter.

The small fixed 1.7-degree pitch mounting correction and the empirical
9.43 m/s^2 gravity term are kept exactly as in the original.
"""

import math

from . import array_utils as au
from .sensor_base import Sensors

_GRAVITY = 9.43  # empirical, matches the C++ source


def _mounting_quaternion():
    angle = [0.0, 1.7 * math.pi / 180.0, 0.0]
    q = au.eulerZYX_to_quaternion(angle)
    return au.quaternion_normalize(q)


class SensorIMUAcc(Sensors):
    def __init__(self, state_space_model, shared):
        super().__init__(state_space_model, shared)
        self.SensorQuaternion = _mounting_quaternion()
        self.SensorQuaternionInv = au.quaternion_conjugate(self.SensorQuaternion)
        self.SensorPosition = [0.0, 0.0, 0.0]
        self.IMUAccEnable = True

    def SensorDataHandle(self, message, time):
        if not self.IMUAccEnable:
            return

        self.ObservationTime = time
        nz = self.StateSpaceModel.Nz
        self.Observation = [float(message[i]) for i in range(nz)]

        H = self.StateSpaceModel.Matrix_H
        for i in range(3):
            H[3 * i + 0, 3 * i + 0] = 0.0
            H[3 * i + 1, 3 * i + 1] = 0.0
            H[3 * i + 2, 3 * i + 2] = 1.0

        self.ObservationCorrect_Acceleration()
        self.Observation[8] -= _GRAVITY

        self.StateSpaceModel.estimate(self.Observation)


class SensorIMUMagGyro(Sensors):
    def __init__(self, state_space_model, shared):
        super().__init__(state_space_model, shared)
        self.SensorQuaternion = _mounting_quaternion()
        self.SensorQuaternionInv = au.quaternion_conjugate(self.SensorQuaternion)
        self.SensorPosition = [0.0, 0.0, 0.0]
        self.IMUQuaternionEnable = True
        self.IMUGyroEnable = True

        # static ObservationAngleLast / ObservationAngleTurn in the C++ source
        self._angle_last = [0.0, 0.0, 0.0]
        self._angle_turn = [0.0, 0.0, 0.0]

    def SensorDataHandle(self, message, time):
        self.ObservationTime = time
        nz = self.StateSpaceModel.Nz
        self.Observation = [float(message[i]) for i in range(nz)]
        obs = self.Observation
        H = self.StateSpaceModel.Matrix_H

        if (not self.IMUQuaternionEnable) and (not self.IMUGyroEnable):
            self.StateSpaceModel.EstimatedState[0] = obs[0]
            self.StateSpaceModel.EstimatedState[3] = obs[3]
            self.StateSpaceModel.EstimatedState[6] = obs[6]
            self.UpdateEst_Quaternion()

        for i in range(9):
            H[i, i] = 0.0

        if self.IMUQuaternionEnable:
            for i in range(3):
                H[3 * i + 0, 3 * i + 0] = 1.0
            self.ObservationCorrect_Orientation()

        if self.IMUGyroEnable:
            for i in range(3):
                H[3 * i + 1, 3 * i + 1] = 1.0
            self.ObservationCorrect_AngularVelocity()

        # Unwrap the orientation observation (roll/pitch/yaw) over time.
        angle = [obs[0], obs[3], obs[6]]
        for k in range(3):
            angle[k], self._angle_last[k], self._angle_turn[k] = au.angle_unwrap(
                angle[k], self._angle_last[k], self._angle_turn[k])
        obs[0] = angle[0]
        obs[3] = angle[1]
        obs[6] = angle[2]

        self.StateSpaceModel.estimate(self.Observation)
        self.UpdateEst_Quaternion()
