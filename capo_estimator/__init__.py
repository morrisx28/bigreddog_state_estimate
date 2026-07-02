"""capo_estimator: pure-Python, ROS-free proprioceptive legged odometry.

A faithful port of the estimator core of
`CAPO-LeggedRobotOdometry <https://github.com/ShineMinxing/CAPO-LeggedRobotOdometry>`_
(the ``FusionEstimator/`` directory), depending only on numpy.

It estimates 6-DoF body odometry from IMU (orientation + gyro +
accelerometer) and joint motor data (position, velocity, torque) alone —
no cameras or LiDAR, no ROS.

Quick start
-----------
>>> from capo_estimator import FusionEstimatorCore, LowlevelState
>>> est = FusionEstimatorCore()
>>> st = LowlevelState()
>>> # fill st.imu.* and st.motorState[*] then:
>>> odom = est.fusion_estimator(st)
"""

from .lowlevel_state import (
    LowlevelState, IMU, MotorState, Odometer, MOTOR_NUM,
)
from .fusion_estimator import (
    FusionEstimatorCore, CreateRobot_Estimation, ConfigIndex,
)
from .big_reddog import BigReddogStateEstimator, BigReddogVel
from .pineapple_v2 import PineappleV2StateEstimator, PineappleV2Vel

__all__ = [
    "FusionEstimatorCore",
    "CreateRobot_Estimation",
    "ConfigIndex",
    "LowlevelState",
    "IMU",
    "MotorState",
    "Odometer",
    "MOTOR_NUM",
    "BigReddogStateEstimator",
    "BigReddogVel",
    "PineappleV2StateEstimator",
    "PineappleV2Vel",
]

__version__ = "1.0.0"
