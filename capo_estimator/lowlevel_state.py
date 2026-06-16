"""Input / output data structures.

Python port of ``FusionEstimator/LowlevelState.h``.

These mirror the C++ ``IMU``, ``MotorState``, ``Odometer`` and
``LowlevelState`` structs and are the only types you need to feed the
estimator and read its result.
"""

MOTOR_NUM = 16


class MotorState:
    """Single joint motor reading.

    Attributes
    ----------
    q : float
        Joint position (rad).
    dq : float
        Joint angular velocity (rad/s).
    ddq : float
        Joint angular acceleration (unused by the estimator).
    tauEst : float
        Estimated joint torque (N·m). For platforms with a contact
        sensor instead of torque, put the contact value here (see config
        notes in the original repo).
    """

    __slots__ = ("mode", "q", "dq", "ddq", "tauEst", "temp", "cnt")

    def __init__(self, q=0.0, dq=0.0, ddq=0.0, tauEst=0.0):
        self.mode = 0
        self.q = q
        self.dq = dq
        self.ddq = ddq
        self.tauEst = tauEst
        self.temp = 0.0
        self.cnt = 0


class IMU:
    """IMU reading.

    Attributes
    ----------
    quaternion : list[float]
        Orientation ``[w, x, y, z]``.
    gyroscope : list[float]
        Body angular velocity ``[wx, wy, wz]`` (rad/s).
    accelerometer : list[float]
        Body linear acceleration ``[ax, ay, az]`` (m/s^2).
    timestamp : int
        Timestamp in milliseconds (the C++ core multiplies by 1e-3).
    """

    __slots__ = ("quaternion", "gyroscope", "accelerometer",
                 "pitch", "roll", "yaw", "timestamp")

    def __init__(self):
        self.quaternion = [0.0, 0.0, 0.0, 0.0]
        self.gyroscope = [0.0, 0.0, 0.0]
        self.accelerometer = [0.0, 0.0, 0.0]
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self.timestamp = 0


class Odometer:
    """Estimator output (full 6-DoF odometry + diagnostics)."""

    __slots__ = (
        "XPos", "YPos", "ZPos",
        "XVel", "YVel", "ZVel",
        "XAcc", "YAcc", "ZAcc",
        "RollRad", "PitchRad", "YawRad",
        "RollVel", "PitchVel", "YawVel",
        "RollAcc", "PitchAcc", "YawAcc",
        "FootfallAverageX", "FootfallAverageY", "FootfallAverageYaw",
        "FLFootLanded", "FRFootLanded", "RLFootLanded", "RRFootLanded",
        "LoadedWeight",
    )

    def __init__(self):
        for name in self.__slots__:
            setattr(self, name, 0.0)

    def __repr__(self):
        return ("Odometer(pos=({:.3f},{:.3f},{:.3f}) "
                "rpy=({:.3f},{:.3f},{:.3f}))").format(
            self.XPos, self.YPos, self.ZPos,
            self.RollRad, self.PitchRad, self.YawRad)


class LowlevelState:
    """Bundle of one IMU sample and ``MOTOR_NUM`` motor states."""

    __slots__ = ("imu", "odometer", "motorState")

    def __init__(self):
        self.imu = IMU()
        self.odometer = Odometer()
        self.motorState = [MotorState() for _ in range(MOTOR_NUM)]
