"""Sensor base class and observation-frame corrections.

Python port of ``FusionEstimator/SensorBase.{h,cpp}``.

A ``Sensors`` instance binds one physical sensor to one Kalman
``EstimatorPort`` (``StateSpaceModel``) and provides helpers that rotate
raw sensor observations into the world/estimation frame before they are
handed to the filter.

In the C++ code ``Est_Quaternion`` / ``Est_QuaternionInv`` are file-scope
globals shared by every sensor. Here that shared body orientation lives on
a :class:`SharedState` object handed to all sensors of one estimator core,
which keeps multiple estimator instances independent.
"""

from . import array_utils as au


class SharedState:
    """Body orientation shared across all sensors of one estimator core.

    ``est_quaternion`` is the current estimated body attitude
    ``[w, x, y, z]``; ``est_quaternion_inv`` is its conjugate.
    """

    def __init__(self):
        self.est_quaternion = [1.0, 0.0, 0.0, 0.0]
        self.est_quaternion_inv = [1.0, 0.0, 0.0, 0.0]


class Sensors:
    def __init__(self, state_space_model, shared):
        self.StateSpaceModel = state_space_model
        self.shared = shared

        self.SensorPosition = [0.0, 0.0, 0.0]
        self.SensorQuaternion = [1.0, 0.0, 0.0, 0.0]
        self.SensorQuaternionInv = [1.0, 0.0, 0.0, 0.0]

        self.Observation = [0.0] * 9
        self.ObservationTime = 0.0
        self.R_diag = [1.0] * 9

        # Mirror the C++ constructor: copy R_diag onto the model's R diagonal.
        for i in range(self.StateSpaceModel.Nz):
            self.StateSpaceModel.Matrix_R[i, i] = self.R_diag[i]

    # ------------------------------------------------------------------
    def SensorDataHandle(self, message, time):  # pragma: no cover - virtual
        raise NotImplementedError

    # ------------------------------------------------------------------
    def UpdateEst_Quaternion(self):
        """Recompute shared body quaternion from the model's Euler state."""
        euler = [
            self.StateSpaceModel.EstimatedState[0],
            self.StateSpaceModel.EstimatedState[3],
            self.StateSpaceModel.EstimatedState[6],
        ]
        self.shared.est_quaternion = au.eulerZYX_to_quaternion(euler)
        self.shared.est_quaternion_inv = au.quaternion_conjugate(
            self.shared.est_quaternion)

    # ------------------------------------------------------------------
    # Observation frame corrections. Each operates in-place on
    # self.Observation, whose layout is [p0,v0,a0, p1,v1,a1, p2,v2,a2].
    # ------------------------------------------------------------------
    def ObservationCorrect_Position(self):
        obs = self.Observation
        eq = self.shared.est_quaternion

        v = [obs[0], obs[3], obs[6]]
        v = au.quaternion_rotate_vector(self.SensorQuaternion, v)
        v = au.quaternion_rotate_vector(eq, v)

        sp = au.quaternion_rotate_vector(eq, list(self.SensorPosition))

        obs[0] = v[0] + sp[0]
        obs[3] = v[1] + sp[1]
        obs[6] = v[2] + sp[2]

    def ObservationCorrect_Velocity(self):
        obs = self.Observation
        eq = self.shared.est_quaternion

        v = [obs[1], obs[4], obs[7]]
        v = au.quaternion_rotate_vector(self.SensorQuaternion, v)
        v = au.quaternion_rotate_vector(eq, v)

        sensor_world_pos = au.quaternion_rotate_vector(eq, list(self.SensorPosition))

        body_ang_vel = [obs[1], obs[4], obs[7]]
        sensor_world_vel = au.vector_cross(body_ang_vel, sensor_world_pos)

        v[0] -= sensor_world_vel[0]
        v[1] -= sensor_world_vel[1]
        v[2] -= sensor_world_vel[2]

        obs[1] = -v[0]
        obs[4] = -v[1]
        obs[7] = -v[2]

    def ObservationCorrect_Acceleration(self):
        obs = self.Observation
        eq = self.shared.est_quaternion

        v = [obs[2], obs[5], obs[8]]
        v = au.quaternion_rotate_vector(self.SensorQuaternion, v)
        v = au.quaternion_rotate_vector(eq, v)

        sensor_pos = list(self.SensorPosition)
        ang_vel = [obs[1], obs[4], obs[7]]

        tmp = au.vector_cross(ang_vel, sensor_pos)
        tmp = au.vector_cross(ang_vel, tmp)

        v[0] -= tmp[0]
        v[1] -= tmp[1]
        v[2] -= tmp[2]

        obs[2] = v[0]
        obs[5] = v[1]
        obs[8] = v[2]

    def ObservationCorrect_Orientation(self):
        obs = self.Observation
        euler = [obs[0], obs[3], obs[6]]

        q1 = au.eulerZYX_to_quaternion(euler)
        q2 = au.quaternion_multiplication(q1, self.SensorQuaternionInv)
        q2 = au.quaternion_normalize(q2)
        e = au.quaternion_to_eulerZYX(q2)

        obs[0] = e[0]
        obs[3] = e[1]
        obs[6] = e[2]

    def ObservationCorrect_AngularVelocity(self):
        obs = self.Observation
        eq = self.shared.est_quaternion

        v = [obs[1], obs[4], obs[7]]
        v = au.quaternion_rotate_vector(self.SensorQuaternion, v)
        v = au.quaternion_rotate_vector(eq, v)

        obs[1] = v[0]
        obs[4] = v[1]
        obs[7] = v[2]

    def ObservationCorrect_AngularAcceleration(self):
        obs = self.Observation
        eq = self.shared.est_quaternion

        v = [obs[2], obs[5], obs[8]]
        v = au.quaternion_rotate_vector(self.SensorQuaternion, v)
        v = au.quaternion_rotate_vector(eq, v)

        obs[2] = v[0]
        obs[5] = v[1]
        obs[8] = v[2]
