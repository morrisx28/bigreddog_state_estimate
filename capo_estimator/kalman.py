"""Linear Kalman estimator and its state-space port.

Python port of:
  * ``Estimators/EstimatorPortN.h``        -> ``EstimatorPort``
  * ``Estimators/Estimator1001_Kalman.c``  -> ``EstimatorPort.estimate``
  * ``Estimators/StateSpaceModel_Go2.c``   -> ``make_go2_estimator``

The original C core keeps the model matrices (F, G, B, H, P, Q, R) and the
estimated state in flat ``double*`` buffers inside an ``EstimatorPortN``
struct, and runs a textbook linear Kalman filter (estimator id 1001) over
them. Here the same data lives in numpy arrays on an ``EstimatorPort``
object and the filter is the same set of matrix operations.

Two 9-state filters are used by the fusion estimator:
  * sensor[0]: linear  states [x,  vx,  ax,  y,  vy,  ay,  z,  vz,  az]
  * sensor[1]: angular states [r, vr,  ar,  p,  vp,  ap, yaw,vyaw,ayaw]

State indices 0/3/6 are "position-like", 1/4/7 "velocity-like",
2/5/8 "acceleration-like". ``Matrix_H`` is used purely as a diagonal
selector of which observation components are active on a given update.
"""

import numpy as np

NX = 9
NZ = 9
INTERVAL = 0.004  # seconds; matches StateSpaceModel_Go2_Interval


def _kinematic_F(dt):
    """Constant-acceleration transition matrix (3 decoupled axes)."""
    F = np.zeros((NX, NX), dtype=float)
    block = np.array([
        [1.0, dt, 0.5 * dt * dt],
        [0.0, 1.0, dt],
        [0.0, 0.0, 1.0],
    ])
    for axis in range(3):
        F[3 * axis:3 * axis + 3, 3 * axis:3 * axis + 3] = block
    return F


class EstimatorPort:
    """Container for one linear Kalman filter + its model matrices.

    Mirrors the fields of the C ``EstimatorPortN`` struct that the rest of
    the estimator touches. ``Double_Par`` / ``Int_Par`` are scratch/telemetry
    arrays preserved for parity with the original (the legs sensor stores
    foot geometry into ``Double_Par``).

    Parameters
    ----------
    interval : float
        Prediction time step (s) baked into the constant-acceleration
        transition matrix F. Should match the rate at which the estimator
        is updated (e.g. 1/200 for a 200 Hz loop).
    """

    def __init__(self, interval=INTERVAL):
        self.Nx = NX
        self.Nz = NZ
        self.Interval = interval

        # x_{i+1} = F x_i (+ w);  z_i = H x_i (+ v)
        self.Matrix_F = _kinematic_F(interval)
        self.Matrix_G = _kinematic_F(interval)
        self.Matrix_B = np.zeros(NX, dtype=float)
        self.Matrix_H = np.zeros((NX, NX), dtype=float)
        self.Matrix_P = np.eye(NX, dtype=float)
        self.Matrix_Q = np.eye(NX, dtype=float)
        self.Matrix_R = np.eye(NZ, dtype=float)

        self.EstimatedState = np.zeros(NX, dtype=float)
        self.PredictedState = np.zeros(NX, dtype=float)
        self.CurrentObservation = np.zeros(NZ, dtype=float)
        self.PredictedObservation = np.zeros(NZ, dtype=float)

        self.Int_Par = [0] * 100
        self.Double_Par = [0.0] * 100

        self._FT = self.Matrix_F.T.copy()

    def estimate(self, observation):
        """Run one Kalman update with the given observation vector.

        Equivalent to ``StateSpaceModel_Go2_EstimatorPort`` followed by
        ``Estimator1001_Estimation``.
        """
        self.CurrentObservation = np.asarray(observation, dtype=float).copy()

        F = self.Matrix_F
        FT = self._FT
        H = self.Matrix_H
        Q = self.Matrix_Q
        R = self.Matrix_R
        P = self.Matrix_P
        Xe = self.EstimatedState
        Z = self.CurrentObservation

        # State prediction
        Xe = F @ Xe

        # Covariance prediction
        P_pre = F @ P @ FT + Q

        # Kalman gain:  K = P_pre H^T (H P_pre H^T + R)^-1
        ZX1 = H @ P_pre                      # (Nz x Nx)
        S = ZX1 @ H.T + R                    # innovation covariance
        K = P_pre @ H.T @ np.linalg.inv(S)

        # State update
        Xe = Xe + K @ (Z - H @ Xe)

        # Covariance update:  P = P_pre - K (H P_pre)
        P = P_pre - K @ ZX1

        self.EstimatedState = Xe
        self.Matrix_P = P


def make_go2_estimator(interval=INTERVAL):
    """Create an estimator initialized like ``StateSpaceModel_Go2``.

    All of F/G follow the constant-acceleration model; P, Q, R start as
    identity (the original initializes them to identity and the sensors do
    not override the noise diagonals). ``interval`` sets the prediction
    time step (default matches the original 0.004 s / 250 Hz).
    """
    return EstimatorPort(interval=interval)
