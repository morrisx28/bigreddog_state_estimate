"""Scalar array / quaternion helpers.

Faithful Python port of the ``array_*`` functions from
``FusionEstimator/Estimators/matrix.c`` of CAPO-LeggedRobotOdometry.

All quaternions are ``[w, x, y, z]`` and all Euler angles are ZYX
``[roll, pitch, yaw]`` (radians), matching the C++ conventions.

The functions operate on / return plain Python lists or numpy arrays of
floats. They are kept deliberately close to the original implementation
(same formulas, same edge-case handling) so the estimator reproduces the
C++ numerics.
"""

import math

PI = math.pi


def vector_cross(a, b):
    """Return a x b for 3D vectors."""
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def quaternion_check(q):
    """Return True if ``q`` is (approximately) a unit quaternion."""
    n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
    return math.isfinite(n2) and abs(n2 - 1.0) < 1.0e-2


def quaternion_conjugate(q):
    return [q[0], -q[1], -q[2], -q[3]]


def quaternion_multiplication(a, b):
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def quaternion_normalize(q):
    """Normalize ``q``; falls back to identity for degenerate input."""
    n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
    if (not math.isfinite(n2)) or n2 <= 1.0e-24:
        return [1.0, 0.0, 0.0, 0.0]
    inv_n = 1.0 / math.sqrt(n2)
    return [q[0] * inv_n, q[1] * inv_n, q[2] * inv_n, q[3] * inv_n]


def quaternion_rotate_vector(q, v):
    """Rotate 3D vector ``v`` by unit quaternion ``q``."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[0], v[1], v[2]

    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    return [
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    ]


def eulerZYX_to_quaternion(euler):
    """ZYX Euler ``[roll, pitch, yaw]`` -> quaternion ``[w, x, y, z]``."""
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    cr = math.cos(0.5 * roll)
    sr = math.sin(0.5 * roll)
    cp = math.cos(0.5 * pitch)
    sp = math.sin(0.5 * pitch)
    cy = math.cos(0.5 * yaw)
    sy = math.sin(0.5 * yaw)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def quaternion_to_eulerZYX(q):
    """Quaternion ``[w, x, y, z]`` -> ZYX Euler ``[roll, pitch, yaw]``."""
    w, x, y, z = q[0], q[1], q[2], q[3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    sinp = 2.0 * (w * y - z * x)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

    roll = math.atan2(sinr_cosp, cosr_cosp)
    if sinp >= 1.0:
        pitch = 0.5 * PI
    elif sinp <= -1.0:
        pitch = -0.5 * PI
    else:
        pitch = math.asin(sinp)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def angle_wrap(a):
    """Wrap a single angle from (-inf, +inf) into [-PI, PI]."""
    if a > PI or a < -PI:
        a_org = a
        a = math.fmod(a + PI, 2.0 * PI)
        if a < 0.0:
            a += 2.0 * PI
        a -= PI
        if a == -PI and a_org > 0.0:
            a = PI
    return a


def angle_unwrap(a, last, turn):
    """Unwrap angle ``a`` given previous wrapped angle and turn counter.

    Returns ``(unwrapped, new_last, new_turn)``.
    """
    diff = last - a
    if diff > PI:
        turn += 1.0
    elif diff < -PI:
        turn -= 1.0
    return a + turn * 2.0 * PI, a, turn


def mat3_inverse(A):
    """Invert a 3x3 (list-of-lists). Returns ``(invA, ok)``.

    ``invA`` is None when the matrix is singular (|det| < 1e-12).
    """
    a00, a01, a02 = A[0][0], A[0][1], A[0][2]
    a10, a11, a12 = A[1][0], A[1][1], A[1][2]
    a20, a21, a22 = A[2][0], A[2][1], A[2][2]

    c00 = a11 * a22 - a12 * a21
    c01 = -(a10 * a22 - a12 * a20)
    c02 = a10 * a21 - a11 * a20
    c10 = -(a01 * a22 - a02 * a21)
    c11 = a00 * a22 - a02 * a20
    c12 = -(a00 * a21 - a01 * a20)
    c20 = a01 * a12 - a02 * a11
    c21 = -(a00 * a12 - a02 * a10)
    c22 = a00 * a11 - a01 * a10

    det = a00 * c00 + a01 * c01 + a02 * c02
    if abs(det) < 1.0e-12:
        return None, False

    invdet = 1.0 / det
    inv = [
        [c00 * invdet, c10 * invdet, c20 * invdet],
        [c01 * invdet, c11 * invdet, c21 * invdet],
        [c02 * invdet, c12 * invdet, c22 * invdet],
    ]
    return inv, True


def mat3_multiply_vector(A, v):
    return [
        A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
        A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
        A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2],
    ]
