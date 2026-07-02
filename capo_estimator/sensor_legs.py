"""Leg kinematics, contact anchoring, slope/height & yaw correction.

Python port of ``FusionEstimator/Sensor_Legs.{h,cpp}``.

Two sensors live here:

  * :class:`SensorLegsPos` -- forward kinematics from joint angles to foot
    positions/velocities, contact detection from Jacobian-estimated foot
    forces, contact anchoring (footfall recording), stair-height
    stabilization and heading/slope estimation. It feeds world-frame body
    position/velocity observations into the *linear* filter.

  * :class:`SensorLegsOri` -- optional kinematics-based yaw correction from
    multi-contact geometric consistency, applied to the *angular* filter.

Robot-specific kinematics come from the ``Use*`` presets, which are
direct transcriptions of the C++ leg chains (point-foot Go2P / MP / SP and
wheel-foot LW / MW / Go2W).
"""

import math

from . import array_utils as au
from .sensor_base import Sensors

MAX_CONTACT_CHAIN = 4
MAX_CHAIN_NODE = 12
MAX_PITCH_SUM_JOINT = 8

TF_AXIS_FIXED = -1
TF_AXIS_X = 0
TF_AXIS_Y = 1
TF_AXIS_Z = 2

M_PI = math.pi


class TFNode:
    """One transform node in a leg kinematic chain."""

    __slots__ = ("parent", "q_index", "dq_index", "tau_index", "axis",
                 "t", "q_fix")

    def __init__(self, parent=-1, q_index=-1, dq_index=-1, tau_index=-1,
                 axis=TF_AXIS_FIXED, x=0.0, y=0.0, z=0.0,
                 roll=0.0, pitch=0.0, yaw=0.0):
        self.parent = parent
        self.q_index = q_index
        self.dq_index = dq_index
        self.tau_index = tau_index
        self.axis = axis
        self.t = [x, y, z]
        q = au.eulerZYX_to_quaternion([roll, pitch, yaw])
        self.q_fix = au.quaternion_normalize(q)


class LegTFChain:
    """Kinematic description of one leg / contact chain."""

    def __init__(self):
        self.node_num = 0
        self.node = [TFNode() for _ in range(MAX_CHAIN_NODE)]
        self.ee = TFNode()
        self.wheel_radius = 0.0
        self.wheel_q_index = -1
        self.wheel_dq_index = -1
        self.pitch_joint_num = 0
        self.pitch_q_index = [-1] * MAX_PITCH_SUM_JOINT
        self.pitch_dq_index = [-1] * MAX_PITCH_SUM_JOINT
        self.roll_joint_num = 0
        self.roll_q_index = [-1] * MAX_PITCH_SUM_JOINT
        self.node_pos_wf = [[0.0, 0.0, 0.0] for _ in range(MAX_CHAIN_NODE)]
        self.node_quat_wf = [[0.0, 0.0, 0.0, 0.0] for _ in range(MAX_CHAIN_NODE)]


class SensorLegsPos(Sensors):
    def __init__(self, state_space_model, shared):
        super().__init__(state_space_model, shared)

        self.ContactChainNum = 4

        self.JointsXYZEnable = True
        self.JointsXYZVelocityEnable = True

        n = MAX_CONTACT_CHAIN
        self.FootfallPositionRecordIsInitiated = [False] * n
        self.FootIsOnGround = [True] * n
        self.FootWasOnGround = [True] * n
        self.FootLastMotion = [True] * n
        self.FootLanding = [False] * n
        self.CalculateWeightEnable = False

        self.FootBodyEff_WF = [[0.0, 0.0, 0.0] for _ in range(n)]
        self.FootBodyPos_WF = [[0.0, 0.0, 0.0] for _ in range(n)]
        self.FootBodyVel_WF = [[0.0, 0.0, 0.0] for _ in range(n)]
        self.FootfallPositionRecord = [[0.0, 0.0, 0.0, 0.0] for _ in range(n)]
        self.FootfallAveragePosition = [0.0, 0.0, 0.0]
        self.FootfallProbability = [0.0] * n
        self.WheelAnglePrev = [0.0] * n

        self.FootEffortThreshold = -80.0
        self.Environement_Height_Scope = 0.08
        self.Data_Fading_Time = 1200.0
        self.MinimumWeight = 25.0
        self.TimelyWeight = 25.0

        self.SlopeModeEnable = True
        self.SlopeModeTimeThreshold = 1.0
        self.SlopeModeAngleThreshold = 5.0 / 180.0 * M_PI
        self.SlopeModeStepHeightThreshold = 0.03
        self.SlopeModeFootForceAccept = 0.5

        self.LegChains_ = [LegTFChain() for _ in range(MAX_CONTACT_CHAIN)]

        # --- persistent state replacing C++ function-local statics ---
        self._yaw_ff_last = 0.0
        self._yaw_ff_turn = 0.0

        # LoadedWeightCheck statics
        self._WIN_T = 50
        self._STABLE_N = 100
        self._buf100 = [0.0] * self._WIN_T
        self._buf100_i = 0
        self._buf100_n = 0
        self._sum100 = 0.0
        self._stable_cnt = 0

        # FootFallPositionRecord statics
        self._ShankPitchPrev = [0.0] * MAX_CONTACT_CHAIN
        self._ShankRollPrev = [0.0] * MAX_CONTACT_CHAIN
        self._MapHeightStore = [[0.0] * 1000 for _ in range(3)]
        self._MapHeightStoreMax = 0

        self.UseGo2P()

    # ==================================================================
    # Kinematic presets (transcribed from Sensor_Legs.h)
    # ==================================================================
    def _reset_chains(self):
        self.LegChains_ = [LegTFChain() for _ in range(MAX_CONTACT_CHAIN)]

    def _set_point_foot_chains(self, hip_x, hip_y, thigh_y, calf_z, foot_z):
        """Build 4 symmetric 3-DOF point-foot chains.

        Layout matches the C++ presets: node0 hip (axis X), node1 thigh
        (axis Y), node2 calf (axis Y), ee fixed foot.
        """
        # joint index bases per leg: FL,FR,RL,RR
        bases = [(0, 16, 32), (4, 20, 36), (8, 24, 40), (12, 28, 44)]
        # sign of x (front/rear) and y (left/right)
        signs = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]
        for leg in range(4):
            qb, dqb, tb = bases[leg]
            sx, sy = signs[leg]
            c = self.LegChains_[leg]
            c.node_num = 3
            c.node[0] = TFNode(-1, qb + 0, dqb + 0, tb + 0, TF_AXIS_X,
                               sx * hip_x, sy * hip_y, 0.0, 0.0, 0.0, 0.0)
            c.node[1] = TFNode(0, qb + 1, dqb + 1, tb + 1, TF_AXIS_Y,
                               0.0, sy * thigh_y, 0.0, 0.0, 0.0, 0.0)
            c.node[2] = TFNode(1, qb + 2, dqb + 2, tb + 2, TF_AXIS_Y,
                               0.0, 0.0, calf_z, 0.0, 0.0, 0.0)
            c.ee = TFNode(2, -1, -1, -1, TF_AXIS_FIXED,
                          0.0, 0.0, foot_z, 0.0, 0.0, 0.0)
            c.wheel_radius = 0.0

    def _set_wheel_foot_chains(self, hip_x, hip_y, hip_z, thigh_y, calf_z,
                               foot_z, wheel_radius):
        bases = [(0, 16, 32), (4, 20, 36), (8, 24, 40), (12, 28, 44)]
        signs = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]
        for leg in range(4):
            qb, dqb, tb = bases[leg]
            sx, sy = signs[leg]
            c = self.LegChains_[leg]
            c.node_num = 3
            c.node[0] = TFNode(-1, qb + 0, dqb + 0, tb + 0, TF_AXIS_X,
                               sx * hip_x, sy * hip_y, hip_z, 0.0, 0.0, 0.0)
            c.node[1] = TFNode(0, qb + 1, dqb + 1, tb + 1, TF_AXIS_Y,
                               0.0, sy * thigh_y, 0.0, 0.0, 0.0, 0.0)
            c.node[2] = TFNode(1, qb + 2, dqb + 2, tb + 2, TF_AXIS_Y,
                               0.0, 0.0, calf_z, 0.0, 0.0, 0.0)
            c.ee = TFNode(2, -1, -1, -1, TF_AXIS_FIXED,
                          0.0, 0.0, foot_z, 0.0, 0.0, 0.0)
            c.wheel_radius = wheel_radius
            c.wheel_q_index = c.node[0].q_index + 3
            c.wheel_dq_index = c.node[0].dq_index + 3
            c.pitch_joint_num = 2
            c.pitch_q_index[0] = c.node[1].q_index
            c.pitch_q_index[1] = c.node[2].q_index
            c.pitch_dq_index[0] = c.node[1].dq_index
            c.pitch_dq_index[1] = c.node[2].dq_index
            c.roll_joint_num = 1
            c.roll_q_index[0] = c.node[0].q_index

    def UseGo2P(self):
        self._reset_chains()
        self.Environement_Height_Scope = 0.08
        self.FootEffortThreshold = -1.0
        self._set_point_foot_chains(0.1934, 0.0465, 0.0955, -0.2130, -0.2350)

    def UseMP(self):
        self._reset_chains()
        self.Environement_Height_Scope = 0.08
        self.FootEffortThreshold = -80.0
        self._set_point_foot_chains(0.2878, 0.0700, 0.1709, -0.2600, -0.2900)

    def UseSP(self):
        self._reset_chains()
        self.Environement_Height_Scope = 0.08
        self.FootEffortThreshold = -40.0
        self._set_point_foot_chains(0.22495, 0.06800, 0.13145, -0.2200, -0.2640)

    def UseLW(self):
        self._reset_chains()
        self.Environement_Height_Scope = 0.10
        self.FootEffortThreshold = -125.0
        self._set_wheel_foot_chains(0.3405, 0.1000, -0.0666, 0.1522,
                                    -0.2700, -0.3510, 0.195 / 2.0)

    def UseMW(self):
        self._reset_chains()
        self.Environement_Height_Scope = 0.05
        self.FootEffortThreshold = -85.0
        self._set_wheel_foot_chains(0.2878, 0.0700, 0.0000, 0.1709,
                                    -0.2600, -0.2600, 0.195 / 2.0)

    def UseGo2W(self):
        self._reset_chains()
        self.Environement_Height_Scope = 0.08
        self.FootEffortThreshold = -30.0
        self.SlopeModeTimeThreshold = 0.25
        self.SlopeModeAngleThreshold = 15.0 / 180.0 * M_PI
        self.SlopeModeStepHeightThreshold = 0.08
        self.SlopeModeFootForceAccept = 0.2
        self._set_wheel_foot_chains(0.1934, 0.0465, 0.0000, 0.0955,
                                    -0.2130, -0.2130, 0.172 / 2.0)

    def UseBigReddog(self):
        """Big Reddog quadruped (point foot), from its URDF.

        Leg slots match the SDK ``joint_sdk_names`` order FL, FR, RL, RR
        (joints 0-2, 3-5, 6-8, 9-11). Each chain is hip(X) -> thigh(Y) ->
        calf(Y) -> fixed foot, with joint-axis origins composed from the
        URDF fixed+revolute offsets. Thigh and calf links are 0.21 m.
        """
        self._reset_chains()
        self.Environement_Height_Scope = 0.08
        self.FootEffortThreshold = -30.0  # torque-based; tune on hardware

        X, Y, F = TF_AXIS_X, TF_AXIS_Y, TF_AXIS_FIXED

        # FL (slot 0): q/dq/tau = 0/16/32 ..
        c = self.LegChains_[0]
        c.node_num = 3
        c.node[0] = TFNode(-1, 0, 16, 32, X, 0.206488, 0.065500, -0.021)
        c.node[1] = TFNode(0, 1, 17, 33, Y, 0.058719, 0.066367, 0.0)
        c.node[2] = TFNode(1, 2, 18, 34, Y, 0.0, 0.036500, -0.21)
        c.ee = TFNode(2, -1, -1, -1, F, 0.0, 0.017500, -0.21)
        c.wheel_radius = 0.0

        # FR (slot 1): q/dq/tau = 4/20/36 ..
        c = self.LegChains_[1]
        c.node_num = 3
        c.node[0] = TFNode(-1, 4, 20, 36, X, 0.206500, -0.065500, -0.021)
        c.node[1] = TFNode(0, 5, 21, 37, Y, 0.058707, -0.066633, 0.0)
        c.node[2] = TFNode(1, 6, 22, 38, Y, 0.0, -0.036500, -0.21)
        c.ee = TFNode(2, -1, -1, -1, F, 0.0, -0.017500, -0.21)
        c.wheel_radius = 0.0

        # RL (slot 2): q/dq/tau = 8/24/40 ..
        c = self.LegChains_[2]
        c.node_num = 3
        c.node[0] = TFNode(-1, 8, 24, 40, X, -0.206500, 0.065500, -0.021)
        c.node[1] = TFNode(0, 9, 25, 41, Y, -0.058731, 0.066367, 0.0)
        c.node[2] = TFNode(1, 10, 26, 42, Y, 0.000028746, 0.036500, -0.21)
        c.ee = TFNode(2, -1, -1, -1, F, 0.0, 0.017500, -0.21)
        c.wheel_radius = 0.0

        # RR (slot 3): q/dq/tau = 12/28/44 ..
        c = self.LegChains_[3]
        c.node_num = 3
        c.node[0] = TFNode(-1, 12, 28, 44, X, -0.205000, -0.065500, -0.021)
        c.node[1] = TFNode(0, 13, 29, 45, Y, -0.060231, -0.066500, 0.0)
        c.node[2] = TFNode(1, 14, 30, 46, Y, 0.0, -0.036500, -0.21)
        c.ee = TFNode(2, -1, -1, -1, F, 0.0, -0.017500, -0.21)
        c.wheel_radius = 0.0

    def UsePineappleV2(self):
        """Pineapple V2 wheeled biped (2 legs), from its URDF.

        Only TWO contact chains exist (``ContactChainNum = 2``): the left leg
        uses joint slots 0-3 and the right leg slots 4-7 of the 16-slot joint
        layout. Each chain is hip(X) -> thigh(Y) -> calf(Y) -> wheel, with the
        end effector at the wheel centre; rolling contact is handled by the
        per-chain wheel-odometry term (``wheel_radius * wheel_speed``, with the
        shank pitch/roll compensation already in :meth:`FootFallPositionRecord`).

        Offsets are the joint origins from
        ``controller/robot/pineapple_v2/urdf/pineapple_v2.urdf``. The wheel
        radius (0.077 m) is the wheel collision-cylinder radius. Joint slot
        order matches the controller's xml order
        ``[L_hip, L_thigh, L_calf, L_wheel, R_hip, R_thigh, R_calf, R_wheel]``.
        ``FootEffortThreshold`` is torque-derived and MUST be tuned on hardware
        (a wheeled biped keeps both wheels loaded almost continuously).
        """
        self._reset_chains()
        self.ContactChainNum = 2
        self.Environement_Height_Scope = 0.05
        self.FootEffortThreshold = -15.0  # torque-based; tune on hardware

        X, Y, FX = TF_AXIS_X, TF_AXIS_Y, TF_AXIS_FIXED
        WHEEL_R = 0.077

        # Left leg: q/dq/tau slots = 0-3 / 16-19 / 32-35
        c = self.LegChains_[0]
        c.node_num = 3
        c.node[0] = TFNode(-1, 0, 16, 32, X, 0.086334, 0.050823, 0.056)
        c.node[1] = TFNode(0, 1, 17, 33, Y, -0.066166, 0.063177, 0.0)
        c.node[2] = TFNode(1, 2, 18, 34, Y, 0.0, 0.056000, -0.19)
        c.ee = TFNode(2, -1, -1, -1, FX, 0.0, 0.035500, -0.19)
        c.wheel_radius = WHEEL_R
        c.wheel_q_index = 3
        c.wheel_dq_index = 19
        c.pitch_joint_num = 2
        c.pitch_q_index[0] = 1
        c.pitch_q_index[1] = 2
        c.pitch_dq_index[0] = 17
        c.pitch_dq_index[1] = 18
        c.roll_joint_num = 1
        c.roll_q_index[0] = 0

        # Right leg: q/dq/tau slots = 4-7 / 20-23 / 36-39
        c = self.LegChains_[1]
        c.node_num = 3
        c.node[0] = TFNode(-1, 4, 20, 36, X, 0.086334, -0.050823, 0.056)
        c.node[1] = TFNode(0, 5, 21, 37, Y, -0.066166, -0.063177, 0.0)
        c.node[2] = TFNode(1, 6, 22, 38, Y, 0.0, -0.056000, -0.19)
        c.ee = TFNode(2, -1, -1, -1, FX, 0.0, -0.035500, -0.19)
        c.wheel_radius = WHEEL_R
        c.wheel_q_index = 7
        c.wheel_dq_index = 23
        c.pitch_joint_num = 2
        c.pitch_q_index[0] = 5
        c.pitch_q_index[1] = 6
        c.pitch_dq_index[0] = 21
        c.pitch_dq_index[1] = 22
        c.roll_joint_num = 1
        c.roll_q_index[0] = 4

    # ==================================================================
    def SensorDataHandle(self, message, time):
        if (not self.JointsXYZEnable) and (not self.JointsXYZVelocityEnable):
            return

        ssm = self.StateSpaceModel
        self.ObservationTime = time

        for i in range(ssm.Nz):
            self.Observation[i] = 0.0

        for LegNumber in range(self.ContactChainNum):
            self.FootBodyEff_WF[LegNumber] = [0.0, 0.0, 0.0]
            self.SensorPosition = list(self.LegChains_[LegNumber].node[0].t)

            self.Joint2HipFoot(message, LegNumber)

            base = 12 + LegNumber * 12
            chain = self.LegChains_[LegNumber]
            for i in range(3):
                ssm.Double_Par[base + 0 * 3 + i] = chain.node_pos_wf[0][i]
                ssm.Double_Par[base + 1 * 3 + i] = chain.node_pos_wf[1][i]
                ssm.Double_Par[base + 2 * 3 + i] = chain.node_pos_wf[2][i]
                ssm.Double_Par[base + 3 * 3 + i] = self.FootBodyPos_WF[LegNumber][i]

            if self.JointsXYZEnable:
                for i in range(3):
                    self.FootBodyPos_WF[LegNumber][i] = self.Observation[3 * i]
            if self.JointsXYZVelocityEnable:
                for i in range(3):
                    self.FootBodyVel_WF[LegNumber][i] = self.Observation[3 * i + 1]

        if (self.FootIsOnGround[0] or self.FootIsOnGround[1]
                or self.FootIsOnGround[2] or self.FootIsOnGround[3]):
            H = ssm.Matrix_H
            for i in range(9):
                H[i, i] = 0.0

            self.FootFallPositionRecord(message)

            if self.JointsXYZEnable:
                for i in range(3):
                    H[3 * i + 0, 3 * i + 0] = 1.0
            if self.JointsXYZVelocityEnable:
                for i in range(3):
                    H[3 * i + 1, 3 * i + 1] = 1.0

            ssm.estimate(self.Observation)

            p_w = [[0.0, 0.0] for _ in range(MAX_CONTACT_CHAIN)]
            for LegNumber in range(self.ContactChainNum):
                if self.FootIsOnGround[LegNumber]:
                    p_w[LegNumber][0] = self.FootfallPositionRecord[LegNumber][0]
                    p_w[LegNumber][1] = self.FootfallPositionRecord[LegNumber][1]
                else:
                    p_w[LegNumber][0] = ssm.EstimatedState[0] + self.FootBodyPos_WF[LegNumber][0]
                    p_w[LegNumber][1] = ssm.EstimatedState[3] + self.FootBodyPos_WF[LegNumber][1]

            if self.ContactChainNum == 4:
                fx = 0.5 * (p_w[0][0] + p_w[1][0])
                fy = 0.5 * (p_w[0][1] + p_w[1][1])
                rx = 0.5 * (p_w[2][0] + p_w[3][0])
                ry = 0.5 * (p_w[2][1] + p_w[3][1])
                x_mean = 0.5 * (fx + rx)
                y_mean = 0.5 * (fy + ry)
                yaw_ff = math.atan2(fy - ry, fx - rx)
                yaw_ff, self._yaw_ff_last, self._yaw_ff_turn = au.angle_unwrap(
                    yaw_ff, self._yaw_ff_last, self._yaw_ff_turn)
                self.FootfallAveragePosition[0] = x_mean
                self.FootfallAveragePosition[1] = y_mean
                self.FootfallAveragePosition[2] = yaw_ff
            else:
                x_sum = 0.0
                y_sum = 0.0
                cnt = 0
                for LegNumber in range(self.ContactChainNum):
                    x_sum += p_w[LegNumber][0]
                    y_sum += p_w[LegNumber][1]
                    cnt += 1
                self.FootfallAveragePosition[0] = x_sum / cnt
                self.FootfallAveragePosition[1] = y_sum / cnt
                self.FootfallAveragePosition[2] = 0.0

    # ==================================================================
    def LoadedWeightCheck(self, message, time):
        WIN_T = self._WIN_T
        STABLE_N = self._STABLE_N

        all_on_ground = (self.FootfallProbability[0] + self.FootfallProbability[1]
                         + self.FootfallProbability[2] + self.FootfallProbability[3]) > 2.5

        if all_on_ground:
            if self._stable_cnt < STABLE_N:
                self._stable_cnt += 1
        else:
            self._stable_cnt = 0
            self._buf100_i = 0
            self._buf100_n = 0
            self._sum100 = 0.0

        if self._stable_cnt >= STABLE_N:
            fz_sum = (self.FootBodyEff_WF[0][2] + self.FootBodyEff_WF[1][2]
                      + self.FootBodyEff_WF[2][2] + self.FootBodyEff_WF[3][2])

            if self._buf100_n < WIN_T:
                self._buf100[self._buf100_i] = fz_sum
                self._sum100 += fz_sum
                self._buf100_n += 1
            else:
                self._sum100 -= self._buf100[self._buf100_i]
                self._buf100[self._buf100_i] = fz_sum
                self._sum100 += fz_sum
            self._buf100_i += 1
            if self._buf100_i >= WIN_T:
                self._buf100_i = 0

            mean100 = (self._sum100 / self._buf100_n) if self._buf100_n > 0 else 0.0
            self.TimelyWeight = -mean100 * 0.1
            if self.TimelyWeight < self.MinimumWeight:
                self.TimelyWeight = self.MinimumWeight

    # ==================================================================
    def Joint2HipFoot(self, message, LegNumber):
        chain = self.LegChains_[LegNumber]
        ssm = self.StateSpaceModel
        eq = self.shared.est_quaternion

        joint_org = [[0.0, 0.0, 0.0] for _ in range(MAX_CHAIN_NODE)]
        joint_axis = [[0.0, 0.0, 0.0] for _ in range(MAX_CHAIN_NODE)]
        joint_dq = [0.0] * MAX_CHAIN_NODE
        joint_tau = [0.0] * MAX_CHAIN_NODE
        joint_num = 0

        p_zero = [0.0, 0.0, 0.0]

        for n in range(chain.node_num):
            node = chain.node[n]
            if node.parent < 0:
                p_parent = p_zero
                q_parent = eq
            else:
                p_parent = chain.node_pos_wf[node.parent]
                q_parent = chain.node_quat_wf[node.parent]

            pos = au.quaternion_rotate_vector(q_parent, node.t)
            pos[0] += p_parent[0]
            pos[1] += p_parent[1]
            pos[2] += p_parent[2]
            chain.node_pos_wf[n] = pos

            q_pre = au.quaternion_multiplication(q_parent, node.q_fix)
            q_pre = au.quaternion_normalize(q_pre)

            if node.q_index >= 0:
                axis_local = [
                    1.0 if node.axis == TF_AXIS_X else 0.0,
                    1.0 if node.axis == TF_AXIS_Y else 0.0,
                    1.0 if node.axis == TF_AXIS_Z else 0.0,
                ]

                joint_org[joint_num] = [pos[0], pos[1], pos[2]]
                joint_axis[joint_num] = au.quaternion_rotate_vector(q_pre, axis_local)

                joint_dq[joint_num] = (message[node.dq_index]
                                       if node.dq_index >= 0 else 0.0)
                joint_tau[joint_num] = (message[node.tau_index]
                                        if node.tau_index >= 0 else 0.0)

                h = 0.5 * message[node.q_index]
                s = math.sin(h)
                q_joint = [math.cos(h), axis_local[0] * s,
                           axis_local[1] * s, axis_local[2] * s]

                qn = au.quaternion_multiplication(q_pre, q_joint)
                chain.node_quat_wf[n] = au.quaternion_normalize(qn)
                joint_num += 1
            else:
                chain.node_quat_wf[n] = list(q_pre)

        ee = chain.ee
        foot = au.quaternion_rotate_vector(chain.node_quat_wf[ee.parent], ee.t)
        foot[0] += chain.node_pos_wf[ee.parent][0]
        foot[1] += chain.node_pos_wf[ee.parent][1]
        foot[2] += chain.node_pos_wf[ee.parent][2]
        self.FootBodyPos_WF[LegNumber] = foot

        self.Observation[0] = foot[0]
        self.Observation[3] = foot[1]
        self.Observation[6] = foot[2]
        self.Observation[1] = 0.0
        self.Observation[4] = 0.0
        self.Observation[7] = 0.0

        Jtau = [0.0, 0.0, 0.0]
        JJT = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        for j in range(joint_num):
            rx = foot[0] - joint_org[j][0]
            ry = foot[1] - joint_org[j][1]
            rz = foot[2] - joint_org[j][2]

            J0 = joint_axis[j][1] * rz - joint_axis[j][2] * ry
            J1 = joint_axis[j][2] * rx - joint_axis[j][0] * rz
            J2 = joint_axis[j][0] * ry - joint_axis[j][1] * rx

            self.Observation[1] += J0 * joint_dq[j]
            self.Observation[4] += J1 * joint_dq[j]
            self.Observation[7] += J2 * joint_dq[j]

            Jtau[0] += J0 * joint_tau[j]
            Jtau[1] += J1 * joint_tau[j]
            Jtau[2] += J2 * joint_tau[j]

            JJT[0][0] += J0 * J0
            JJT[0][1] += J0 * J1
            JJT[0][2] += J0 * J2
            JJT[1][1] += J1 * J1
            JJT[1][2] += J1 * J2
            JJT[2][2] += J2 * J2

        self.Observation[1] = -self.Observation[1]
        self.Observation[4] = -self.Observation[4]
        self.Observation[7] = -self.Observation[7]

        JJT[1][0] = JJT[0][1]
        JJT[2][0] = JJT[0][2]
        JJT[2][1] = JJT[1][2]

        self.FootBodyEff_WF[LegNumber] = [0.0, 0.0, 0.0]
        JJT_inv, ok = au.mat3_inverse(JJT)
        if ok:
            self.FootBodyEff_WF[LegNumber] = au.mat3_multiply_vector(JJT_inv, Jtau)

        eff = self.FootBodyEff_WF[LegNumber]
        for i in range(3):
            ssm.Double_Par[LegNumber * 3 + i] = eff[i]

        thr = self.FootEffortThreshold
        if eff[2] >= 0.3 * thr:
            self.FootfallProbability[LegNumber] = 0.0
        elif eff[2] <= 1.3 * thr:
            self.FootfallProbability[LegNumber] = 1.0
        else:
            self.FootfallProbability[LegNumber] = (eff[2] - 0.3 * thr) / thr

        self.FootIsOnGround[LegNumber] = eff[2] < thr

        if self.FootIsOnGround[LegNumber] and not self.FootWasOnGround[LegNumber]:
            self.FootLanding[LegNumber] = True
            self.FootLastMotion[LegNumber] = True
        else:
            self.FootLanding[LegNumber] = False

        if (not self.FootIsOnGround[LegNumber]) and self.FootWasOnGround[LegNumber]:
            self.FootLastMotion[LegNumber] = False

        # Detect the "lying on the ground" posture.
        if LegNumber == self.ContactChainNum - 1:
            count = 0
            while count < self.ContactChainNum:
                if self.FootIsOnGround[count]:
                    break
                count += 1
            if count == self.ContactChainNum and self.FootBodyPos_WF[LegNumber][2] > -0.25:
                self.FootIsOnGround[LegNumber] = True

        self.FootWasOnGround[LegNumber] = self.FootIsOnGround[LegNumber]

    # ==================================================================
    def _update_shank_prev(self, message, LegNumber, body_pitch, body_roll):
        chain = self.LegChains_[LegNumber]
        if chain.wheel_q_index >= 0:
            self.WheelAnglePrev[LegNumber] = message[chain.wheel_q_index]
        else:
            self.WheelAnglePrev[LegNumber] = 0.0

        sp = body_pitch
        for k in range(chain.pitch_joint_num):
            if chain.pitch_q_index[k] >= 0:
                sp -= message[chain.pitch_q_index[k]]
        self._ShankPitchPrev[LegNumber] = sp

        sr = body_roll
        for k in range(chain.roll_joint_num):
            if chain.roll_q_index[k] >= 0:
                sr -= message[chain.roll_q_index[k]]
        self._ShankRollPrev[LegNumber] = sr

    def FootFallPositionRecord(self, message):
        ssm = self.StateSpaceModel
        store = self._MapHeightStore

        p_sum = [0.0, 0.0, 0.0]
        v_sum = [0.0, 0.0, 0.0]
        leg_cnt = 0

        body_rpy = au.quaternion_to_eulerZYX(self.shared.est_quaternion)
        body_roll = body_rpy[0]
        body_pitch = body_rpy[1]

        move_dir_x, move_dir_y, move_dir_z = self.EstimateGroundPitchAlongHeading()

        for LegNumber in range(self.ContactChainNum):
            if not self.FootIsOnGround[LegNumber]:
                continue

            rec = self.FootfallPositionRecord[LegNumber]
            chain = self.LegChains_[LegNumber]
            foot = self.FootBodyPos_WF[LegNumber]

            if not self.FootfallPositionRecordIsInitiated[LegNumber]:
                self.FootfallPositionRecordIsInitiated[LegNumber] = True
                self.FootLanding[LegNumber] = False
                rec[0] = ssm.EstimatedState[0] + foot[0]
                rec[1] = ssm.EstimatedState[3] + foot[1]
                rec[2] = 0.0
                rec[3] = self.ObservationTime
                self._update_shank_prev(message, LegNumber, body_pitch, body_roll)

            elif self.FootLanding[LegNumber]:
                self.FootLanding[LegNumber] = False
                rec[0] = ssm.EstimatedState[0] + foot[0]
                rec[1] = ssm.EstimatedState[3] + foot[1]
                rec[2] = ssm.EstimatedState[6] + foot[2]
                rec[3] = self.ObservationTime
                self._update_shank_prev(message, LegNumber, body_pitch, body_roll)

                Zdifference = 99.0
                nmax = self._MapHeightStoreMax + 1

                # Fade out stale stored steps.
                for i in range(nmax):
                    if store[2][i] != 0 and abs(self.ObservationTime - store[2][i]) > self.Data_Fading_Time:
                        store[0][i] = 0.0
                        store[1][i] = 0.0
                        store[2][i] = 0.0

                # Try to match an existing support-plane height.
                for i in range(nmax):
                    dz = abs(store[0][i] - rec[2])
                    if (dz <= self.Environement_Height_Scope and move_dir_z == 0.0) \
                            or dz <= self.SlopeModeStepHeightThreshold:
                        store[1][i] *= math.exp(-(self.ObservationTime - store[2][i]) / (10 * self.Data_Fading_Time))
                        store[1][i] += 1
                        store[2][i] = self.ObservationTime
                        if abs(store[0][i] - rec[2]) <= self.Environement_Height_Scope / 10:
                            Zdifference = 0.0
                        else:
                            Zdifference = rec[2] - store[0][i]
                        break

                if Zdifference == 99.0:
                    Zdifference = 0.0
                    n = self._MapHeightStoreMax + 1
                    i = n  # value if the loop finds no empty slot
                    for idx in range(n):
                        if store[2][idx] == 0:
                            store[0][idx] = rec[2]
                            store[1][idx] = 1.0
                            store[2][idx] = self.ObservationTime
                            i = idx
                            break
                    if i >= 999:
                        for idx in range(self._MapHeightStoreMax + 1):
                            if store[2][idx] != 0 and abs(self.ObservationTime - store[2][idx]) > 60:
                                store[0][idx] = 0.0
                                store[1][idx] = 0.0
                                store[2][idx] = 0.0
                        i = 0
                        store[0][i] = rec[2]
                        store[1][i] = 1.0
                        store[2][i] = self.ObservationTime
                    if i == self._MapHeightStoreMax + 1:
                        self._MapHeightStoreMax = i
                        store[0][i] = rec[2]
                        store[1][i] = 1.0
                        store[2][i] = self.ObservationTime

                rec[2] = rec[2] - Zdifference

            # Wheel rolling compensation.
            WheelMove = 0.0
            WheelSidewayMove = 0.0
            WheelVel = 0.0

            if (chain.wheel_radius > 0.0 and chain.wheel_q_index >= 0
                    and chain.wheel_dq_index >= 0):
                WheelRotationAngle = message[chain.wheel_q_index] - self.WheelAnglePrev[LegNumber]
                while WheelRotationAngle > M_PI:
                    WheelRotationAngle -= 2.0 * M_PI
                while WheelRotationAngle < -M_PI:
                    WheelRotationAngle += 2.0 * M_PI
                self.WheelAnglePrev[LegNumber] = message[chain.wheel_q_index]

                ShankPitchAngle = body_pitch
                ShankRollAngle = body_roll
                WheelRotationVelocityEff = message[chain.wheel_dq_index]

                for k in range(chain.pitch_joint_num):
                    if chain.pitch_q_index[k] >= 0:
                        ShankPitchAngle -= message[chain.pitch_q_index[k]]
                    if chain.pitch_dq_index[k] >= 0:
                        WheelRotationVelocityEff -= message[chain.pitch_dq_index[k]]
                for k in range(chain.roll_joint_num):
                    if chain.roll_q_index[k] >= 0:
                        ShankRollAngle -= message[chain.roll_q_index[k]]

                temp = ShankPitchAngle
                ShankPitchAngle -= self._ShankPitchPrev[LegNumber]
                while ShankPitchAngle > M_PI:
                    ShankPitchAngle -= 2.0 * M_PI
                while ShankPitchAngle < -M_PI:
                    ShankPitchAngle += 2.0 * M_PI
                self._ShankPitchPrev[LegNumber] = temp

                temp = ShankRollAngle
                ShankRollAngle -= self._ShankRollPrev[LegNumber]
                while ShankRollAngle > M_PI:
                    ShankRollAngle -= 2.0 * M_PI
                while ShankRollAngle < -M_PI:
                    ShankRollAngle += 2.0 * M_PI
                self._ShankRollPrev[LegNumber] = temp

                WheelMove = chain.wheel_radius * (WheelRotationAngle - ShankPitchAngle)
                WheelSidewayMove = chain.wheel_radius * 2.0 * math.sin(0.5 * ShankRollAngle)
                WheelVel = chain.wheel_radius * WheelRotationVelocityEff

            rec[0] += WheelMove * move_dir_x + WheelSidewayMove * (-move_dir_y)
            rec[1] += WheelMove * move_dir_y + WheelSidewayMove * (move_dir_x)
            rec[2] += WheelMove * move_dir_z

            foot = self.FootBodyPos_WF[LegNumber]
            p_sum[0] += rec[0] - foot[0]
            p_sum[1] += rec[1] - foot[1]
            p_sum[2] += rec[2] - foot[2]

            vel = self.FootBodyVel_WF[LegNumber]
            v_sum[0] += vel[0] + WheelVel * move_dir_x
            v_sum[1] += vel[1] + WheelVel * move_dir_y
            v_sum[2] += vel[2] + WheelVel * move_dir_z

            leg_cnt += 1

        # No contacting chain this step (e.g. all wheels momentarily airborne):
        # leave the position/velocity observation at zero rather than dividing
        # by zero. The H gate is set per the contact flags upstream.
        if leg_cnt == 0:
            return

        self.Observation[0] = p_sum[0] / leg_cnt
        self.Observation[3] = p_sum[1] / leg_cnt
        self.Observation[6] = p_sum[2] / leg_cnt
        self.Observation[1] = v_sum[0] / leg_cnt
        self.Observation[4] = v_sum[1] / leg_cnt
        self.Observation[7] = v_sum[2] / leg_cnt

    # ==================================================================
    def EstimateGroundPitchAlongHeading(self):
        """Fit a support plane and project heading onto it.

        Returns the unit travel direction ``(move_dir_x, y, z)``: flat
        (z == 0) on level ground or when slope mode is off / under-determined,
        otherwise inclined along the fitted slope.
        """
        eq = self.shared.est_quaternion

        sxx = sxy = syy = sx = sy = sxz = syz = sz = 0.0
        n = 0
        count = 0

        for LegNumber in range(self.ContactChainNum):
            if self.FootBodyEff_WF[LegNumber][2] >= self.FootEffortThreshold * self.SlopeModeFootForceAccept:
                continue
            if not self.FootfallPositionRecordIsInitiated[LegNumber]:
                continue
            if self.ObservationTime - self.FootfallPositionRecord[LegNumber][3] > self.SlopeModeTimeThreshold:
                count += 1

            x = self.FootBodyPos_WF[LegNumber][0]
            y = self.FootBodyPos_WF[LegNumber][1]
            z = self.FootBodyPos_WF[LegNumber][2]

            sxx += x * x
            sxy += x * y
            syy += y * y
            sx += x
            sy += y
            sxz += x * z
            syz += y * z
            sz += z
            n += 1

        A = [
            [sxx, sxy, sx],
            [sxy, syy, sy],
            [sx, sy, float(n)],
        ]
        rhs = [sxz, syz, sz]

        hx = 1.0 - 2.0 * (eq[2] * eq[2] + eq[3] * eq[3])
        hy = 2.0 * (eq[1] * eq[2] + eq[0] * eq[3])
        hn = math.sqrt(hx * hx + hy * hy)
        if hn < 1e-9:
            hx = 1.0
            hy = 0.0
            hn = 1.0
        hx /= hn
        hy /= hn

        Ainv, ok = au.mat3_inverse(A)
        if (not self.SlopeModeEnable) or n < 3 or (not ok) or count < 2:
            return hx, hy, 0.0

        abc = au.mat3_multiply_vector(Ainv, rhs)
        a = abc[0]
        b = abc[1]
        k = a * hx + b * hy

        if abs(k) <= math.tan(self.SlopeModeAngleThreshold):
            return hx, hy, 0.0

        hxy_n = 1.0 / math.sqrt(1.0 + k * k)
        hz_n = k * hxy_n
        return hx * hxy_n, hy * hxy_n, hz_n


class SensorLegsOri(Sensors):
    def __init__(self, state_space_model, shared):
        super().__init__(state_space_model, shared)
        self.legs_pos_ref_ = None
        self.legori_init_weight = 0.001
        self.legori_time_weight = 10000.0
        self.legori_current_weight = 0.0001
        self.legori_correct = 0.0
        self.JointsRPYEnable = False
        self._TimeRecord = None  # set on first call (static double TimeRecord = Time)

    def SetLegsPosRef(self, ref):
        self.legs_pos_ref_ = ref

    def SensorDataHandle(self, message, time):
        ref = self.legs_pos_ref_
        ssm = self.StateSpaceModel

        if self._TimeRecord is None:
            self._TimeRecord = time

        yaw_now = ssm.EstimatedState[6]
        self.legori_correct = yaw_now

        n_ground = 0
        for LegNumber in range(ref.ContactChainNum):
            if ref.FootIsOnGround[LegNumber]:
                n_ground += 1

        if n_ground < 2:
            return

        if n_ground < ref.ContactChainNum:
            self._TimeRecord = time
            self.legori_current_weight = self.legori_init_weight
        else:
            self.legori_current_weight = ((time - self._TimeRecord)
                                          * (1.0 - self.legori_init_weight)
                                          / self.legori_time_weight
                                          + self.legori_init_weight)
            if self.legori_current_weight > 1.0:
                self.legori_current_weight = 1.0

        q_yaw_inv = au.eulerZYX_to_quaternion([0.0, 0.0, -yaw_now])
        q_yaw_inv = au.quaternion_normalize(q_yaw_inv)

        sx = 0.0
        sy = 0.0
        for i in range(ref.ContactChainNum):
            if not ref.FootIsOnGround[i]:
                continue
            for j in range(i + 1, ref.ContactChainNum):
                if not ref.FootIsOnGround[j]:
                    continue

                v_wf = [
                    ref.FootBodyPos_WF[j][0] - ref.FootBodyPos_WF[i][0],
                    ref.FootBodyPos_WF[j][1] - ref.FootBodyPos_WF[i][1],
                    ref.FootBodyPos_WF[j][2] - ref.FootBodyPos_WF[i][2],
                ]
                v_rp = au.quaternion_rotate_vector(q_yaw_inv, v_wf)

                vw_x = ref.FootfallPositionRecord[j][0] - ref.FootfallPositionRecord[i][0]
                vw_y = ref.FootfallPositionRecord[j][1] - ref.FootfallPositionRecord[i][1]

                ang_rp = math.atan2(v_rp[1], v_rp[0])
                ang_w = math.atan2(vw_y, vw_x)

                yaw_ij = ang_w - ang_rp
                yaw_ij = au.angle_wrap(yaw_ij)

                pair_weight = ref.FootfallProbability[i] * ref.FootfallProbability[j]
                if pair_weight <= 0.0:
                    continue

                sx += pair_weight * math.cos(yaw_ij)
                sy += pair_weight * math.sin(yaw_ij)

        if sx != 0.0 or sy != 0.0:
            yaw_est = math.atan2(sy, sx)
            yaw_now = ssm.EstimatedState[6]
            err = yaw_est - yaw_now
            err = au.angle_wrap(err)
            self.legori_correct = yaw_now + self.legori_current_weight * err
            self.UpdateEst_Quaternion()
