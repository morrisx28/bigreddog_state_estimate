from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

from headless_teleop import HeadlessTeleop

# The CAPO proprioceptive estimator package lives in the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "pineapple_v2.yaml"
SIPO_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "sipo_config.yaml"
LOG_DIR = Path(__file__).resolve().parent / "logs"


@dataclass
class SimConfig:
    policy_path: str
    xml_path: str
    simulation_duration: float
    simulation_dt: float
    control_decimation: int
    kps: np.ndarray
    kds: np.ndarray
    default_angles: np.ndarray
    lin_vel_scale: float
    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float
    pos_action_scale: float
    vel_action_scale: float
    cmd_scale: np.ndarray
    num_actions: int
    num_obs: int
    one_step_obs_size: int
    obs_buffer_size: int
    leg_joint_indices: list[int]
    wheel_joint_indices: list[int]
    cmd_init: np.ndarray
    max_lin: float
    max_ang: float
    height_scale: float
    cmd_height_init: float
    min_height: float
    max_height: float
    height_step: float
    enable_height_command: bool
    policy_index_map: np.ndarray | None

    @classmethod
    def from_dict(cls, config: dict) -> "SimConfig":
        policy_index_map = config.get("policy_index_map", None)
        if policy_index_map is not None:
            policy_index_map = np.array(policy_index_map, dtype=np.int64)

        return cls(
            policy_path=config["policy_path"],
            xml_path=config["xml_path"],
            simulation_duration=config["simulation_duration"],
            simulation_dt=config["simulation_dt"],
            control_decimation=config["control_decimation"],
            kps=np.array(config["kps"], dtype=np.float32),
            kds=np.array(config["kds"], dtype=np.float32),
            default_angles=np.array(config["default_angles"], dtype=np.float32),
            lin_vel_scale=config["lin_vel_scale"],
            ang_vel_scale=config["ang_vel_scale"],
            dof_pos_scale=config["dof_pos_scale"],
            dof_vel_scale=config["dof_vel_scale"],
            pos_action_scale=config["pos_action_scale"],
            vel_action_scale=config["vel_action_scale"],
            cmd_scale=np.array(config["cmd_scale"], dtype=np.float32),
            num_actions=config["num_actions"],
            num_obs=config["num_obs"],
            one_step_obs_size=config["one_step_obs_size"],
            obs_buffer_size=config.get("obs_buffer_size", 1),
            leg_joint_indices=config["leg_joint_indices"],
            wheel_joint_indices=config["wheel_joint_indices"],
            cmd_init=np.array(config["cmd_init"], dtype=np.float32),
            max_lin=config.get("max_lin_vel", 1.0),
            max_ang=config.get("max_ang_vel", 1.0),
            height_scale=config.get("height_scale", 1.0),
            cmd_height_init=config.get("cmd_height_init", 0.3),
            min_height=config.get("min_height", 0.2),
            max_height=config.get("max_height", 0.35),
            height_step=config.get("height_step", 0.005),
            enable_height_command=config.get("enable_height_command", True),
            policy_index_map=policy_index_map,
        )


@dataclass
class HistoryBuffers:
    lin_vel: list[np.ndarray] = field(default_factory=list)
    ang_vel: list[np.ndarray] = field(default_factory=list)
    gravity_b: list[np.ndarray] = field(default_factory=list)
    joint_pos: list[np.ndarray] = field(default_factory=list)
    joint_vel: list[np.ndarray] = field(default_factory=list)
    action: list[np.ndarray] = field(default_factory=list)
    time: list[float] = field(default_factory=list)
    cmd: list[np.ndarray] = field(default_factory=list)
    tau: list[np.ndarray] = field(default_factory=list)


@dataclass
class SipoBuffers:
    pos: list[np.ndarray] = field(default_factory=list)
    vel: list[np.ndarray] = field(default_factory=list)
    quat: list[np.ndarray] = field(default_factory=list)
    vel_body: list[np.ndarray] = field(default_factory=list)
    gt_pos: list[np.ndarray] = field(default_factory=list)
    gt_vel: list[np.ndarray] = field(default_factory=list)
    gt_quat: list[np.ndarray] = field(default_factory=list)
    gt_vel_body: list[np.ndarray] = field(default_factory=list)
    imu_yaw_rate: list[float] = field(default_factory=list)
    yaw_rate: list[float] = field(default_factory=list)
    gt_yaw_rate: list[float] = field(default_factory=list)
    wheel_radius: float = 0.0


@dataclass
class SipoRunner:
    sipo: Any
    get_contact_states: Callable
    buffers: SipoBuffers
    base_body_id: int

    @classmethod
    def create(
        cls,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: SimConfig,
        base_body_id: int,
    ) -> "SipoRunner":
        # from mujoco_sipo_v3 import SIPO, get_contact_states
        from mujoco_sipo_v3_new import SIPO, get_contact_states
        sipo = SIPO(config.xml_path, config_path=str(SIPO_CONFIG_PATH))
        mujoco.mj_forward(model, data)

        init_qpos_sense = data.qpos[7 : 7 + config.num_actions].copy()
        init_qvel_sense = data.qvel[6 : 6 + config.num_actions].copy()
        z_kin = sipo.get_kinematics(init_qpos_sense, init_qvel_sense)
        feet_pos_flat = z_kin.reshape(sipo.num_legs, sipo.fk_stride)[:, :3].flatten()

        sipo.init_state(
            data.xipos[base_body_id].copy(),
            data.xquat[base_body_id].copy(),
            feet_pos_flat,
        )
        print("SIPO initialized after reset.")

        buffers = SipoBuffers(wheel_radius=sipo.wheel_radius)
        return cls(sipo, get_contact_states, buffers, base_body_id)

    def update(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        qpos,
        qvel,
        imu_acc,
        ang_vel_b,
        config: SimConfig,
    ) -> None:
        contacts = self.get_contact_states(model, data, self.sipo.leg_names)
        contacts.fill(1.0)

        self.sipo.predict(imu_acc, ang_vel_b, config.simulation_dt, qpos)
        wheel_vel_meas = qvel[config.wheel_joint_indices]
        sipo_state = self.sipo.update(
            qpos,
            qvel,
            contacts,
            ang_vel_b,
            wheel_vel_meas,
            yaw_meas=None,
        )

        buffers = self.buffers
        buffers.pos.append(sipo_state[self.sipo.idx_pos].copy())
        buffers.vel.append(sipo_state[self.sipo.idx_vel].copy())
        buffers.quat.append(sipo_state[self.sipo.idx_quat].copy())
        buffers.vel_body.append(
            quat_rotate_inverse(sipo_state[self.sipo.idx_quat], sipo_state[self.sipo.idx_vel])
        )
        buffers.gt_pos.append(data.xpos[self.base_body_id].copy())
        buffers.gt_vel.append(data.cvel[self.base_body_id][3:6].copy())
        buffers.gt_quat.append(data.xquat[self.base_body_id].copy())
        buffers.gt_vel_body.append(
            quat_rotate_inverse(data.xquat[self.base_body_id], data.cvel[self.base_body_id][3:6])
        )
        buffers.imu_yaw_rate.append(ang_vel_b[2])

        bg = sipo_state[self.sipo.idx_bg]
        buffers.yaw_rate.append(ang_vel_b[2] - bg[2])

        gt_w_world = data.cvel[self.base_body_id][0:3].copy()
        gt_w_body = quat_rotate_inverse(data.xquat[self.base_body_id], gt_w_world)
        buffers.gt_yaw_rate.append(gt_w_body[2])


@dataclass
class CapoBuffers:
    est_world: list[np.ndarray] = field(default_factory=list)
    est_body: list[np.ndarray] = field(default_factory=list)
    est_pos: list[np.ndarray] = field(default_factory=list)
    gt_world: list[np.ndarray] = field(default_factory=list)
    gt_body: list[np.ndarray] = field(default_factory=list)
    gt_pos: list[np.ndarray] = field(default_factory=list)
    imu_yaw_rate: list[float] = field(default_factory=list)
    est_yaw_rate: list[float] = field(default_factory=list)
    gt_yaw_rate: list[float] = field(default_factory=list)
    contact: list[np.ndarray] = field(default_factory=list)


@dataclass
class CapoRunner:
    """Runs the CAPO proprioceptive estimator alongside the sim and records
    its base-velocity estimate against MuJoCo ground truth."""

    estimator: Any
    buffers: CapoBuffers
    base_body_id: int

    @classmethod
    def create(cls, config: SimConfig, base_body_id: int) -> "CapoRunner":
        from capo_estimator import PineappleV2StateEstimator

        estimator = PineappleV2StateEstimator(
            foot_force_threshold=-15.0,
            enable_slope=True,
            update_rate_hz=1.0 / config.simulation_dt,
        )
        print("CAPO estimator initialized.")
        return cls(estimator, CapoBuffers(), base_body_id)

    def update(
        self,
        data: mujoco.MjData,
        qpos,
        qvel,
        qtau,
        imu_quat,
        ang_vel_b,
        imu_acc,
        lin_vel_i,
        sim_time: float,
    ) -> None:
        # qpos/qvel/qtau are in xml order
        # [L_hip, L_thigh, L_calf, L_wheel, R_hip, R_thigh, R_calf, R_wheel],
        # which maps directly onto the estimator's joint slots 0-7.
        result = self.estimator.update(
            imu_quat, ang_vel_b, imu_acc, qpos, qvel, qtau, int(sim_time * 1000.0)
        )

        b = self.buffers
        b.est_world.append(np.asarray(result.lin_vel_world, dtype=np.float64))
        b.est_body.append(np.asarray(result.lin_vel_body, dtype=np.float64))
        b.est_pos.append(np.asarray(result.pos_world, dtype=np.float64))
        b.est_yaw_rate.append(float(result.odom.YawVel))
        b.contact.append(np.asarray(result.wheel_contact, dtype=np.float64))

        # Ground truth from MuJoCo (framelinvel is world-frame base velocity).
        # Copy: lin_vel_i is a live view into data.sensordata, overwritten each step.
        gt_world = np.array(lin_vel_i, dtype=np.float64)
        b.gt_world.append(gt_world)
        b.gt_body.append(quat_rotate_inverse(np.asarray(imu_quat), gt_world))
        b.gt_pos.append(data.xpos[self.base_body_id].copy())

        b.imu_yaw_rate.append(float(ang_vel_b[2]))
        gt_w_world = data.cvel[self.base_body_id][0:3].copy()
        gt_w_body = quat_rotate_inverse(data.xquat[self.base_body_id], gt_w_world)
        b.gt_yaw_rate.append(float(gt_w_body[2]))


@dataclass
class CommandStep:
    duration: float
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    height: float | None = None


class CommandSequencer:
    """Replays a fixed timed sequence of velocity/height commands."""

    def __init__(self, steps: list[CommandStep], default_height: float):
        self.steps = steps
        self.default_height = default_height
        self._boundaries: list[float] = []
        t = 0.0
        for s in steps:
            t += s.duration
            self._boundaries.append(t)

    @property
    def total_duration(self) -> float:
        return self._boundaries[-1] if self._boundaries else 0.0

    def get_command(self, sim_time: float) -> tuple[np.ndarray, float]:
        for i, step in enumerate(self.steps):
            if sim_time < self._boundaries[i]:
                height = step.height if step.height is not None else self.default_height
                return np.array([step.vx, step.vy, step.wz], dtype=np.float32), height
        last = self.steps[-1]
        height = last.height if last.height is not None else self.default_height
        return np.array([last.vx, last.vy, last.wz], dtype=np.float32), height


def load_config(config_file: str) -> SimConfig:
    with open(config_file, "r") as f:
        return SimConfig.from_dict(yaml.load(f, Loader=yaml.FullLoader))


def load_command_sequence(config_file: str, default_height: float) -> CommandSequencer | None:
    with open(config_file, "r") as f:
        raw = yaml.load(f, Loader=yaml.FullLoader)
    steps_raw = raw.get("command_sequence", None)
    if steps_raw is None:
        return None
    steps = [CommandStep(**s) for s in steps_raw]
    seq = CommandSequencer(steps, default_height)
    print(f"Command sequence loaded: {len(steps)} steps, {seq.total_duration:.1f}s total.")
    return seq


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = value.lower()
    if normalized in ("true", "1", "yes", "y", "on"):
        return True
    if normalized in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def make_teleop(config: SimConfig) -> HeadlessTeleop:
    return HeadlessTeleop(
        config_init=config.cmd_init,
        max_lin=config.max_lin,
        max_ang=config.max_ang,
        height_init=config.cmd_height_init,
        height_step=config.height_step,
        min_height=config.min_height,
        max_height=config.max_height,
    )


def quat_rotate_inverse(q, v):
    """
    Rotate a vector by the inverse of a quaternion.
    Direct translation from the PyTorch version to NumPy.
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]

    term1 = 2.0 * np.square(q_w) - 1.0
    term1_expanded = np.expand_dims(term1, axis=-1)
    a = v * term1_expanded

    q_w_expanded = np.expand_dims(q_w, axis=-1)
    b = np.cross(q_vec, v) * q_w_expanded * 2.0

    dot_product = np.sum(q_vec * v, axis=-1)
    dot_product_expanded = np.expand_dims(dot_product, axis=-1)
    c = q_vec * dot_product_expanded * 2.0

    return a - b + c


def get_gravity_orientation(quaternion):
    """Get the gravity vector in the robot base frame."""
    q = np.array(quaternion)
    gravity = np.zeros(3, dtype=np.float32)
    gravity[0] = 2 * (-q[1] * q[3] + q[0] * q[2])
    gravity[1] = -2 * (q[2] * q[3] + q[0] * q[1])
    gravity[2] = 1 - 2 * (q[0] * q[0] + q[3] * q[3])
    return gravity


def apply_diamond_constraint(cmd, max_lin, max_ang):
    """
    Apply an L1 velocity limit:
    |v_x| / v_max + |w_z| / w_max <= 1
    """
    limit_vx = max_lin if max_lin >= 1e-6 else 1.0
    limit_wz = max_ang if max_ang >= 1e-6 else 1.0

    ratio = abs(cmd[0]) / limit_vx + abs(cmd[2]) / limit_wz
    if ratio > 1.0:
        cmd *= 1.0 / ratio
    return cmd


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculate torques from position and velocity targets."""
    return (target_q - q) * kp + (target_dq - dq) * kd


def read_sensors(data: mujoco.MjData, config: SimConfig, include_imu_acc: bool = False):
    num_actions = config.num_actions
    qpos = data.sensordata[:num_actions]
    qvel = data.sensordata[num_actions : 2 * num_actions]
    # jointactuatorfrc sensors (measured joint torque), xml order.
    qtau = data.sensordata[2 * num_actions : 3 * num_actions]
    imu_quat = data.sensordata[3 * num_actions : 3 * num_actions + 4]
    ang_vel_b = data.sensordata[3 * num_actions + 4 : 3 * num_actions + 7]
    lin_vel_i = data.sensordata[3 * num_actions + 13 : 3 * num_actions + 16]
    imu_acc = data.sensordata[3 * num_actions + 7 : 3 * num_actions + 10] if include_imu_acc else None
    return qpos, qvel, qtau, imu_quat, ang_vel_b, imu_acc, lin_vel_i


def get_policy_joint_state(qpos, qvel, config: SimConfig):
    if config.policy_index_map is None:
        return qpos, qvel, config.default_angles

    return (
        qpos[config.policy_index_map],
        qvel[config.policy_index_map],
        config.default_angles[config.policy_index_map],
    )


def build_observation(
    qpos_obs,
    qvel_obs,
    default_angles_obs,
    ang_vel_b,
    gravity_b,
    cmd_vel,
    cmd_height,
    action,
    config: SimConfig,
):
    valid_leg_idx = [
        i for i in config.leg_joint_indices if i < len(qpos_obs) and i < len(default_angles_obs)
    ]
    leg_pos_delta = (qpos_obs[valid_leg_idx] - default_angles_obs[valid_leg_idx]) * config.dof_pos_scale
    leg_pos_delta = leg_pos_delta.astype(np.float32).ravel()

    obs_list = [
        ang_vel_b * config.ang_vel_scale,
        gravity_b,
        cmd_vel * config.cmd_scale,
        leg_pos_delta,
        qvel_obs * config.dof_vel_scale,
        action.astype(np.float32),
    ]
    if config.enable_height_command:
        obs_list.insert(3, cmd_height * config.height_scale)

    return obs_list


def update_observation_history(obs_history_buffer, obs_list):
    obs_tensors = [
        torch.tensor(obs, dtype=torch.float32) if isinstance(obs, np.ndarray) else obs
        for obs in obs_list
    ]
    current_obs = torch.cat(obs_tensors, dim=0)

    obs_history_buffer = torch.roll(obs_history_buffer, shifts=-1, dims=0)
    obs_history_buffer[-1] = current_obs

    split_sizes = [obs.numel() for obs in obs_tensors]
    feature_groups = torch.split(obs_history_buffer, split_sizes, dim=1)
    flat_groups = [group.flatten() for group in feature_groups]
    obs_tensor = torch.cat(flat_groups).unsqueeze(0)
    return obs_history_buffer, torch.clip(obs_tensor, -100, 100)


def apply_policy_action(action, target_dof_pos, target_dof_vel, config: SimConfig):
    for idx in config.leg_joint_indices:
        idx_xml = config.policy_index_map[idx] if config.policy_index_map is not None else idx
        if idx_xml < len(target_dof_pos) and idx < len(action):
            target_dof_pos[idx_xml] = (
                config.default_angles[idx_xml] + action[idx] * config.pos_action_scale
            )

    for idx in config.wheel_joint_indices:
        idx_xml = config.policy_index_map[idx] if config.policy_index_map is not None else idx
        if idx_xml < len(target_dof_vel) and idx < len(action):
            target_dof_vel[idx_xml] = action[idx] * config.vel_action_scale


def record_step(
    buffers: HistoryBuffers,
    lin_vel_b,
    ang_vel_b,
    gravity_b,
    qpos_obs,
    qvel_obs,
    action,
    cmd_vel,
    tau,
    counter,
    config: SimConfig,
):
    scaled_action = action.copy()
    scaled_action[config.leg_joint_indices] *= config.pos_action_scale
    scaled_action[config.wheel_joint_indices] *= config.vel_action_scale

    buffers.lin_vel.append(lin_vel_b.copy())
    buffers.ang_vel.append(ang_vel_b.copy())
    buffers.gravity_b.append(gravity_b.copy())
    buffers.joint_pos.append(qpos_obs.copy())
    buffers.joint_vel.append(qvel_obs.copy())
    buffers.action.append(scaled_action)
    buffers.time.append(counter * config.simulation_dt)
    buffers.cmd.append(cmd_vel.copy())
    buffers.tau.append(tau.copy())


def run_simulation(
    config: SimConfig,
    policy,
    teleop: HeadlessTeleop,
    enable_sipo: bool = False,
    enable_capo: bool = False,
    sequencer: CommandSequencer | None = None,
) -> tuple[HistoryBuffers, SipoBuffers | None, CapoBuffers | None]:
    target_dof_pos = config.default_angles.copy()
    target_dof_vel = np.zeros(config.num_actions)
    action = np.zeros(config.num_actions, dtype=np.float32)
    obs_history_buffer = torch.zeros((config.obs_buffer_size, config.one_step_obs_size))
    buffers = HistoryBuffers()

    model = mujoco.MjModel.from_xml_path(config.xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = config.simulation_dt
    base_body_id = 1
    sipo_runner = None
    capo_runner = None
    counter = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        viewer.sync()
        if enable_sipo:
            sipo_runner = SipoRunner.create(model, data, config, base_body_id)
        if enable_capo:
            capo_runner = CapoRunner.create(config, base_body_id)

        while viewer.is_running() and counter * config.simulation_dt < config.simulation_duration:
            step_start = time.time()

            qpos_for_pd = data.sensordata[: config.num_actions]
            qvel_for_pd = data.sensordata[config.num_actions : 2 * config.num_actions]
            tau = pd_control(
                target_dof_pos,
                qpos_for_pd,
                config.kps,
                target_dof_vel,
                qvel_for_pd,
                config.kds,
            )
            data.ctrl[:] = tau

            mujoco.mj_step(model, data)
            viewer.cam.lookat[:] = data.xipos[base_body_id]
            counter += 1

            qpos, qvel, qtau, imu_quat, ang_vel_b, imu_acc, lin_vel_i = read_sensors(
                data,
                config,
                include_imu_acc=(sipo_runner is not None or capo_runner is not None),
            )
            qpos_obs, qvel_obs, default_angles_obs = get_policy_joint_state(qpos, qvel, config)

            if sipo_runner is not None:
                sipo_runner.update(model, data, qpos, qvel, imu_acc, ang_vel_b, config)

            if capo_runner is not None:
                capo_runner.update(
                    data, qpos, qvel, qtau, imu_quat, ang_vel_b, imu_acc, lin_vel_i,
                    counter * config.simulation_dt,
                )

            if sequencer is not None:
                sim_time = counter * config.simulation_dt
                cmd_vel, cmd_height_val = sequencer.get_command(sim_time)
            else:
                cmd_vel = np.array(teleop.get_command(), dtype=np.float32)
                cmd_height_val = teleop.get_height_command()
            cmd_vel = apply_diamond_constraint(cmd_vel, config.max_lin, config.max_ang)
            cmd_height = np.array([cmd_height_val], dtype=np.float32)

            lin_vel_b = quat_rotate_inverse(imu_quat, lin_vel_i)
            gravity_b = get_gravity_orientation(imu_quat)

            obs_list = build_observation(
                qpos_obs,
                qvel_obs,
                default_angles_obs,
                ang_vel_b,
                gravity_b,
                cmd_vel,
                cmd_height,
                action,
                config,
            )

            record_step(
                buffers,
                lin_vel_b,
                ang_vel_b,
                gravity_b,
                qpos_obs,
                qvel_obs,
                action,
                cmd_vel,
                tau,
                counter,
                config,
            )

            if counter % config.control_decimation == 0 and counter > 0:
                obs_history_buffer, obs_tensor = update_observation_history(
                    obs_history_buffer, obs_list
                )
                action = policy(obs_tensor).detach().numpy().squeeze()
                apply_policy_action(action, target_dof_pos, target_dof_vel, config)

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    return (
        buffers,
        sipo_runner.buffers if sipo_runner is not None else None,
        capo_runner.buffers if capo_runner is not None else None,
    )


def plot_history(buffers: HistoryBuffers):
    fig_hist = plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    for i in range(3):
        plt.plot(buffers.time, [step[i] for step in buffers.lin_vel], label=f"Linear Velocity {i}")
    plt.plot(buffers.time, [step[0] for step in buffers.cmd], label="Command Velocity x", linestyle="--")
    plt.title("History Linear Velocity", fontsize=10, pad=10)
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    for i in range(3):
        plt.plot(buffers.time, [step[i] for step in buffers.ang_vel], label=f"Angular Velocity {i}")
    plt.plot(buffers.time, [step[2] for step in buffers.cmd], label="Command Velocity yaw", linestyle="--")
    plt.title("History Angular Velocity", fontsize=10, pad=10)
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 3)
    for i in (3, 7):
        plt.plot(buffers.time, [step[i] for step in buffers.tau], label=f"Joint Torque {i}")
    for i in (6, 7):
        plt.plot(buffers.time, [step[i] for step in buffers.joint_vel], label=f"Joint vel {i}")
    for i in (6, 7):
        plt.plot(buffers.time, [step[i] for step in buffers.action], label=f"Joint action {i}", linestyle="--")
    plt.title("History Joint", fontsize=10, pad=10)
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 4)
    for i in (0, 4):
        plt.plot(buffers.time, [step[i] for step in buffers.tau], label=f"Joint Torque {i}")
    for i in (0, 1):
        plt.plot(buffers.time, [step[i] for step in buffers.joint_pos], label=f"Joint pos {i}")
    for i in (0, 1):
        plt.plot(buffers.time, [step[i] for step in buffers.action], label=f"Joint action {i}", linestyle="--")
    plt.title("History Joint", fontsize=10, pad=10)
    plt.legend()
    plt.grid()

    plt.tight_layout()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LOG_DIR / "history_data.png"
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close(fig_hist)


def plot_sipo(buffers: SipoBuffers, time_data: list[float]):
    fig_sipo = plt.figure(figsize=(12, 20))
    t = time_data

    ax_traj = plt.subplot(5, 1, 1)
    ax_traj.set_title("2D Position Trajectory (XY Plane)")
    ax_traj.plot([p[0] for p in buffers.pos], [p[1] for p in buffers.pos], label="SIPO")
    ax_traj.plot(
        [p[0] for p in buffers.gt_pos],
        [p[1] for p in buffers.gt_pos],
        label="GT",
        linestyle="--",
    )
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.legend()
    ax_traj.grid()
    ax_traj.axis("equal")

    ax_vx = plt.subplot(5, 2, 3)
    ax_vx.set_title("Velocity X (World)")
    ax_vx.plot(t, [v[0] for v in buffers.vel], label="SIPO")
    ax_vx.plot(t, [v[0] for v in buffers.gt_vel], label="GT", linestyle="--")
    ax_vx.set_xlabel("Time (s)")
    ax_vx.legend()
    ax_vx.grid()

    ax_vy = plt.subplot(5, 2, 4)
    ax_vy.set_title("Velocity Y (World)")
    ax_vy.plot(t, [v[1] for v in buffers.vel], label="SIPO")
    ax_vy.plot(t, [v[1] for v in buffers.gt_vel], label="GT", linestyle="--")
    ax_vy.set_xlabel("Time (s)")
    ax_vy.legend()
    ax_vy.grid()

    ax_vbx = plt.subplot(5, 2, 5)
    ax_vbx.set_title("Velocity X (Body/Robot Frame)")
    ax_vbx.plot(t, [v[0] for v in buffers.vel_body], label="SIPO Body")
    ax_vbx.plot(t, [v[0] for v in buffers.gt_vel_body], label="GT Body", linestyle="--")
    ax_vbx.set_xlabel("Time (s)")
    ax_vbx.legend()
    ax_vbx.grid()

    ax_vby = plt.subplot(5, 2, 6)
    ax_vby.set_title("Velocity Y (Body/Robot Frame)")
    ax_vby.plot(t, [v[1] for v in buffers.vel_body], label="SIPO Body")
    ax_vby.plot(t, [v[1] for v in buffers.gt_vel_body], label="GT Body", linestyle="--")
    ax_vby.set_xlabel("Time (s)")
    ax_vby.legend()
    ax_vby.grid()

    ax_yaw = plt.subplot(5, 1, 4)
    ax_yaw.set_title("Yaw Angular Velocity (Z-axis) [Rad/s]")
    ax_yaw.plot(t, buffers.imu_yaw_rate, label="IMU Raw", color="purple", alpha=0.5, linewidth=1.0)
    ax_yaw.plot(t, buffers.yaw_rate, label="SIPO Est (Corrected)", color="blue", linewidth=1.5)
    ax_yaw.plot(t, buffers.gt_yaw_rate, label="GT Body", color="green", linestyle="--", linewidth=1.5)
    ax_yaw.set_xlabel("Time (s)")
    ax_yaw.legend()
    ax_yaw.grid()

    ax_z = plt.subplot(5, 1, 5)
    ax_z.set_title("Z Height (Position Z) [m]")
    ax_z.plot(t, [p[2] for p in buffers.pos], label="SIPO")
    ax_z.plot(t, [p[2] for p in buffers.gt_pos], label="GT", linestyle="--")
    ax_z.set_xlabel("Time (s)")
    ax_z.legend()
    ax_z.grid()

    plt.tight_layout()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LOG_DIR / "sipo_results.png"
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close(fig_sipo)


def plot_capo(buffers: CapoBuffers, time_data: list[float]):
    """Plot CAPO estimate vs MuJoCo ground truth.

    The body-frame velocity comparison is the robust one: CAPO integrates its
    own heading from the gyro, so its world frame (and hence the XY trajectory)
    can carry a constant yaw offset relative to MuJoCo world, whereas the
    body-frame velocity is frame-offset-immune.
    """
    t = time_data
    est_b = np.array(buffers.est_body, dtype=np.float64)
    gt_b = np.array(buffers.gt_body, dtype=np.float64)
    rmse_b = np.sqrt(np.mean((est_b - gt_b) ** 2, axis=0))
    print(
        f"[capo] body-frame RMSE  vx={rmse_b[0]:.3f}  vy={rmse_b[1]:.3f}  "
        f"vz={rmse_b[2]:.3f}  (m/s)"
    )

    fig_capo = plt.figure(figsize=(12, 18))
    labels = ["vx", "vy", "vz"]

    ax_traj = plt.subplot(4, 1, 1)
    ax_traj.set_title("2D Position Trajectory (XY) -- CAPO world frame may be yaw-offset from GT")
    ax_traj.plot([p[0] for p in buffers.est_pos], [p[1] for p in buffers.est_pos], label="CAPO")
    ax_traj.plot(
        [p[0] for p in buffers.gt_pos], [p[1] for p in buffers.gt_pos], label="GT", linestyle="--"
    )
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.legend()
    ax_traj.grid()
    ax_traj.axis("equal")

    for i in range(3):
        ax = plt.subplot(4, 3, 4 + i)
        ax.set_title(f"Body-frame {labels[i]} (RMSE={rmse_b[i]:.3f} m/s)")
        ax.plot(t, est_b[:, i], label="CAPO", color="red", linewidth=1.2)
        ax.plot(t, gt_b[:, i], label="GT", color="black", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.grid()

    ax_yaw = plt.subplot(4, 1, 3)
    ax_yaw.set_title("Yaw Angular Velocity (Z) [rad/s]")
    ax_yaw.plot(t, buffers.imu_yaw_rate, label="IMU Raw", color="purple", alpha=0.5, linewidth=1.0)
    ax_yaw.plot(t, buffers.est_yaw_rate, label="CAPO Est", color="blue", linewidth=1.5)
    ax_yaw.plot(t, buffers.gt_yaw_rate, label="GT Body", color="green", linestyle="--", linewidth=1.5)
    ax_yaw.set_xlabel("Time (s)")
    ax_yaw.legend()
    ax_yaw.grid()

    ax_z = plt.subplot(4, 1, 4)
    ax_z.set_title("Z Height (Position Z) [m]")
    ax_z.plot(t, [p[2] for p in buffers.est_pos], label="CAPO")
    ax_z.plot(t, [p[2] for p in buffers.gt_pos], label="GT", linestyle="--")
    ax_z.set_xlabel("Time (s)")
    ax_z.legend()
    ax_z.grid()

    plt.tight_layout()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LOG_DIR / "capo_results.png"
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close(fig_capo)


def plot_comparison(sipo: SipoBuffers, capo: CapoBuffers, time_data: list[float]):
    """Head-to-head SIPO vs CAPO against the SAME ground-truth reference.

    The body-frame velocity is the fair metric: both estimators integrate
    their own heading, so a world-frame / trajectory comparison would bake in
    each one's yaw offset. We use the exact MuJoCo base-body ground truth
    recorded by the SIPO runner (data.cvel / data.xquat, noise-free) as the
    single reference and score both estimators against it, so the RMSE numbers
    are directly comparable.
    """
    # Align lengths defensively (both should already match the sim step count).
    n = min(len(sipo.vel_body), len(capo.est_body), len(time_data))
    t = np.asarray(time_data[:n])
    sipo_b = np.array(sipo.vel_body[:n], dtype=np.float64)
    capo_b = np.array(capo.est_body[:n], dtype=np.float64)
    gt_b = np.array(sipo.gt_vel_body[:n], dtype=np.float64)

    rmse_sipo = np.sqrt(np.mean((sipo_b - gt_b) ** 2, axis=0))
    rmse_capo = np.sqrt(np.mean((capo_b - gt_b) ** 2, axis=0))

    sipo_yaw = np.asarray(sipo.yaw_rate[:n], dtype=np.float64)
    capo_yaw = np.asarray(capo.est_yaw_rate[:n], dtype=np.float64)
    gt_yaw = np.asarray(sipo.gt_yaw_rate[:n], dtype=np.float64)
    rmse_yaw_sipo = float(np.sqrt(np.mean((sipo_yaw - gt_yaw) ** 2)))
    rmse_yaw_capo = float(np.sqrt(np.mean((capo_yaw - gt_yaw) ** 2)))

    print("[compare] body-frame velocity RMSE (vs MuJoCo GT, m/s)")
    print(f"[compare]   SIPO  vx={rmse_sipo[0]:.3f}  vy={rmse_sipo[1]:.3f}  vz={rmse_sipo[2]:.3f}")
    print(f"[compare]   CAPO  vx={rmse_capo[0]:.3f}  vy={rmse_capo[1]:.3f}  vz={rmse_capo[2]:.3f}")
    print(f"[compare] yaw-rate RMSE (rad/s)  SIPO={rmse_yaw_sipo:.3f}  CAPO={rmse_yaw_capo:.3f}")

    labels = ["vx", "vy", "vz"]
    fig = plt.figure(figsize=(14, 12))

    for i in range(3):
        ax = plt.subplot(4, 1, i + 1)
        ax.set_title(
            f"Body-frame {labels[i]}  |  RMSE: SIPO={rmse_sipo[i]:.3f}  "
            f"CAPO={rmse_capo[i]:.3f} m/s"
        )
        ax.plot(t, gt_b[:, i], label="GT", color="black", linestyle="--", linewidth=1.6)
        ax.plot(t, sipo_b[:, i], label="SIPO", color="tab:blue", linewidth=1.2)
        ax.plot(t, capo_b[:, i], label="CAPO", color="tab:red", linewidth=1.2)
        ax.set_ylabel(f"{labels[i]} [m/s]")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    ax_yaw = plt.subplot(4, 1, 4)
    ax_yaw.set_title(
        f"Yaw rate  |  RMSE: SIPO={rmse_yaw_sipo:.3f}  CAPO={rmse_yaw_capo:.3f} rad/s"
    )
    ax_yaw.plot(t, gt_yaw, label="GT", color="black", linestyle="--", linewidth=1.6)
    ax_yaw.plot(t, sipo_yaw, label="SIPO", color="tab:blue", linewidth=1.2)
    ax_yaw.plot(t, capo_yaw, label="CAPO", color="tab:red", linewidth=1.2)
    ax_yaw.plot(t, np.asarray(sipo.imu_yaw_rate[:n]), label="IMU raw",
                color="0.6", alpha=0.6, linewidth=0.9)
    ax_yaw.set_ylabel("yaw rate [rad/s]")
    ax_yaw.set_xlabel("Time (s)")
    ax_yaw.legend(loc="upper right")
    ax_yaw.grid(True, alpha=0.3)

    fig.suptitle("SIPO vs CAPO base-velocity estimation (body frame) vs MuJoCo ground truth")
    plt.tight_layout()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LOG_DIR / "sipo_vs_capo.png"
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="path to the yaml config file",
    )
    parser.add_argument(
        "--sipo",
        type=parse_bool,
        default=False,
        help="enable SIPO estimation and result plotting (true/false)",
    )
    parser.add_argument(
        "--capo",
        type=parse_bool,
        default=False,
        help="enable CAPO estimation and result plotting (true/false)",
    )
    parser.add_argument(
        "--seq",
        type=parse_bool,
        default=False,
        help="run scripted command sequence from config instead of teleop (true/false)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    sequencer = load_command_sequence(args.config, config.cmd_height_init) if args.seq else None
    policy = torch.jit.load(config.policy_path)
    teleop = make_teleop(config)
    if sequencer is None:
        print("Headless teleop active.")
    else:
        print("Running scripted command sequence (keyboard/gamepad ignored).")

    try:
        buffers, sipo_buffers, capo_buffers = run_simulation(
            config, policy, teleop, args.sipo, args.capo, sequencer
        )
        if sipo_buffers is not None:
            plot_sipo(sipo_buffers, buffers.time)
        if capo_buffers is not None:
            plot_capo(capo_buffers, buffers.time)
        if sipo_buffers is not None and capo_buffers is not None:
            plot_comparison(sipo_buffers, capo_buffers, buffers.time)
        plot_history(buffers)
    finally:
        teleop.close()


if __name__ == "__main__":
    main()
