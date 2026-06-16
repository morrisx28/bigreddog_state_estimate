import time
import sys
import numpy as np
import threading
import traceback
import torch
import yaml
import argparse
import matplotlib.pyplot as plt
import csv
import pathlib
import os
import gui_teleop

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capo_estimator import BigReddogStateEstimator

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
import struct


NUM_MOTORS = 12

class Controller:
    def __init__(self):


        config_file = 'big_reddog_isaaclab.yaml'
        with open(f"{config_file}", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.dt = config["dt"]
            self.ang_vel_scale = config["ang_vel_scale"]
            policy_path = config["policy_path"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.decimation = config["control_decimation"]
            self.cmd_scale = config["cmd_scale"]
            num_actions = config["num_actions"]
            one_step_obs_size = config["one_step_obs_size"]
            obs_buffer_size = config.get("obs_buffer_size", 1)
            # self.action_scale = config["action_scale"]



            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)
            self.action_scale = np.array(config["action_scale"], dtype=np.float32)

            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.sit_angles = np.array(config["sit_angles"], dtype=np.float32)
            
            self.cmd_init = np.array(config["cmd_init"], dtype=np.float32)

        self.low_cmd = unitree_go_msg_dds__LowCmd_()  
        self.low_state = None  

        self.teleop = gui_teleop.GUITeleop(config_init=config["cmd_init"], max_lin=0.8, max_ang=0.5)


        self.controller_rt = 0.0
        self.is_running = False

        # thread handling
        self.lowCmdWriteThreadPtr = None

        # state
        self.target_dof_pos = self.default_angles.copy()
        self.target_dof_vel = np.zeros(NUM_MOTORS)
        self.qpos = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.qvel = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.qtau = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.quat = np.zeros(4) # q_w q_x q_y q_z
        self.ang_vel = np.zeros(3)
        self.accel = np.zeros(3)  # body linear acceleration from IMU

        self.mode = ''

        # --- Proprioceptive linear-velocity estimator (CAPO) ---
        # foot_force_threshold is torque-derived and must be tuned on the
        # real robot; start conservative and adjust while watching contacts.
        self.est_rate_hz = 200.0          # estimator update frequency
        self.estimator = BigReddogStateEstimator(
            foot_force_threshold=-30.0, enable_leg_yaw=False, enable_slope=True,
            update_rate_hz=self.est_rate_hz)
        self._est_period = 1.0 / self.est_rate_hz  # throttle period (s)
        self._last_est_perf = None        # perf_counter of last estimator run
        self.lin_vel_world = np.zeros(3)  # base linear velocity in world frame (filtered)
        self.lin_vel_body = np.zeros(3)   # base linear velocity in body frame (filtered)
        self.lin_vel_world_raw = np.zeros(3)  # unfiltered estimate, world frame
        self.foot_contact = np.zeros(4)   # FL FR RL RR contact probability

        # Ground truth from the MuJoCo bridge (rt/sportmodestate.velocity is the
        # base framelinvel in WORLD frame). Updated by HighStateMessageHandler.
        self.gt_lin_vel_world = np.zeros(3)
        self.gt_received = False

        # Record (estimate vs ground truth) for plotting after exit.
        self.rec_t = []          # seconds
        self.rec_est_world = []  # estimated base velocity, world frame (filtered)
        self.rec_est_body = []   # estimated base velocity, body frame (filtered)
        self.rec_est_world_raw = []  # unfiltered estimate, world frame
        self.rec_gt_world = []   # ground-truth base velocity, world frame
        self.rec_quat = []       # body quaternion [w,x,y,z] at each sample
        self.rec_contact = []    # FL FR RL RR contact probability

        # Record
        self.ang_vel_data_list = []
        self.gravity_b_list = []
        self.joint_vel_list = []
        self.joint_pos_list = []
        self.lin_vel_list = []

        # RL related
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load TorchScript policy to GPU
        # self.policy = torch.jit.load(policy_path, map_location=self.device).to(self.device)

        self.policy = torch.jit.load(policy_path)
        self.counter = 0

        self.action = np.zeros(num_actions, dtype=np.float32)
        self.obs_history_buffer = torch.zeros((obs_buffer_size, one_step_obs_size))

        

        self.crc = CRC()

    # Public methods
    def Init(self):
        self.InitLowCmd()

        # create publisher #
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        # create subscriber #
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

        # ground-truth base velocity from the MuJoCo bridge (rt/sportmodestate)
        self.highstate_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.highstate_subscriber.Init(self.HighStateMessageHandler, 10)

        # Init default pos #
        self.Start()

        print("Initial Sucess !!!")

    def HighStateMessageHandler(self, msg: SportModeState_):
        # SportModeState.velocity is the base framelinvel in world frame.
        for i in range(3):
            self.gt_lin_vel_world[i] = msg.velocity[i]
        self.gt_received = True

    def get_root_local_rot_tan_norm(self, quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        # tangent = body x-axis in world frame
        tan = np.array([
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy + qw * qz),
            2 * (qx * qz - qw * qy),
        ])
        # normal = body z-axis in world frame
        norm = np.array([
            2 * (qx * qz + qw * qy),
            2 * (qy * qz - qw * qx),
            1 - 2 * (qx * qx + qy * qy),
        ])

        return np.concatenate([tan, norm])


    def Start(self):
        self.is_running = True
        self.lowCmdWriteThreadPtr = threading.Thread(target=self.LowCmdWrite)
        self.lowCmdWriteThreadPtr.start()

    def ShutDown(self):
        self.is_running = False
        self.teleop.close()
        self.lowCmdWriteThreadPtr.join()
        # Plot estimated vs ground-truth linear velocity after the run.
        self.plot_and_save()

    def InitLowCmd(self):
        self.low_cmd.head[0]=0xFE
        self.low_cmd.head[1]=0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(NUM_MOTORS):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q= self.sit_angles[i]
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg
        self.update_state()
    

    def stand(self):
        self.controller_rt += self.dt
        ## Get into Default Joint pos ##
        if (self.controller_rt < 6.0):
            # Stand up in first 3 second
            # Total time for standing up or standing down is about 1.2s
            phase = np.tanh(self.controller_rt / 3)
            for i in range(NUM_MOTORS):
                self.low_cmd.motor_cmd[i].q = phase * self.default_angles[i] + (
                    1 - phase) * self.qpos[i]
                self.low_cmd.motor_cmd[i].kp = 40
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kd = 1.0
                self.low_cmd.motor_cmd[i].tau = 0.0
    
    def reset_timer(self):
        self.controller_rt = 0.0
    
    def sit(self):
        self.controller_rt += self.dt
        ## Get into Default Joint pos ##
        if (self.controller_rt < 3.0):
            # Stand up in first 3 second
            # Total time for standing up or standing down is about 1.2s
            phase = np.tanh(self.controller_rt / 1.2)
            for i in range(NUM_MOTORS):
                self.low_cmd.motor_cmd[i].q = phase * self.sit_angles[i] + (
                    1 - phase) * self.qpos[i]
                self.low_cmd.motor_cmd[i].kp = 40
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kd = 1.0
                self.low_cmd.motor_cmd[i].tau = 0.0
    
    def move(self):
        if self.counter % self.decimation == 0 and self.counter > 0:
            self.action = self.step()
            for i in range(NUM_MOTORS):
                self.target_dof_pos[i] = self.default_angles[i] + self.action[i] * self.action_scale[i]


        for i in range(NUM_MOTORS):
            self.low_cmd.motor_cmd[i].q = self.target_dof_pos[i]
            self.low_cmd.motor_cmd[i].kp = self.kps[i]
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = self.kds[i]
            self.low_cmd.motor_cmd[i].tau = 0.0

        self.counter += 1

    def step(self):

        qpos, qvel, ang_vel, quat = self.get_current_state()
        root_rot = self.get_root_local_rot_tan_norm(quat)
        cmd = np.array(self.teleop.get_command(), dtype=np.float32)


        obs_list = [
            torch.tensor(ang_vel * self.ang_vel_scale, dtype=torch.float32),
            torch.tensor(root_rot, dtype=torch.float32),
            torch.tensor(cmd, dtype=torch.float32),
            torch.tensor((qpos - self.default_angles) * self.dof_pos_scale, dtype=torch.float32),
            torch.tensor(qvel * self.dof_vel_scale, dtype=torch.float32),
            torch.tensor(self.action, dtype=torch.float32),
        ]

        current_obs = torch.cat(obs_list, dim=0)
        self.obs_history_buffer = torch.roll(self.obs_history_buffer, shifts=-1, dims=0)
        self.obs_history_buffer[-1] = current_obs

        split_sizes = [o.numel() for o in obs_list]
        feature_groups = torch.split(self.obs_history_buffer, split_sizes, dim=1)
        flat_groups = [g.flatten() for g in feature_groups]

        obs_tensor = torch.cat(flat_groups).unsqueeze(0)
        with torch.no_grad():
            self.action = self.policy(obs_tensor).numpy().squeeze()

        return self.action
    
    

    def stand_up(self):
        self.mode = 'stand'
        self.reset_timer()

    def sit_down(self):
        self.mode = 'sit'
        self.reset_timer()
    
    def move_rl(self):
        self.mode = 'move'
        self.reset_timer()
    
    
    
    def update_state(self):
        for i in range(NUM_MOTORS):
            self.qpos[i] = self.low_state.motor_state[i].q
            self.qvel[i] = self.low_state.motor_state[i].dq
            self.qtau[i] = self.low_state.motor_state[i].tau_est


        for i in range(3):
            self.ang_vel[i] = self.low_state.imu_state.gyroscope[i]
            self.accel[i] = self.low_state.imu_state.accelerometer[i]

        for i in range(4):
            self.quat[i] = self.low_state.imu_state.quaternion[i]

        # Throttle the estimator to a fixed rate (default 200 Hz) so it is
        # rate-consistent with its Kalman model regardless of how fast
        # rt/lowstate arrives (200 Hz in sim, up to 500 Hz-1 kHz on hardware).
        now = time.perf_counter()
        if (self._last_est_perf is None
                or (now - self._last_est_perf) >= 0.9 * self._est_period):
            self._last_est_perf = now
            self.estimate_linear_velocity()

    def estimate_linear_velocity(self):
        """Run the proprioceptive estimator and cache the velocity result."""
        # Prefer the SDK tick (milliseconds); fall back to a wall clock so
        # the estimator's time base advances even when tick is unset (sim).
        tick = getattr(self.low_state, "tick", 0)
        if tick:
            t_ms = int(tick)
        else:
            t_ms = int(time.perf_counter() * 1000.0)

        result = self.estimator.update(
            self.quat, self.ang_vel, self.accel,
            self.qpos, self.qvel, self.qtau, t_ms)

        self.lin_vel_world = np.array(result.lin_vel_world, dtype=np.float32)
        self.lin_vel_body = np.array(result.lin_vel_body, dtype=np.float32)
        self.lin_vel_world_raw = np.array(result.lin_vel_world_raw, dtype=np.float32)
        self.foot_contact = np.array(result.foot_contact, dtype=np.float32)

        # Record estimate + ground truth (paired with the latest GT sample).
        self.rec_t.append(t_ms / 1000.0)
        self.rec_est_world.append(self.lin_vel_world.copy())
        self.rec_est_body.append(self.lin_vel_body.copy())
        self.rec_est_world_raw.append(self.lin_vel_world_raw.copy())
        self.rec_gt_world.append(self.gt_lin_vel_world.copy())
        self.rec_quat.append(self.quat.copy())
        self.rec_contact.append(self.foot_contact.copy())

    def get_current_state(self):
        return self.qpos, self.qvel, self.ang_vel, self.quat

    


    def LowCmdWrite(self):
        
        while self.is_running:
            step_start = time.perf_counter()
            if self.mode == 'stand':
                self.stand()
            elif self.mode == 'sit':
                self.sit()
            elif self.mode == 'move':
                self.move()
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)

            time_until_next_step = self.dt - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        self.ResetParam()
    
    
        
    def ResetParam(self):
        self.controller_rt = 0
        self.is_running = False

    @staticmethod
    def _quat_rotate_inv(q, v):
        """Rotate world-frame vector v into body frame: v_body = R(q)^T v."""
        w, x, y, z = q
        # conjugate quaternion rotation
        t = 2.0 * np.array([
            (-y) * v[2] - (-z) * v[1],
            (-z) * v[0] - (-x) * v[2],
            (-x) * v[1] - (-y) * v[0],
        ])
        return v + w * t + np.array([
            (-y) * t[2] - (-z) * t[1],
            (-z) * t[0] - (-x) * t[2],
            (-x) * t[1] - (-y) * t[0],
        ])

    def plot_and_save(self, out_dir=None):
        """Plot estimated vs ground-truth base linear velocity and save it.

        Produces a 3x2 figure (vx/vy/vz, world | body frames) plus a CSV of
        the raw records. Called automatically on shutdown.
        """
        if len(self.rec_t) == 0:
            print("[plot] no recorded data, skipping.")
            return

        out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "lin_vel_logs")
        os.makedirs(out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")

        t = np.array(self.rec_t, dtype=np.float64)
        t = t - t[0]
        est_w = np.array(self.rec_est_world, dtype=np.float64)
        est_b = np.array(self.rec_est_body, dtype=np.float64)
        est_w_raw = np.array(self.rec_est_world_raw, dtype=np.float64)
        gt_w = np.array(self.rec_gt_world, dtype=np.float64)
        quat = np.array(self.rec_quat, dtype=np.float64)

        # ground truth rotated into the body frame for the right column
        gt_b = np.array([self._quat_rotate_inv(quat[i], gt_w[i])
                         for i in range(len(t))], dtype=np.float64)

        if not self.gt_received:
            print("[plot] WARNING: no rt/sportmodestate received; "
                  "ground truth is all zeros (is the MuJoCo sim running?).")

        # --- save CSV ---
        csv_path = os.path.join(out_dir, f"lin_vel_{stamp}.csv")
        with open(csv_path, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["t",
                           "est_vx_w", "est_vy_w", "est_vz_w",
                           "gt_vx_w", "gt_vy_w", "gt_vz_w",
                           "est_vx_b", "est_vy_b", "est_vz_b",
                           "gt_vx_b", "gt_vy_b", "gt_vz_b",
                           "est_raw_vx_w", "est_raw_vy_w", "est_raw_vz_w"])
            for i in range(len(t)):
                wcsv.writerow([t[i], *est_w[i], *gt_w[i], *est_b[i], *gt_b[i],
                               *est_w_raw[i]])

        # --- error metrics (body frame, the usual RL observation) ---
        err_b = est_b - gt_b
        rmse_b = np.sqrt(np.mean(err_b ** 2, axis=0))
        rmse_w = np.sqrt(np.mean((est_w - gt_w) ** 2, axis=0))
        rmse_w_raw = np.sqrt(np.mean((est_w_raw - gt_w) ** 2, axis=0))
        print(f"[plot] world RMSE raw      vx={rmse_w_raw[0]:.3f}  "
              f"vy={rmse_w_raw[1]:.3f}  vz={rmse_w_raw[2]:.3f}  (m/s)")
        print(f"[plot] world RMSE filtered vx={rmse_w[0]:.3f}  "
              f"vy={rmse_w[1]:.3f}  vz={rmse_w[2]:.3f}  (m/s)")
        print(f"[plot] body  RMSE filtered vx={rmse_b[0]:.3f}  "
              f"vy={rmse_b[1]:.3f}  vz={rmse_b[2]:.3f}  (m/s)")

        # --- figure ---
        labels = ["vx", "vy", "vz"]
        fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
        for r in range(3):
            axes[r, 0].plot(t, est_w_raw[:, r], color="0.7", lw=0.7, label="estimate (raw)")
            axes[r, 0].plot(t, gt_w[:, r], "k-", lw=1.5, label="ground truth")
            axes[r, 0].plot(t, est_w[:, r], "r-", lw=1.2, label="estimate (filtered)")
            axes[r, 0].set_ylabel(f"{labels[r]} [m/s]")
            axes[r, 0].grid(True, alpha=0.3)

            axes[r, 1].plot(t, gt_b[:, r], "k-", lw=1.5, label="ground truth")
            axes[r, 1].plot(t, est_b[:, r], "r--", lw=1.2, label="estimate")
            axes[r, 1].grid(True, alpha=0.3)
        axes[0, 0].set_title("World frame")
        axes[0, 1].set_title(f"Body frame (RMSE vx/vy/vz = "
                             f"{rmse_b[0]:.3f}/{rmse_b[1]:.3f}/{rmse_b[2]:.3f})")
        axes[0, 0].legend(loc="upper right")
        axes[2, 0].set_xlabel("time [s]")
        axes[2, 1].set_xlabel("time [s]")
        fig.suptitle("Big Reddog base linear velocity: estimate vs MuJoCo ground truth")
        fig.tight_layout()

        png_path = os.path.join(out_dir, f"lin_vel_{stamp}.png")
        fig.savefig(png_path, dpi=120)
        print(f"[plot] saved {png_path}")
        print(f"[plot] saved {csv_path}")
        try:
            plt.show()
        except Exception:
            pass


if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(1, sys.argv[1])
    else:
        ChannelFactoryInitialize(1, "lo") # default DDS port for pineapple

    controller = Controller()
    controller.Init()

    command_dict = {
        "stand": controller.stand_up,
        "sit": controller.sit_down,
        "move": controller.move_rl,
    }

    while True:
        try:
            cmd = input("CMD :")
            if cmd in command_dict:
                command_dict[cmd]()
            elif cmd == "exit":
                controller.ShutDown()
                break

        except (KeyboardInterrupt, EOFError):
            # Ctrl-C / Ctrl-D: shut down cleanly and plot before exiting.
            controller.ShutDown()
            break
        except Exception as e:
            traceback.print_exc()
            try:
                controller.ShutDown()
            except Exception:
                traceback.print_exc()
            break
    sys.exit(0)
