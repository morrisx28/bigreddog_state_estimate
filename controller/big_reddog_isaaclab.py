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
        # vel_filter_tau is the per-axis body-frame low-pass (forward, lateral,
        # vertical). The vertical (vz) axis gets a heavier low-pass than the
        # library default (0.03 -> 0.12 s) to smooth the contact-transition
        # jerk; raise it further for more smoothing (at the cost of lag).
        self.estimator = BigReddogStateEstimator(
            foot_force_threshold=-30.0, enable_leg_yaw=False, enable_slope=True,
            vel_filter_tau=(0.08, 0.15, 0.12), vel_median_window=7,
            update_rate_hz=self.est_rate_hz)
        self._est_period = 1.0 / self.est_rate_hz  # throttle period (s)
        self._last_est_perf = None        # perf_counter of last estimator run
        self.lin_vel_world = np.zeros(3)  # base linear velocity in world frame (filtered)
        self.lin_vel_body = np.zeros(3)   # base linear velocity in body frame (filtered)
        self.lin_vel_world_raw = np.zeros(3)  # unfiltered estimate, world frame
        self.foot_contact = np.zeros(4)   # FL FR RL RR contact probability

        # Record three sources for plotting after exit:
        #   1) estimator output, 2) raw IMU from the robot, 3) user cmd_vel.
        self.rec_t = []            # seconds
        # estimator output
        self.rec_est_world = []    # estimated base lin. velocity, world frame (filtered)
        self.rec_est_body = []     # estimated base lin. velocity, body frame (filtered)
        self.rec_est_ang_vel = []  # estimator output angular velocity [roll, pitch, yaw] rate
        # raw IMU from the robot
        self.rec_imu_gyro = []     # raw IMU gyroscope [wx, wy, wz]
        self.rec_imu_accel = []    # raw IMU accelerometer [ax, ay, az]
        self.rec_imu_quat = []     # raw IMU quaternion [w, x, y, z]
        # user command
        self.rec_cmd_vel = []      # teleop cmd_vel [vx, vy, yaw_rate]

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

        # Init default pos #
        self.Start()

        print("Initial Sucess !!!")

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

        # Record the three sources: estimator output, raw IMU, user cmd_vel.
        self.rec_t.append(t_ms / 1000.0)
        # 1) estimator output
        self.rec_est_world.append(self.lin_vel_world.copy())
        self.rec_est_body.append(self.lin_vel_body.copy())
        self.rec_est_ang_vel.append(np.array(
            [result.odom.RollVel, result.odom.PitchVel, result.odom.YawVel],
            dtype=np.float32))
        # 2) raw IMU from the robot
        self.rec_imu_gyro.append(self.ang_vel.copy())
        self.rec_imu_accel.append(self.accel.copy())
        self.rec_imu_quat.append(self.quat.copy())
        # 3) user command (vx, vy, yaw_rate)
        self.rec_cmd_vel.append(
            np.array(self.teleop.get_command(), dtype=np.float32))

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

    def plot_and_save(self, out_dir=None):
        """Plot and save the three recorded sources after the run.

        Records (no ground truth):
          1) estimator output  -- base linear velocity (world/body) and
             angular velocity (roll/pitch/yaw rate),
          2) raw IMU from the robot -- gyroscope, accelerometer, quaternion,
          3) user cmd_vel -- teleop command (vx, vy, yaw_rate).

        Saves one CSV plus three PNGs (linear velocity, angular velocity,
        raw IMU). Called automatically on shutdown.
        """
        if len(self.rec_t) == 0:
            print("[plot] no recorded data, skipping.")
            return

        out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "lin_vel_logs")
        os.makedirs(out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")

        t = np.array(self.rec_t, dtype=np.float64)
        t = t - t[0]
        est_w = np.array(self.rec_est_world, dtype=np.float64)    # est lin vel, world
        est_b = np.array(self.rec_est_body, dtype=np.float64)     # est lin vel, body
        est_ang = np.array(self.rec_est_ang_vel, dtype=np.float64)  # est ang vel
        gyro = np.array(self.rec_imu_gyro, dtype=np.float64)      # raw IMU gyro
        accel = np.array(self.rec_imu_accel, dtype=np.float64)    # raw IMU accel
        quat = np.array(self.rec_imu_quat, dtype=np.float64)      # raw IMU quat
        cmd = np.array(self.rec_cmd_vel, dtype=np.float64)        # user cmd_vel

        # --- save CSV ---
        csv_path = os.path.join(out_dir, f"state_log_{stamp}.csv")
        with open(csv_path, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["t",
                           "est_vx_w", "est_vy_w", "est_vz_w",
                           "est_vx_b", "est_vy_b", "est_vz_b",
                           "est_wx", "est_wy", "est_wz",
                           "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",
                           "imu_acc_x", "imu_acc_y", "imu_acc_z",
                           "imu_qw", "imu_qx", "imu_qy", "imu_qz",
                           "cmd_vx", "cmd_vy", "cmd_yaw_rate"])
            for i in range(len(t)):
                wcsv.writerow([t[i], *est_w[i], *est_b[i], *est_ang[i],
                               *gyro[i], *accel[i], *quat[i], *cmd[i]])
        print(f"[plot] saved {csv_path}")

        # --- figure 1: estimator linear velocity (body frame) vs user cmd ---
        # cmd_vel is (vx, vy, yaw_rate); overlay cmd on the vx/vy axes only.
        labels = ["vx", "vy", "vz"]
        fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
        for r in range(3):
            axes[r, 0].plot(t, est_w[:, r], "r-", lw=1.2, label="estimator")
            axes[r, 0].set_ylabel(f"{labels[r]} [m/s]")
            axes[r, 0].grid(True, alpha=0.3)

            axes[r, 1].plot(t, est_b[:, r], "r-", lw=1.2, label="estimator")
            if r < 2:  # commanded vx / vy
                axes[r, 1].plot(t, cmd[:, r], "b--", lw=1.2, label="cmd_vel")
            axes[r, 1].grid(True, alpha=0.3)
        axes[0, 0].set_title("World frame (estimator)")
        axes[0, 1].set_title("Body frame (estimator vs cmd_vel)")
        axes[0, 0].legend(loc="upper right")
        axes[0, 1].legend(loc="upper right")
        axes[2, 0].set_xlabel("time [s]")
        axes[2, 1].set_xlabel("time [s]")
        fig.suptitle("Big Reddog base linear velocity: estimator output vs user cmd_vel")
        fig.tight_layout()
        png_path = os.path.join(out_dir, f"lin_vel_{stamp}.png")
        fig.savefig(png_path, dpi=120)
        print(f"[plot] saved {png_path}")

        # --- figure 2: angular velocity -- raw IMU gyro vs estimator output ---
        # overlay the user yaw-rate command on the wz (yaw) axis.
        w_labels = ["wx (roll rate)", "wy (pitch rate)", "wz (yaw rate)"]
        fig2, axes2 = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        for r in range(3):
            axes2[r].plot(t, gyro[:, r], "k-", lw=1.2, label="raw IMU gyro")
            axes2[r].plot(t, est_ang[:, r], "r-", lw=1.2, label="estimator output")
            if r == 2:  # commanded yaw rate
                axes2[r].plot(t, cmd[:, 2], "b--", lw=1.2, label="cmd yaw rate")
            axes2[r].set_ylabel(f"{w_labels[r]} [rad/s]")
            axes2[r].grid(True, alpha=0.3)
        axes2[0].legend(loc="upper right")
        axes2[2].legend(loc="upper right")
        axes2[2].set_xlabel("time [s]")
        fig2.suptitle("Big Reddog angular velocity: raw IMU gyro vs estimator output")
        fig2.tight_layout()
        ang_png_path = os.path.join(out_dir, f"ang_vel_{stamp}.png")
        fig2.savefig(ang_png_path, dpi=120)
        print(f"[plot] saved {ang_png_path}")

        # --- figure 3: raw IMU -- gyroscope and accelerometer ---
        g_labels = ["gyro x", "gyro y", "gyro z"]
        a_labels = ["accel x", "accel y", "accel z"]
        fig3, axes3 = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
        for r in range(3):
            axes3[r, 0].plot(t, gyro[:, r], "g-", lw=1.0)
            axes3[r, 0].set_ylabel(f"{g_labels[r]} [rad/s]")
            axes3[r, 0].grid(True, alpha=0.3)

            axes3[r, 1].plot(t, accel[:, r], "m-", lw=1.0)
            axes3[r, 1].set_ylabel(f"{a_labels[r]} [m/s^2]")
            axes3[r, 1].grid(True, alpha=0.3)
        axes3[0, 0].set_title("Gyroscope")
        axes3[0, 1].set_title("Accelerometer")
        axes3[2, 0].set_xlabel("time [s]")
        axes3[2, 1].set_xlabel("time [s]")
        fig3.suptitle("Big Reddog raw IMU data from robot")
        fig3.tight_layout()
        imu_png_path = os.path.join(out_dir, f"imu_{stamp}.png")
        fig3.savefig(imu_png_path, dpi=120)
        print(f"[plot] saved {imu_png_path}")

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
