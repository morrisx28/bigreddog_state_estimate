import select
import sys
import termios
import threading
import time
import tty
import importlib
import math

import numpy as np

_inputs_mod = None
try:
    _inputs_mod = importlib.import_module("inputs")
except ImportError:
    _inputs_mod = None

if _inputs_mod is not None:
    get_gamepad = getattr(_inputs_mod, "get_gamepad", None)
    UnpluggedError = getattr(_inputs_mod, "UnpluggedError", Exception)
else:
    get_gamepad = None

    class UnpluggedError(Exception):
        pass


class HeadlessTeleop:
    """Headless teleop with terminal keyboard and optional gamepad support."""

    def __init__(
        self,
        config_init=(0.0, 0.0, 0.0),
        lin_step=0.2,
        ang_step=0.2,
        max_lin=1.0,
        max_ang=3.0,
        height_init=0.3,
        height_step=0.01,
        min_height=0.2,
        max_height=0.35,
        ee_pose_init=None,
        ee_pos_step=0.01,
        ee_rot_step=0.05,
        ee_pos_min=None,
        ee_pos_max=None,
        gamepad_deadzone=0.1,
    ):
        self.cmd_vel = np.array(config_init, dtype=np.float32)
        self.cmd_height = float(height_init)
        self.ee_pose = None if ee_pose_init is None else np.array(ee_pose_init, dtype=np.float32)
        self.ee_pose_init = None if self.ee_pose is None else self.ee_pose.copy()
        self.ee_rpy = np.zeros(3, dtype=np.float32)
        if self.ee_pose is not None:
            self.ee_pose[3:] = self._normalize_quat(self.ee_pose[3:])
            self.ee_rpy = self._quat_to_rpy(self.ee_pose[3:])

        self.lin_step = float(lin_step)
        self.ang_step = float(ang_step)
        self.height_step = float(height_step)
        self.ee_pos_step = float(ee_pos_step)
        self.ee_rot_step = float(ee_rot_step)

        self.max_lin = float(max_lin)
        self.max_ang = float(max_ang)
        self.min_height = float(min_height)
        self.max_height = float(max_height)
        self.ee_pos_min = None if ee_pos_min is None else np.array(ee_pos_min, dtype=np.float32)
        self.ee_pos_max = None if ee_pos_max is None else np.array(ee_pos_max, dtype=np.float32)
        self.gamepad_deadzone = float(gamepad_deadzone)

        self.lock = threading.Lock()
        self.running = True

        self._keyboard_fd = None
        self._keyboard_old_settings = None
        self._keyboard_thread = None
        self._gamepad_thread = None

        self._setup_keyboard()
        self._start_gamepad_thread_if_available()

        print("Headless teleop active.")
        if self.ee_pose is not None:
            print("Keyboard: W/S linear, A/D yaw, SPACE stop.")
            print("EE pose: I/K x, J/L y, R/F z, T/G roll, Y/H pitch, U/O yaw, P reset.")
        else:
            print("Keyboard: W/S linear, A/D yaw, R/F height, SPACE stop.")
        if self._gamepad_thread is not None:
            if self.ee_pose is None:
                print("Gamepad: left stick Y linear, right/left stick X yaw, d-pad up/down height, A stop.")
            else:
                print("Gamepad: left stick Y linear, right/left stick X yaw, d-pad up/down EE z, A stop.")
        else:
            print("Gamepad: python package 'inputs' not found or no gamepad events available.")

    def get_command(self):
        with self.lock:
            return self.cmd_vel.copy()
            
    def get_height_command(self):
        with self.lock:
            return self.cmd_height

    def get_ee_pose_command(self):
        with self.lock:
            if self.ee_pose is None:
                raise RuntimeError("EE pose command is not configured for this teleop instance.")
            return self.ee_pose.copy()

    def close(self):
        self.running = False

        if self._keyboard_old_settings is not None and self._keyboard_fd is not None:
            try:
                termios.tcsetattr(self._keyboard_fd, termios.TCSADRAIN, self._keyboard_old_settings)
            except termios.error:
                pass

        if self._keyboard_thread is not None and self._keyboard_thread.is_alive():
            self._keyboard_thread.join(timeout=0.2)

    def _setup_keyboard(self):
        if not sys.stdin.isatty():
            print("Keyboard control disabled: stdin is not a TTY.")
            return

        self._keyboard_fd = sys.stdin.fileno()
        self._keyboard_old_settings = termios.tcgetattr(self._keyboard_fd)
        tty.setcbreak(self._keyboard_fd)

        self._keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._keyboard_thread.start()

    def _start_gamepad_thread_if_available(self):
        if get_gamepad is None:
            return

        self._gamepad_thread = threading.Thread(target=self._gamepad_loop, daemon=True)
        self._gamepad_thread.start()

    def _keyboard_loop(self):
        while self.running:
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                continue

            ch = sys.stdin.read(1)
            if not ch:
                continue

            key = ch.lower()
            with self.lock:
                if key == "w":
                    self.cmd_vel[0] = np.clip(self.cmd_vel[0] + self.lin_step, -self.max_lin, self.max_lin)
                elif key == "s":
                    self.cmd_vel[0] = np.clip(self.cmd_vel[0] - self.lin_step, -self.max_lin, self.max_lin)
                elif key == "a":
                    self.cmd_vel[2] = np.clip(self.cmd_vel[2] + self.ang_step, -self.max_ang, self.max_ang)
                elif key == "d":
                    self.cmd_vel[2] = np.clip(self.cmd_vel[2] - self.ang_step, -self.max_ang, self.max_ang)
                elif key == "r":
                    if self.ee_pose is None:
                        self.cmd_height = np.clip(self.cmd_height + self.height_step, self.min_height, self.max_height)
                    else:
                        self._adjust_ee_position(2, self.ee_pos_step)
                elif key == "f":
                    if self.ee_pose is None:
                        self.cmd_height = np.clip(self.cmd_height - self.height_step, self.min_height, self.max_height)
                    else:
                        self._adjust_ee_position(2, -self.ee_pos_step)
                elif self.ee_pose is not None and key == "i":
                    self._adjust_ee_position(0, self.ee_pos_step)
                elif self.ee_pose is not None and key == "k":
                    self._adjust_ee_position(0, -self.ee_pos_step)
                elif self.ee_pose is not None and key == "j":
                    self._adjust_ee_position(1, self.ee_pos_step)
                elif self.ee_pose is not None and key == "l":
                    self._adjust_ee_position(1, -self.ee_pos_step)
                elif self.ee_pose is not None and key == "t":
                    self._adjust_ee_rotation(0, self.ee_rot_step)
                elif self.ee_pose is not None and key == "g":
                    self._adjust_ee_rotation(0, -self.ee_rot_step)
                elif self.ee_pose is not None and key == "y":
                    self._adjust_ee_rotation(1, self.ee_rot_step)
                elif self.ee_pose is not None and key == "h":
                    self._adjust_ee_rotation(1, -self.ee_rot_step)
                elif self.ee_pose is not None and key == "u":
                    self._adjust_ee_rotation(2, self.ee_rot_step)
                elif self.ee_pose is not None and key == "o":
                    self._adjust_ee_rotation(2, -self.ee_rot_step)
                elif self.ee_pose is not None and key == "p":
                    self.ee_pose[:] = self.ee_pose_init
                    self.ee_rpy = self._quat_to_rpy(self.ee_pose[3:])
                    self.ee_pose[3:] = self._rpy_to_quat(self.ee_rpy)
                elif key == " ":
                    self.cmd_vel[:] = 0.0

    def _normalize_axis(self, raw, max_raw=32767.0):
        value = float(raw) / float(max_raw)
        value = np.clip(value, -1.0, 1.0)
        if abs(value) < self.gamepad_deadzone:
            return 0.0
        return value

    def _gamepad_loop(self):
        while self.running:
            try:
                events = get_gamepad()
            except UnpluggedError:
                time.sleep(0.5)
                continue
            except Exception:
                time.sleep(0.1)
                continue

            with self.lock:
                for ev in events:
                    print(ev.code, ev.state)
                    if ev.code == "ABS_Y":
                        self.cmd_vel[0] = np.clip(-self._normalize_axis(ev.state-127) * self.max_lin, -self.max_lin, self.max_lin)
                    elif ev.code in ("ABS_RX", "ABS_X"):
                        self.cmd_vel[2] = np.clip(self._normalize_axis(-(ev.state-127)) * self.max_ang, -self.max_ang, self.max_ang)
                    elif ev.code == "ABS_HAT0Y":
                        if ev.state == -1:
                            if self.ee_pose is None:
                                self.cmd_height = np.clip(self.cmd_height + self.height_step, self.min_height, self.max_height)
                            else:
                                self._adjust_ee_position(2, self.ee_pos_step)
                        elif ev.state == 1:
                            if self.ee_pose is None:
                                self.cmd_height = np.clip(self.cmd_height - self.height_step, self.min_height, self.max_height)
                            else:
                                self._adjust_ee_position(2, -self.ee_pos_step)
                    elif ev.code == "BTN_SOUTH" and int(ev.state) == 1:
                        self.cmd_vel[:] = 0.0

    def _adjust_ee_position(self, index, delta):
        self.ee_pose[index] += float(delta)
        if self.ee_pos_min is not None:
            self.ee_pose[:3] = np.maximum(self.ee_pose[:3], self.ee_pos_min)
        if self.ee_pos_max is not None:
            self.ee_pose[:3] = np.minimum(self.ee_pose[:3], self.ee_pos_max)

    def _adjust_ee_rotation(self, index, delta):
        self.ee_rpy[index] += float(delta)
        self.ee_pose[3:] = self._rpy_to_quat(self.ee_rpy)

    def _normalize_quat(self, quat):
        quat = np.array(quat, dtype=np.float32)
        norm = np.linalg.norm(quat)
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return quat / norm

    def _quat_to_rpy(self, quat):
        w, x, y, z = self._normalize_quat(quat)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw], dtype=np.float32)

    def _rpy_to_quat(self, rpy):
        roll, pitch, yaw = rpy
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        quat = np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float32,
        )
        return self._normalize_quat(quat)
