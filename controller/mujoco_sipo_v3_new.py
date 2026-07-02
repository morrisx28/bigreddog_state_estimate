# Reference: https://github.com/ShuoYangRobotics/Multi-IMU-Proprioceptive-Odometry

import numpy as np
import mujoco
import yaml
import casadi as cs
import time

def quat_to_rot_casadi(q):
    """ Quaternion (w, x, y, z) to Rotation Matrix """
    w, x, y, z = q[0], q[1], q[2], q[3]
    row1 = cs.horzcat(1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w))
    row2 = cs.horzcat(    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w))
    row3 = cs.horzcat(    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2))
    return cs.vertcat(row1, row2, row3)

def quat_kinematics_casadi(q, w_body):
    """ Quaternion derivative: q_dot = 0.5 * Omega(w) * q """
    wx, wy, wz = w_body[0], w_body[1], w_body[2]
    Omega = cs.vertcat(
        cs.horzcat(0, -wx, -wy, -wz),
        cs.horzcat(wx, 0, wz, -wy),
        cs.horzcat(wy, -wz, 0, wx),
        cs.horzcat(wz, wy, -wx, 0)
    )
    return 0.5 * cs.mtimes(Omega, q)

class SIPO:
    def __init__(self, xml_path, config_path="scripts/sim2sim/config/sipo_config.yaml"):
        print(f"Loading SIPO Config from: {config_path}")
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        self.num_legs = self.cfg['num_legs']
        self.meas_per_leg = self.cfg['meas_per_leg']
        self.state_dim = self.cfg['state_dim']
        self.dt = 0.002 # Default, will be overridden by predict(dt)
        self.wheel_radius = self.cfg.get('wheel_radius', 0.077)
        self.contact_scaling = self.cfg.get('contact_scaling', 10000.0)

        # MuJoCo Model
        print(f"Loading Kinematics Model from: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.leg_names = ['L_wheel', 'R_wheel'] # Could be in config

        # Slices
        self.idx_pos = slice(0, 3)
        self.idx_vel = slice(3, 6)
        self.idx_quat = slice(6, 10)
        self.idx_feet = slice(10, 10 + 3*self.num_legs)
        self.idx_ba = slice(self.idx_feet.stop, self.idx_feet.stop + 3)
        self.idx_bg = slice(self.idx_ba.stop, self.idx_ba.stop + 3)
        self.idx_wvel = slice(self.idx_bg.stop, self.idx_bg.stop + self.num_legs)
        self.idx_time = self.state_dim - 1
        
        # Initial Covariance P
        self.P = np.zeros((self.state_dim, self.state_dim))
        init_cov = self.cfg['init_cov']
        
        np.fill_diagonal(self.P[self.idx_pos, self.idx_pos], init_cov['pos'])
        np.fill_diagonal(self.P[self.idx_vel, self.idx_vel], init_cov['vel'])
        np.fill_diagonal(self.P[self.idx_quat, self.idx_quat], init_cov['quat'])
        np.fill_diagonal(self.P[self.idx_feet, self.idx_feet], init_cov['foot_pos'])
        np.fill_diagonal(self.P[self.idx_ba, self.idx_ba], init_cov['bias'])
        np.fill_diagonal(self.P[self.idx_bg, self.idx_bg], init_cov['bias'])
        np.fill_diagonal(self.P[self.idx_wvel, self.idx_wvel], init_cov['wheel_vel'])
        
        # State vector
        self.x = np.zeros(self.state_dim)

        # Extract kinematic parameters from Mujoco model once
        self._extract_kinematic_chain()

        # Build CasADi Functions
        self._build_casadi_functions()
        
    def _extract_kinematic_chain(self):
        """
        Extracts kinematic parameters (body offsets, joint axes, etc.) from Mujoco model.
        This allows us to build a symbolic kinematics function in CasADi without calling Mujoco at runtime.
        """
        self.kinematic_chains = {}
        
        for leg_name in self.leg_names:
            chain = []
            # Start from end-effector body
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, leg_name)
            
            # Trace back to root (Body 1, usually)
            while body_id > 1:
                # Body info
                pos = self.model.body_pos[body_id].copy()
                quat = self.model.body_quat[body_id].copy() # [w, x, y, z]
                
                # Joint info
                jnt_adr = self.model.body_jntadr[body_id]
                jnt_num = self.model.body_jntnum[body_id]
                
                joints = []
                if jnt_num > 0:
                    for j in range(jnt_num):
                        j_id = jnt_adr + j
                        j_type = self.model.jnt_type[j_id]
                        j_axis = self.model.jnt_axis[j_id].copy()
                        j_pos = self.model.jnt_pos[j_id].copy() # Anchor point relative to body
                        j_qpos_adr = self.model.jnt_qposadr[j_id]
                        
                        # We assume qpos_sensed starts at qpos[7]
                        input_idx = j_qpos_adr - 7
                        
                        joints.append({
                            'input_idx': input_idx,
                            'type': j_type, # 2: slide, 3: hinge
                            'axis': j_axis,
                            'pos': j_pos
                        })
                
                chain.append({
                    'pos': pos,
                    'quat': quat,
                    'joints': joints
                })
                
                body_id = self.model.body_parentid[body_id]
                
            # Reverse chain to go from Root -> End Effector
            self.kinematic_chains[leg_name] = chain[::-1]

    def _build_casadi_functions(self):
        max_idx = 0
        for leg in self.leg_names:
            if leg in self.kinematic_chains:
                for link in self.kinematic_chains[leg]:
                    for j in link['joints']:
                        if j['input_idx'] > max_idx:
                            max_idx = j['input_idx']
        self.n_q_kin = max_idx + 1

        # ---------------------------------------------------------
        # 1. Symbolic Forward Kinematics
        # ---------------------------------------------------------
        q_kin_sym = cs.SX.sym('q_kin', self.n_q_kin)
        dq_kin_sym = cs.SX.sym('dq_kin', self.n_q_kin)
        
        z_kin_sym_list = []
        R_hub_sym_list = [] # Store each wheel's Hub rotation matrix (relative to Base)

        for leg_name in self.leg_names:
            if leg_name not in self.kinematic_chains:
                continue
                
            p_curr = cs.DM.zeros(3)
            R_curr = cs.DM.eye(3)
            R_hub = cs.DM.eye(3)
            
            for link in self.kinematic_chains[leg_name]:
                pos_off = cs.DM(link['pos'])
                quat_off = link['quat'] 
                w, x, y, z = quat_off[0], quat_off[1], quat_off[2], quat_off[3]
                
                R_off = cs.vertcat(
                    cs.horzcat(1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)),
                    cs.horzcat(    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)),
                    cs.horzcat(    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2))
                )
                
                p_curr = p_curr + cs.mtimes(R_curr, pos_off)
                R_curr = cs.mtimes(R_curr, R_off)
                
                for j in link['joints']:
                    j_type = j['type']
                    j_axis = cs.DM(j['axis'])
                    j_pos = cs.DM(j['pos'])
                    idx = j['input_idx']
                    theta = q_kin_sym[idx]
                    
                    if j_type == 2: # Slide
                        trans = j_axis * theta
                        p_curr = p_curr + cs.mtimes(R_curr, trans)
                        
                    elif j_type == 3: # Hinge
                        kx, ky, kz = j_axis[0], j_axis[1], j_axis[2]
                        K = cs.vertcat(
                            cs.horzcat(0, -kz, ky),
                            cs.horzcat(kz, 0, -kx),
                            cs.horzcat(-ky, kx, 0)
                        )
                        R_j = cs.DM.eye(3) + cs.sin(theta)*K + (1-cs.cos(theta))*cs.mtimes(K, K)
                        term = cs.mtimes((cs.DM.eye(3) - R_j), j_pos)
                        p_curr = p_curr + cs.mtimes(R_curr, term)
                        
                        # Store rotation matrix before wheel rotation joint
                        R_hub = R_curr 
                        R_curr = cs.mtimes(R_curr, R_j)
            
            p_center = p_curr
            v_center = cs.jtimes(p_center, q_kin_sym, dq_kin_sym)
            
            R_dot = cs.jtimes(R_curr, q_kin_sym, dq_kin_sym)
            S_w = cs.mtimes(R_dot, R_curr.T)
            w_wheel = cs.vertcat(S_w[2,1], S_w[0,2], S_w[1,0])
            
            # Output hub center, hub velocity, and wheel angular velocity separately.
            # The radius is applied in the correct frame (world Z) inside the measurement model.
            z_kin_sym_list.append(p_center)   # hub position in body frame
            z_kin_sym_list.append(v_center)   # hub linear velocity in body frame
            z_kin_sym_list.append(w_wheel)    # wheel angular velocity in body frame (from measured dq)
            R_hub_sym_list.append(R_hub)

        z_kin_sym = cs.vertcat(*z_kin_sym_list)
        self.fk_func = cs.Function('fk', [q_kin_sym, dq_kin_sym], [z_kin_sym])
        self.R_hub_func = cs.Function('R_hub_func', [q_kin_sym], R_hub_sym_list)
        self.fk_stride = 9  # values per leg: [p_hub(3), v_hub(3), w_wheel(3)]
        
        # ---------------------------------------------------------
        # 2. Prediction Model f(x, u, dt)
        # ---------------------------------------------------------
        x_sym = cs.SX.sym('x', self.state_dim)
        u_sym = cs.SX.sym('u', 7 + self.n_q_kin)  
        
        pos = x_sym[self.idx_pos]
        vel = x_sym[self.idx_vel]
        quat = x_sym[self.idx_quat]
        feet = x_sym[self.idx_feet]
        ba = x_sym[self.idx_ba]
        bg = x_sym[self.idx_bg]
        wvel = x_sym[self.idx_wvel]
        tk = x_sym[self.idx_time]
        
        acc_meas = u_sym[0:3]
        gyro_meas = u_sym[3:6]
        dt_in = u_sym[6]
        q_kin_in = u_sym[7:7 + self.n_q_kin]
        
        acc_body = acc_meas - ba
        w_body = gyro_meas - bg
        
        R_base = quat_to_rot_casadi(quat)
        
        # Dynamically calculate current Hub rotation matrix
        R_hubs = self.R_hub_func(q_kin_in)
        
        d_pos = vel
        d_vel = cs.mtimes(R_base, acc_body) - cs.vertcat(0, 0, 9.81)
        d_quat = quat_kinematics_casadi(quat, w_body)
        
        d_feet_list = []
        for i in range(self.num_legs):
            # The wheel always rotates around the Hub's Y axis
            w_vec_hub = cs.vertcat(0, wvel[i], 0)
            
            # Transform angular velocity vector to Base frame
            w_vec_base = cs.mtimes(R_hubs[i], w_vec_hub)
            
            # Contact point lever arm is vertically down in Base frame
            r_vec_base = cs.vertcat(0, 0, -self.wheel_radius)
            
            # Calculate pure rolling Base velocity via v = w x r, then transform to World
            v_base = -cs.cross(w_vec_base, r_vec_base)
            v_world = cs.mtimes(R_base, v_base)
            
            d_feet_list.append(v_world)
            
        d_feet = cs.vertcat(*d_feet_list)
        d_ba = cs.DM.zeros(3, 1)
        d_bg = cs.DM.zeros(3, 1)
        d_wvel = cs.DM.zeros(self.num_legs, 1)
        d_tk = 1.0
        
        x_dot = cs.vertcat(d_pos, d_vel, d_quat, d_feet, d_ba, d_bg, d_wvel, d_tk)  

        dyn_func = cs.Function('dyn', [x_sym, u_sym], [x_dot])
        k1_val = dyn_func(x_sym, u_sym)
        k2_val = dyn_func(x_sym + 0.5*dt_in*k1_val, u_sym)
        k3_val = dyn_func(x_sym + 0.5*dt_in*k2_val, u_sym)
        k4_val = dyn_func(x_sym + dt_in*k3_val, u_sym)
        
        x_next = x_sym + (dt_in/6.0)*(k1_val + 2*k2_val + 2*k3_val + k4_val)
        
        self.f_func = cs.Function('f', [x_sym, u_sym], [x_next])
        self.df_dx_func = cs.Function('df_dx', [x_sym, u_sym], [cs.jacobian(x_next, x_sym)])

        # ---------------------------------------------------------
        # 3. Measurement Model h(x, q_kin, dq_kin)
        # ---------------------------------------------------------
        gyro_in = cs.SX.sym('gyro_in', 3)
        yaw_in = cs.SX.sym('yaw_in', 1)
        wvel_meas = cs.SX.sym('wvel_meas', self.num_legs)
        
        y_list = []
        R_inv = R_base.T
        w_body_curr = gyro_in - bg

        r_world = cs.DM([0.0, 0.0, -self.wheel_radius])

        for i in range(self.num_legs):
            # FK outputs 9 values per leg: [p_hub(3), v_hub(3), w_wheel(3)]
            p_hub  = z_kin_sym[i*9 : i*9+3]
            v_hub  = z_kin_sym[i*9+3 : i*9+6]
            w_whl  = z_kin_sym[i*9+6 : i*9+9]

            # Contact-from-hub in body frame: radius drops straight down in world Z
            r_body = cs.mtimes(R_base.T, r_world)

            # Foot position in world: hub + radius straight down in world Z
            p_foot_kin_world = pos + cs.mtimes(R_base, p_hub) + r_world

            # 1. Position Residual
            # Wheeled robots cannot bind X, Y rigidly, otherwise turning will cause tearing.
            # We force X, Y errors to 0, keeping only the Z height constraint (Z=0).
            res_pos = cs.vertcat(0, 0, p_foot_kin_world[2])
            y_list.append(res_pos)

            # 2. Velocity Residual
            # Contact velocity in body frame = hub vel
            #   + body rotation at hub
            #   + body rotation + wheel spin at contact offset (r_body)
            v_contact_body = (v_hub
                              + cs.cross(w_body_curr, p_hub)
                              + cs.cross(w_body_curr + w_whl, r_body))
            res_vel = vel + cs.mtimes(R_base, v_contact_body)
            y_list.append(res_vel)

            # 3. Height Residual
            res_h = p_foot_kin_world[2]
            y_list.append(res_h)
            
            # 4. Wheel Velocity Residual
            res_wvel = wvel_meas[i] - wvel[i]
            y_list.append(res_wvel)
            
        # 5. Yaw Residual
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        yaw_est = cs.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy**2 + qz**2))
        res_yaw = yaw_in - yaw_est
        y_list.append(res_yaw)

        # 6. Non-Holonomic Constraint: lateral body velocity must be zero
        # Differential drive robots cannot slide sideways — enforce vy_body = 0
        vel_body = cs.mtimes(R_base.T, vel)
        res_nhc = vel_body[1]
        y_list.append(res_nhc)

        y_vec = cs.vertcat(*y_list)
        self.h_func = cs.Function('h', [x_sym, q_kin_sym, dq_kin_sym, gyro_in, yaw_in, wvel_meas], [y_vec])
        self.avg_func = cs.Function('H', [x_sym, q_kin_sym, dq_kin_sym, gyro_in, yaw_in, wvel_meas], [cs.jacobian(y_vec, x_sym)])

    def init_state(self, pos, quat, feet_pos):
        self.x[self.idx_pos] = pos
        self.x[self.idx_quat] = quat 
        self.x[self.idx_feet] = feet_pos
        self.x[self.idx_vel] = 0
        self.x[self.idx_ba] = 0
        self.x[self.idx_bg] = 0
        self.x[self.idx_wvel] = 0
        self.x[self.idx_time] = 0

    def predict(self, acc, gyro, dt, q_kin):
        u = np.concatenate([acc, gyro, [dt], q_kin])
        
        x_next = self.f_func(self.x, u).full().flatten()
        F = self.df_dx_func(self.x, u).full()
        
        pn = self.cfg['process_noise']
        Q_base = np.zeros(self.state_dim)
        Q_base[self.idx_pos] = pn['pos']
        Q_base[self.idx_vel] = pn['vel']
        Q_base[self.idx_quat] = pn['quat']
        Q_base[self.idx_feet] = pn['foot_pos']
        Q_base[self.idx_ba] = pn['acc_bias']
        Q_base[self.idx_bg] = pn['gyro_bias']
        Q_base[self.idx_wvel] = pn['wheel_vel']
        
        Q = np.diag(Q_base) * dt
        
        self.x = x_next
        self.P = F @ self.P @ F.T + Q

        q_norm = np.linalg.norm(self.x[self.idx_quat])
        if q_norm > 1e-6:
            self.x[self.idx_quat] /= q_norm

    def get_kinematics(self, q, dq):
        """
        Calculates Forward Kinematics using the CasADi function.
        Returns flattened array [p1_x, p1_y, p1_z, v1_x, ...].
        Used for initialization.
        """
        res = self.fk_func(q, dq).full().flatten()
        return res

    def update(self, qpos_sense, qvel_sense, contact_flags, gyro, wheel_vel_meas, yaw_meas=None):
        meas_dim = self.meas_per_leg * self.num_legs + 2  # +1 yaw, +1 NHC
        
        if yaw_meas is None:
            qw, qx, qy, qz = self.x[self.idx_quat]
            yaw_meas_val = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy**2 + qz**2))
        else:
            yaw_meas_val = yaw_meas
            
        y = self.h_func(self.x, qpos_sense, qvel_sense, gyro, yaw_meas_val, wheel_vel_meas).full().flatten()
        H = self.avg_func(self.x, qpos_sense, qvel_sense, gyro, yaw_meas_val, wheel_vel_meas).full()
        
        R = np.zeros((meas_dim, meas_dim))
        mn = self.cfg['meas_noise']
        
        r_base_diag = np.concatenate([
            [mn.get('pos', 1e-2)]*3,
            [mn.get('vel', 1e-2)]*3,
            [mn.get('height', 1e-3)],
            [mn.get('wheel_vel', 1e-2)]
        ])
        
        for i in range(self.num_legs):
            c_flag = contact_flags[i]
            scale = 1.0 if c_flag > 0.5 else self.contact_scaling
            idx = i * self.meas_per_leg
            R[idx:idx+self.meas_per_leg, idx:idx+self.meas_per_leg] = np.diag(r_base_diag * scale)
            
        yaw_idx = meas_dim - 2
        nhc_idx = meas_dim - 1
        if yaw_meas is None:
            R[yaw_idx, yaw_idx] = 1e6
        else:
            R[yaw_idx, yaw_idx] = mn.get('yaw', 1e-2)
        R[nhc_idx, nhc_idx] = mn.get('nhc', 5e-3)
            
        innovation = -y
        S = H @ self.P @ H.T + R
        
        try:
            K = np.linalg.solve(S, H @ self.P).T
        except np.linalg.LinAlgError:
            K = np.zeros((self.state_dim, meas_dim))

        K[0:2, :] = 0
            
        dx = K @ innovation
        self.x += dx
        
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
        q_norm = np.linalg.norm(self.x[self.idx_quat])
        if q_norm > 1e-6:
            self.x[self.idx_quat] /= q_norm
            
        return self.x.copy()

def get_contact_states(m, d, body_names):
    contacts = np.zeros(len(body_names))
    for i, name in enumerate(body_names):
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        for j in range(d.ncon):
            con = d.contact[j]
            b1 = m.geom_bodyid[con.geom1]
            b2 = m.geom_bodyid[con.geom2]
            if b1 == body_id or b2 == body_id:
                contacts[i] = 1.0
                break
    return contacts
