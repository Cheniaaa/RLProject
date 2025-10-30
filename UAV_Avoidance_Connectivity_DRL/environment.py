import math
from method import *
import numpy as np


class GroundBaseStation:
    """地面基站配置"""

    def __init__(self):
        self.gbs_num = Config.GBS_N
        self.area_size = Config.AREA_SIZE
        self.gbs_height = Config.GBS_H
        self.gbs_positions = self._generate_gbs_positions()

    def _generate_gbs_positions(self):
        """生成GBS位置"""
        positions = []
        grid_size = int(np.sqrt(self.gbs_num))
        x_step = self.area_size / (grid_size + 1)
        y_step = self.area_size / (grid_size + 1)

        for i in range(grid_size):
            for j in range(grid_size):
                if len(positions) < self.gbs_num:
                    x = (i + 1) * x_step + np.random.uniform(-10, 10)
                    y = (j + 1) * y_step + np.random.uniform(-10, 10)
                    positions.append([x, y, self.gbs_height])
        while len(positions) < self.gbs_num:
            x = np.random.uniform(0, self.area_size)
            y = np.random.uniform(0, self.area_size)
            positions.append([x, y, self.gbs_height])
        # for _ in range(Config.GBS_N):
        #     x = np.random.uniform(0, self.area_size)
        #     y = np.random.uniform(0, self.area_size)
        #     positions.append([x, y, self.gbs_height])
        return np.array(positions)

    def compute_sinr(self, uav_pos):
        """计算SINR，获取信号最好的基站"""
        max_power = -np.inf  # 初始化负无穷
        best_gbs_idx = 0

        # 计算干扰（除了服务中的BS的信号，其他BS的信号是干扰）
        received_powers = []
        for k in range(self.gbs_num):
            dist_k = np.linalg.norm(uav_pos[:2] - self.gbs_positions[k][:2])
            received_signal = compute_signal(dist_k)
            if received_signal > max_power:
                max_power = received_signal
                best_gbs_idx = k
            received_powers.append(received_signal)
        interference = sum(received_powers) - max_power
        sinr = max_power / (Config.NOISE_P + interference)

        return sinr, best_gbs_idx

    def get_best_gbs(self, uav_pos):
        max_power = -np.inf  # 初始化负无穷
        best_gbs_idx = 0
        for k in range(self.gbs_num):
            d_k = np.linalg.norm(uav_pos[:2] - self.gbs_positions[k][:2])
            received_signal = compute_signal(d_k)
            if received_signal > max_power:
                max_power = received_signal
                best_gbs_idx = k
        return best_gbs_idx


class Environment:
    """UAV环境"""

    def __init__(self, gbs_network: GroundBaseStation):
        self.area_size = Config.AREA_SIZE
        self.gbs_network = gbs_network
        # UAV状态
        self.uav_position = None
        self.uav_velocity = None
        self.uav_speed = None
        self.uav_orientation = None  # 当前朝向（弧度）
        # 任务点信息
        self.pos_origin = None
        self.pos_destination = None
        # 其他信息
        self.last_connection_time = 0  # TL(t)是无人机在时间t之前与蜂窝网络最后一次连接的时间
        self.time_step = 0
        self.trajectory = []

        self.max_steps = 1000

    def reset(self, start_position, end_position):
        """重置环境

        Returns:
            state: 环境观测状态
            sinr_state: SINR相关状态
        """
        self.pos_origin = np.array([
            start_position[0],
            start_position[1],
            Config.UAV_H
        ])
        self.pos_destination = np.array([
            end_position[0],
            end_position[1],
            Config.UAV_H
        ])
        self.uav_position = self.pos_origin.copy()  # copy()使用浅拷贝，新数组改变不会修改原数组
        self.uav_velocity = np.array([0.0, 0.0])
        # self.uav_orientation = 0.0
        to_goal_x_y = self.pos_destination[:2] - self.uav_position[:2]
        to_goal_angle = np.arctan2(to_goal_x_y[1], to_goal_x_y[0])
        self.uav_orientation = to_goal_angle
        self.uav_speed = 0.0

        self.trajectory = [self.uav_position[:2].copy()]
        self.time_step = 0
        return self.get_state()

    def step(self, action):
        """执行动作

        Parameters:
            action: [velocity_x, velocity_y, uav_speed, phi]

        Returns:
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 更新UAV状态
        self.uav_velocity = action[:2]
        self.uav_speed = action[2]
        self.uav_orientation = action[3]
        self.uav_position[:2] = self.uav_position[:2] + self.uav_velocity * Config.DT  # 更新位置

        # 边界约束
        self.uav_position[0] = np.clip(self.uav_position[0], 0, Config.AREA_SIZE)
        self.uav_position[1] = np.clip(self.uav_position[1], 0, Config.AREA_SIZE)
        # 添加轨迹
        self.trajectory.append(self.uav_position[:2].copy())

        sinr_signal, serving_gbs = self.gbs_network.compute_sinr(self.uav_position)
        # 更新连接状态
        is_connected = sinr_signal >= Config.SINR_T
        if is_connected:
            self.last_connection_time = self.time_step  # 处于连接状态时，将最后连接时间调整到同等时间步数

        dist_to_dest = np.linalg.norm(self.pos_destination[:2] - self.uav_position[:2])
        reward = self.get_reward(dist_to_dest, sinr_signal)

        # 判断是否结束
        done = False
        info = {
            'sinr': sinr_signal,
            'dist_to_dest': dist_to_dest,
            'is_connected': is_connected,
            'success': False,
            'disconnected': False,
            'out_of_time': False
        }

        if dist_to_dest < Config.DEST_THRESHOLD:
            done = True
            info['success'] = True

        # 判断是否断联时间过长
        disconnected_time = (self.time_step - self.last_connection_time) * Config.DT
        if disconnected_time > Config.T_T:
            # done = True
            info['disconnected'] = True
            # reward -= 20  # 额外惩罚

        # 超时
        if self.time_step > self.max_steps:
            # done = True
            info['out_of_time'] = True

        self.time_step += 1
        next_state, _ = self.get_state()
        return next_state, reward, done, info

    def get_reward(self, dist_to_dest, sinr):
        """计算奖励"""
        reward = 0

        # 连接性奖励 (R_s)
        if self.time_step % Config.N_T == 0:
            if sinr < Config.SINR_T:
                reward -= Config.ALPHA_2  # 断开连接惩罚
            elif sinr < Config.SINR_T + Config.SINR_B:
                reward -= Config.ALPHA_2 / 2  # 接近断开连接惩罚

        # 目的地奖励 (R_d)
        if dist_to_dest < Config.DEST_THRESHOLD:
            reward += Config.ALPHA_3

        # 时间惩罚 (R_t)
        reward -= Config.ALPHA_4

        return reward

    def get_state(self):
        """获取当前状态（以Agent为中心的坐标系）

        Returns:
            state: 环境观测状态
            sinr_state: SINR网络输入状态
        Notes:
            原文：由于最优策略应对任何坐标平面保持不变，我们遵循如[6]、[42]和[9]中所述的以智能体为中心的参数化方法，其中智能体位于原点，x 轴指向智能体的目标位置。
        """
        to_goal_x_y = self.pos_destination[:2] - self.uav_position[:2]
        d_g = np.linalg.norm(to_goal_x_y)
        to_goal_angle = np.arctan2(to_goal_x_y[1], to_goal_x_y[0])

        # 坐标系变换
        theta = -to_goal_angle  # 旋转角度
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # 旋转矩阵（固定）
        rot_matrix = np.array([[cos_theta, -sin_theta],
                               [sin_theta, cos_theta]])
        v_local = np.dot(rot_matrix, self.uav_velocity)  # dot()用于矩阵乘积

        phi_local = (self.uav_orientation - to_goal_angle - math.pi) % (2 * math.pi) + math.pi

        # SINR网络输入状态，式(31): Sjn_Bi=[[d_Bk, φ_Bk, θ_Bk]:k∈{1,...,Kn}]
        sinr_state = []
        # UAV自身状态（S_i=[d_g_i, v_max_i, ˜v_xi, ˜v_yi, ri, ˜φi]）
        state = [d_g, Config.MAX_V, v_local[0], v_local[1], Config.MAX_T_R, phi_local]

        # TODO: state还需要包含其他智能体信息，目前单智能体不需要

        # GBS信息（P_Bk=[˜px_Bk, ˜py_Bk, d_Bk, φ_Bk, θ_Bk]）
        gbs_positions = self.gbs_network.gbs_positions
        uav_gbs_dists = [np.linalg.norm(self.uav_position[:2] - gbs[:2]) for gbs in gbs_positions]
        nearest_indices = np.argsort(uav_gbs_dists)[:Config.NEAR_GBS_N]  # np.argsort返回排序后的下标

        for idx in nearest_indices:
            gbs_pos = gbs_positions[idx]
            relative_gbs_pos = gbs_pos[:2] - self.uav_position[:2]
            relative_gbs_local = np.dot(rot_matrix, relative_gbs_pos)  # 变换到新坐标

            d_b = uav_gbs_dists[idx]
            phi_b = np.arctan2(relative_gbs_pos[1], relative_gbs_pos[0])  # 水平角
            theta_b = np.arctan2(gbs_pos[2] - self.uav_position[2], d_b)  # 垂直角

            state.extend([relative_gbs_local[0], relative_gbs_local[1], d_b, phi_b, theta_b])
            sinr_state.extend([d_b, phi_b, theta_b])
        return np.array(state), np.array(sinr_state)

    def sample_action_space(self):
        """运动学约束采样"""
        actions = []
        for norm_speed in Config.ACTION_SPEEDS:
            for norm_angle in Config.ACTION_ANGLES:
                phi = self.uav_orientation + norm_angle * Config.MAX_T_R * Config.DT
                phi = (phi + math.pi) % (2 * math.pi) - math.pi  # 归一化到[-π, π]

                speed = norm_speed * Config.MAX_V
                velocity_x = speed * np.cos(phi)
                velocity_y = speed * np.sin(phi)
                actions.append([velocity_x, velocity_y, speed, phi])
        # 添加当前动作
        actions.append([self.uav_velocity[0], self.uav_velocity[1], self.uav_speed, self.uav_orientation])
        return np.array(actions)

    def pred_next_position(self, action):
        """下一个位置"""
        next_position = self.uav_position.copy()
        next_position[:2] += action[:2] * Config.DT
        return next_position
