"""配置参数文件"""
import math
import numpy as np


class Config:
    # 环境参数
    AREA_SIZE = 1000  # 区域大小 (m)

    # GBS参数
    GBS_N = 12  # 基站数量
    GBS_H = 32  # 基站高度 (m)
    GBS_P = 1  # 基站发射功率 (dBW)
    GAIN_M = 20  # 增益最值
    THETA_TILT = np.deg2rad(10)
    THETA_3DB = np.deg2rad(15)

    # 通信参数
    ALPHA = 2.5
    NOISE_P = 1e-6  # 噪声功率
    SINR_T_DB = -3  # SINR阈值 (dB)
    SINR_T = 10 ** (SINR_T_DB / 10)
    SINR_B = 0.1  # sinr缓冲
    T_T = 5  # 最大允许断开时间 (s)
    N_T = 5  # 每nt个时间步长处理一次连接问题，Δt=Tt/nt 视为一个时间步长

    # UAV参数
    UAV_N = 1  # 智能体个数
    UAV_H = 50  # UAV高度 (m)
    UAV_RADIUS = 5  # UAV半径 (m)
    MAX_V = 10  # 最大速度 (m/s)
    MAX_T_R = math.radians(30)  # 最大转向率 (弧度/s)
    NEAR_GBS_N = 8  # 观察的最近GBS数量

    # 奖励参数
    ALPHA_2 = 1.0
    ALPHA_3 = 2.0
    ALPHA_4 = 0.1

    # 动作空间参数 (论文中使用22个动作)
    ACTION_SPEEDS = [0, 0.5, 1.0]  # 归一化速度
    ACTION_ANGLES = [-1.0, -2 / 3, -1 / 3, 0, 1 / 3, 2 / 3, 1.0]  # 归一化转向

    # 训练参数
    BATCH_SIZE = 200
    LEARNING_RATE = 0.01
    GAMMA = 0.99  # 折扣因子
    EPSILON_START = 0.5
    EPSILON_END = 0.1
    EPSILON_DECAY = 0.995
    MEMORY_SIZE = 30000
    NUM_EPISODES = 100
    NUM_CASES = 20

    # 神经网络参数
    VALUE_NET_HIDDEN = [64, 32, 16]
    SINR_NET_HIDDEN = [32, 16, 8]

    # 仿真参数
    DT = 0.5  # 每个时间步长对应的时间
    DEST_THRESHOLD = 10  # 到达目标的距离阈值

    MODEL_PATH = './model/'
