import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from config import Config


def generate_positions():
    while True:
        pos_origin = np.array([
            np.random.uniform(10, Config.AREA_SIZE - 10),
            np.random.uniform(10, Config.AREA_SIZE - 10),
        ])
        pos_destination = np.array([
            np.random.uniform(10, Config.AREA_SIZE - 10),
            np.random.uniform(10, Config.AREA_SIZE - 10),
        ])
        if np.linalg.norm(pos_destination[:2] - pos_origin[:2]) >= 50:
            break
    return pos_origin, pos_destination


def compute_gbs_antenna_gain(distance, gbs_h, uav_h, theta_tilt, theta_3db, gain_m):
    """计算GBS天线增益"""
    if distance == 0:
        distance = 1e-6
    angle = np.arctan2((gbs_h - uav_h), distance)
    gain_v = -min(12 * ((angle - theta_tilt) / theta_3db) ** 2, gain_m)
    gain_h = 0
    return gain_v + gain_h  # dB


def compute_uav_antenna_gain(distance, gbs_h, uav_h):
    """计算UAV天线增益"""
    if distance == 0:
        return 0
    sin_theta = (uav_h - gbs_h) / np.sqrt(distance ** 2 + (uav_h - gbs_h) ** 2)
    return sin_theta


def compute_path_loss(distance, gbs_h, uav_h, alpha):
    """计算路径损耗"""
    distance_3d = np.sqrt(distance ** 2 + (gbs_h - uav_h) ** 2)
    if distance_3d == 0:
        distance_3d = 1e-6
    return distance_3d ** alpha


def to_linear(k):
    """转化为线性常量"""
    return 10 ** (k / 10)


def compute_signal(distance):
    gbs_p_linear = to_linear(Config.GBS_P)
    gain_gbs = compute_gbs_antenna_gain(distance, Config.GBS_H, Config.UAV_H, Config.THETA_TILT, Config.THETA_3DB,
                                        Config.GAIN_M)
    gain_gbs_linear = to_linear(gain_gbs)
    gain_uav = compute_uav_antenna_gain(distance, Config.GBS_H, Config.UAV_H)
    pass_loss = compute_path_loss(distance, Config.GBS_H, Config.UAV_H, Config.ALPHA)
    return gbs_p_linear * gain_gbs_linear * gain_uav / pass_loss


def plot_trained_date(rewards, value_losses, sinr_losses):
    """绘制训练曲线"""
    fig, (ax1, ax2, ax3) = plt.subplots(2, 2, figsize=(15, 5))

    ax1.plot(rewards)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Training Rewards')
    ax1.grid(True)

    ax2.plot(value_losses)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Value Loss')
    ax2.set_title('Training Value Loss')
    ax2.grid(True)

    ax3.plot(sinr_losses)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('SINR Loss')
    ax3.set_title('Training SINR Loss')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


def visualize_trajectory(gbs_positions, start_position, end_position, trajectory, is_success, episode):
    """绘制轨迹

    Args:
        gbs_positions: 基站位置
        start_position: 起点
        end_position: 终点
        trajectory: 无人机轨迹
        is_success: 是否到达终点
        episode: 训练轮数
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    for i, gbs in enumerate(gbs_positions):
        ax.plot(gbs[0], gbs[1], 'g^', markersize=15,
                label='GBS' if i == 0 else '')
        circle = Circle((gbs[0], gbs[1]), 20, fill=True, alpha=0.1, color='yellow')
        ax.add_patch(circle)

    # 绘制轨迹
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b.-', linewidth=2,
            markersize=3, label='Trajectory')
    ax.plot(start_position[0], start_position[1], 'ro', markersize=15, label='Start')
    ax.plot(end_position[0], end_position[1], 'r*', markersize=20, label='Goal')

    ax.set_xlim(0, Config.AREA_SIZE)
    ax.set_ylim(0, Config.AREA_SIZE)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'UAV Trajectory (Episode:{episode}  Success: {is_success})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def start_trajectory(gbs_positions, start_position, end_position):
    """绘制轨迹

    Args:
        gbs_positions: 基站位置
        start_position: 起点
        end_position: 终点
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    for i, gbs in enumerate(gbs_positions):
        ax.plot(gbs[0], gbs[1], 'g^', markersize=15,
                label='GBS' if i == 0 else '')
        circle = Circle((gbs[0], gbs[1]), 20, fill=True, alpha=0.1, color='yellow')
        ax.add_patch(circle)
    ax.plot(start_position[0], start_position[1], 'ro', markersize=15, label='Start')
    ax.plot(end_position[0], end_position[1], 'r*', markersize=20, label='Goal')
    ax.set_xlim(0, Config.AREA_SIZE)
    ax.set_ylim(0, Config.AREA_SIZE)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
