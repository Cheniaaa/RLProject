import numpy as np
from matplotlib import pyplot as plt

from config import Config


def compute_gbs_antenna_gain(distance, gbs_h, uav_h, theta_tilt, theta_3db, gain_m):
    """计算GBS天线增益"""
    if distance == 0:
        distance = 1e-6
    angle = np.arctan((gbs_h - uav_h) / distance)
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


def plot_trained_date(rewards, success_rates, value_losses, sinr_losses):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(rewards)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Training Rewards')
    ax1.grid(True)

    ax2.plot(success_rates)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Training Success Rate')
    ax2.grid(True)

    ax2.plot(value_losses)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Value Loss')
    ax2.set_title('Training Value Loss')
    ax2.grid(True)

    ax2.plot(sinr_losses)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('SINR Loss')
    ax2.set_title('Training SINR Loss')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
