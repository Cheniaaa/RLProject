from collections import deque

from environment import Environment, GroundBaseStation
from method import generate_positions, start_trajectory, plot_trained_date, write_pos_json, read_pos_json
from config import Config
import numpy as np
import torch

# env = Environment(None)
# env.reset()
# print(env.sample_action_space())

# gbs = GroundBaseStation()
# env = Environment(gbs)
# start, end = generate_positions()
# env.reset(start, end)
# start_trajectory(env.gbs_network.gbs_positions, start, end)

# print(gbs.gbs_positions)

# pre_position = np.array([10, 20, 30], dtype=float)
# action = np.array([-1, 2, 5, 0.5])
# next_position = pre_position.copy()
# next_position[:2] += action[:2] * Config.DT
# print(next_position)

arr = [1, 2, 3, 4]
# # arr.append([5, 6, 7])
# arr.extend([8, 9, 0])
# brr = arr
# brr.extend([5, 6, 7])
# print(np.array(arr))
#
# print(torch.tensor(np.array(arr)).unsqueeze(1))
# print(np.array(arr).reshape(-1, 1))
# print(brr)

# buffer = deque(maxlen=10)
# buffer.append((1,1,1))
# buffer.append((2,2,2,2))
# buffer.append([1,2,3,3])
# print(buffer[2][1])
# print(type(buffer[2]))
# print(type(buffer[1]))

# print(torch.cuda.is_available())
# arr = np.array([1, 2, 3])
# brr = np.array([4, 5, 6, 7])
# brr[:3] = brr[:3] + arr
# print(brr)

# print(10 ** (-3 / 10))
#
# speed = np.array([1.0, 0])
# position = np.array([0.0, 1.0])
# theta = np.pi / 4
# cos_theta = np.cos(theta)
# sin_theta = np.sin(theta)
# rot_matrix = np.array([[cos_theta, -sin_theta],
#                        [sin_theta, cos_theta]])
# print(np.dot(rot_matrix, speed))
# print(np.dot(speed, rot_matrix))
# print(np.dot(position, rot_matrix))
# print(np.dot(rot_matrix, position))

# rewards = [1, 3, 5, 7, 2, 3, 5, 6, 1, 2, 1, 4, 5]
# value_losses = [1, 2, 3, 4]
# sinr_losses = [1, 2, 3, 4]
# plot_trained_date(rewards, value_losses, sinr_losses)


# 你的数据
# GBS_POSITION = np.array([[1, 2], [2, 4], [3, 5], [4, 6]]).tolist()
# START_POSITION = [1, 2, 3]
# END_POSITION = [2, 3, 4]
#
# # 将数据组织成字典
# data = {
#     "GBS_POSITION": GBS_POSITION,
#     "START_POSITION": START_POSITION,
#     "END_POSITION": END_POSITION
# }

file_path = './data/position.json'

# write_pos_json(file_path, data)
data = read_pos_json(file_path)

GBS_POSITION = data['GBS_POSITION']
START_POSITION = data['START_POSITION']
END_POSITION = data['END_POSITION']

print("GBS_POSITION:", GBS_POSITION)
print("START_POSITION:", START_POSITION)
print("END_POSITION:", END_POSITION)
