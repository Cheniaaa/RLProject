from collections import deque

from environment import Environment, GroundBaseStation
from config import Config
import numpy as np
import torch

# env = Environment(None)
# env.reset()
# print(env.sample_action_space())

# gbs = GroundBaseStation()
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

print(10**(-3/10))