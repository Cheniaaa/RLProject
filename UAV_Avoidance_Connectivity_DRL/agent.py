import random

import torch
from torch import optim
from core import ValueNetwork, SINRNetwork, ReplayBuffer
from environment import *



class ValueBasedAgent:
    def __init__(self, state_dim, sinr_state_dim):
        self.state_dim = state_dim
        self.sinr_state_dim = sinr_state_dim

        # 价值网络
        self.value_net = ValueNetwork(state_dim)

        # SINR预测网络
        self.sinr_net = SINRNetwork(sinr_state_dim)

        # 优化器
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.0001)
        self.sinr_optimizer = optim.Adam(self.sinr_net.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.0001)

        # 经验回放
        self.value_buffer = ReplayBuffer(Config.MEMORY_SIZE)
        self.sinr_buffer = ReplayBuffer(Config.MEMORY_SIZE)
        self.update_num = 0

        self.epsilon = Config.EPSILON_START

    def select_action(self, env: Environment, use_epsilon=True):
        """
        根据论文方法选择动作：枚举动作空间，对每个动作预测下一状态和价值
        Args
        -------
        env: 当前环境
        use_epsilon: 是否使用ε-greedy策略

        Returns
        -------
        action: [velocity_x, velocity_y, uav_speed, phi]
        """
        actions = env.sample_action_space()
        if use_epsilon and random.random() < self.epsilon:
            action_idx = np.random.randint(0, len(actions) - 1)
            return actions[action_idx]
        else:
            # 贪心选择
            return self._greed_action(env, actions)

    def _greed_action(self, env, actions):
        """选择价值最高的动作"""
        max_value = -np.inf
        best_action = actions[0]

        # 保存当前环境
        origin_position = env.uav_position.copy()
        origin_velocity = env.uav_velocity.copy()
        origin_time = env.time_step

        for action in actions:
            _, sinr_state = env.get_state()

            next_position = env.pred_next_position(action)
            # 更新临时环境，获取下一状态
            env.uav_position = next_position
            env.uav_velocity = action[:2]
            next_state, _ = env.get_state()

            # 预测SINR
            predicted_sinr = self.predict_sinr(sinr_state)

            dist_to_dest = np.linalg.norm(next_position[:2] - env.pos_destination[:2])
            reward = env.get_reward(dist_to_dest, predicted_sinr)
            # 计算状态价值
            next_value = self.predict_value(next_state)

            # 总价值
            total_value = reward + Config.GAMMA * next_value
            if total_value > max_value:
                max_value = total_value
                best_action = action

        # 环境恢复
        env.uav_position = origin_position
        env.uav_velocity = origin_velocity
        env.time_step = origin_time
        return best_action

    def predict_value(self, state):
        """预测状态价值"""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            value = self.value_net(state_tensor).item()
        return value

    def predict_sinr(self, sinr_state):
        """预测SINR"""
        sinr_state_tensor = torch.tensor(sinr_state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            sinr_level = self.sinr_net(sinr_state_tensor).item()
        return sinr_level

    def store_experience(self, state, value, sinr_state, sinr_value):
        """经验存储
        Notes:
            原文: Update state-value pairs D with <S,V>, Update location-SINR pairs Dw with <S_B,V_L>
        """
        self.value_buffer.push((state, value))
        self.sinr_buffer.push((sinr_state, sinr_value))

    def update_epsilon(self):
        """epsilon衰减"""
        self.epsilon = max(Config.EPSILON_END, self.epsilon * Config.EPSILON_DECAY)

    def train(self):
        """训练网络"""
        if len(self.value_buffer) < Config.BATCH_SIZE:
            return None, None

        # 训练价值网络
        batch = self.value_buffer.sample(Config.BATCH_SIZE)
        states, values = zip(*batch)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        values_tensor = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1)
        pred_value = self.value_net(states_tensor)
        value_loss = torch.nn.MSELoss()(pred_value, values_tensor)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # 训练SINR预测网络
        sinr_batch = self.sinr_buffer.sample(Config.BATCH_SIZE)
        sinr_states, sinr_values = zip(*batch)

        sinr_states_tensor = torch.tensor(np.array(sinr_states), dtype=torch.float32)
        sinr_values_tensor = torch.tensor(np.array(sinr_values), dtype=torch.float32).unsqueeze(1)
        pred_sinr = self.sinr_net(sinr_states_tensor)
        sinr_loss = torch.nn.MSELoss()(pred_sinr, sinr_values_tensor)
        self.sinr_optimizer.zero_grad()
        sinr_loss.backward()
        self.sinr_optimizer.step()
        return value_loss.item(), sinr_loss.item()

    def save_model(self, name):
        """保存模型
        Args:
            name: 模型名称
        """
        torch.save({
            'value_net': self.value_net.state_dict(),
            'sinr_net': self.sinr_net.state_dict(),
            'epsilon': self.epsilon
        }, Config.MODEL_PATH + name)

    def load_model(self, name):
        """加载模型"""
        model = torch.load(Config.MODEL_PATH + name)
        self.value_net.load_state_dict(model['value_net'])
        self.sinr_net.load_state_dict(model['sinr_net'])
        self.epsilon = model['epsilon']
