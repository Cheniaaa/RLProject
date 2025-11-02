from agent import ValueBasedAgent
from config import Config
from environment import GroundBaseStation, Environment
import numpy as np
from method import *


def train():
    """模型训练"""
    # 初始化GBS网络
    print("初始化GBS网络...\n")
    gbs_network = GroundBaseStation()

    # 初始化环境
    print("初始化环境...\n")
    env = Environment(gbs_network)

    # 初始化智能体
    print("初始化智能体...\n")
    state_dim = 6 + Config.NEAR_GBS_N * 5
    sinr_state_dim = Config.NEAR_GBS_N * 3
    agent = ValueBasedAgent(state_dim, sinr_state_dim)

    # 训练
    print("开始训练...\n")
    print(f"训练参数: {Config.NUM_EPISODES} episodes, {Config.NUM_CASES} cases/episode")
    episode_rewards = []
    episode_success_rates = []
    value_losses = []
    sinr_losses = []

    for episode in range(Config.NUM_EPISODES):
        episode_reward_list = []
        success_count = 0
        disconnected_count = 0
        out_time_count = 0
        # 在每个训练周期（episode）中，运行 n 个随机初始化的训练场景，收集完整的轨迹数据
        for case in range(Config.NUM_CASES):
            start_pos, end_pos = generate_positions()
            state, _ = env.reset(start_pos, end_pos)

            # 收集轨迹
            trajectory_states = []
            trajectory_rewards = []
            trajectory_sinr_states = []
            trajectory_sinr_values = []

            total_reward = 0

            while True:
                _, sinr_state = env.get_state()
                action = agent.select_action(env, use_epsilon=True)

                next_state, reward, done, info = env.step(action)

                trajectory_states.append(state)
                trajectory_rewards.append(reward)
                trajectory_sinr_states.append(sinr_state)
                trajectory_sinr_values.append(info['sinr'])

                total_reward += reward
                state = next_state

                if done:
                    if info["success"]:
                        success_count += 1
                        # visualize_trajectory(env.gbs_network.gbs_positions, env.pos_origin, env.pos_destination,
                        #                      np.array(env.trajectory), info["success"])
                        break
                    elif info["disconnected"]:
                        disconnected_count += 1
                        break
                    elif info["out_time"]:
                        out_time_count += 1
                        break
            episode_reward_list.append(total_reward)

            # TODO: 计算状态价值
            # 获取状态价值
            # 原文：V_{i,0:Ti} <- updateValue(S_jn_{i,0:Ti}, R_{i,0:Ti}, ξ)
            values = []
            v = 0.0
            for r in reversed(trajectory_rewards):
                v = r + Config.GAMMA * v
                values.insert(0, v)

            for i in range(len(trajectory_states)):
                agent.store_experience(
                    trajectory_states[i],
                    values[i],
                    trajectory_sinr_states[i],
                    trajectory_sinr_values[i]
                )

        value_loss, sinr_loss = agent.train()
        value_losses.append(value_loss)
        sinr_losses.append(sinr_loss)
        agent.update_epsilon()

        # 记录统计
        avg_reward = np.mean(episode_reward_list)
        success_rate = success_count / Config.NUM_CASES
        disconnected_rate = disconnected_count / Config.NUM_CASES
        out_time_rate = out_time_count / Config.NUM_CASES
        episode_rewards.append(avg_reward)
        episode_success_rates.append(success_rate)
        if episode % 10 == 0:
            print(f"Episode {episode}/{Config.NUM_EPISODES}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Success Rate: {success_rate:.2%}, "
                  f"Disconnected Rate: {disconnected_rate:.2%}, "
                  f"out-time Rate: {out_time_rate:.2%}, "
                  f"Epsilon: {agent.epsilon:.3f}")
            if value_loss is not None:
                print(f"  Value Loss: {value_loss:.4f}, SINR Loss: {sinr_loss:.4f}")

    print("保存模型...\n")
    agent.save_model("RLTCW-SP.pth")
    # 绘制训练曲线
    print("绘制训练曲线...\n")
    plot_trained_date(episode_rewards, episode_success_rates, value_losses, sinr_losses)

    print("训练完成！\n")
    print("=" * 60)


def train_fixed():
    """固定任务点模型训练"""

    # 初始化GBS网络
    print("初始化GBS网络...\n")
    gbs_network = GroundBaseStation()

    # 初始化环境
    print("初始化环境...\n")
    env = Environment(gbs_network)

    # 初始化智能体
    print("初始化智能体...\n")
    state_dim = 6 + Config.NEAR_GBS_N * 5
    sinr_state_dim = Config.NEAR_GBS_N * 3
    agent = ValueBasedAgent(state_dim, sinr_state_dim)

    # 训练
    print("开始训练...\n")
    print(f"训练参数: {Config.NUM_EPISODES} episodes")
    episode_rewards = []
    episode_reward_list = []
    value_losses = []
    sinr_losses = []
    start_pos, end_pos = generate_positions()
    start_trajectory(env.gbs_network.gbs_positions, start_pos, end_pos)

    position_json = {
        "GBS_POSITION": env.gbs_network.gbs_positions.tolist(),
        "START_POSITION": start_pos.tolist(),
        "END_POSITION": end_pos.tolist()
    }
    write_pos_json(Config.POS_FILE_PATH, position_json)

    for episode in range(Config.NUM_EPISODES):
        # success_count = 0
        state, _ = env.reset(start_pos, end_pos)

        # 收集轨迹
        trajectory_states = []
        trajectory_rewards = []
        trajectory_sinr_states = []
        trajectory_sinr_values = []

        total_reward = 0
        value_loss = None
        sinr_loss = None

        while True:
            _, sinr_state = env.get_state()
            action = agent.select_action(env, use_epsilon=True)

            next_state, reward, done, info = env.step(action)

            trajectory_states.append(state)
            trajectory_rewards.append(reward)
            trajectory_sinr_states.append(sinr_state)
            trajectory_sinr_values.append(info['sinr'])

            total_reward += reward
            state = next_state

            if done:
                break
        episode_reward_list.append(total_reward)

        # 获取状态价值
        # 原文：V_{i,0:Ti} <- updateValue(S_jn_{i,0:Ti}, R_{i,0:Ti}, ξ)
        values = []
        v = 0.0
        for r in reversed(trajectory_rewards):
            v = r + Config.GAMMA * v
            values.insert(0, v)

        for i in range(len(trajectory_states)):
            agent.store_experience(
                trajectory_states[i],
                values[i],
                trajectory_sinr_states[i],
                trajectory_sinr_values[i]
            )

        # 网络更新频率
        if episode % 5 == 0:
            value_loss, sinr_loss = agent.train()
            value_losses.append(value_loss)
            sinr_losses.append(sinr_loss)
        agent.update_epsilon()

        if episode % 20 == 0:
            visualize_trajectory(env.gbs_network.gbs_positions, env.pos_origin, env.pos_destination,
                                 np.array(env.trajectory), info["success"], episode)

        if episode % 10 == 0:
            avg_reward = np.mean(episode_reward_list)
            episode_reward_list = []
            episode_rewards.append(avg_reward)

            print(f"Episode {episode}/{Config.NUM_EPISODES}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
            if value_loss is not None:
                print(f"  Value Loss: {value_loss:.4f}, SINR Loss: {sinr_loss:.4f}")

    print("保存模型...\n")
    agent.save_model("RLTCW-SP-FIXED.pth")
    # 绘制训练曲线
    print("绘制训练曲线...\n")
    plot_trained_date(episode_rewards, value_losses, sinr_losses)

    print("训练完成！\n")
    print("=" * 60)


def test():
    gbs_network = GroundBaseStation()
    env = Environment(gbs_network)
    state_dim = 6 + Config.NEAR_GBS_N * 5
    sinr_state_dim = Config.NEAR_GBS_N * 3
    agent = ValueBasedAgent(state_dim, sinr_state_dim)
    agent.load_model("RLTCW-SP-FIXED.pth")

    info = None
    start_position, end_position = generate_positions()
    env.reset(start_position, end_position)
    done = False
    while not done:
        action = agent.select_action(env, use_epsilon=False)
        state, reward, done, info = env.step(action)
        if info["disconnected"]:
            break
        elif info["out_of_time"]:
            break

    trajectory = np.array(env.trajectory)
    visualize_trajectory(env.gbs_network.gbs_positions, env.pos_origin, env.pos_destination, trajectory,
                         info['success'], 0)
    print(f"\n测试结果:")
    print(f"  成功: {info['success']}")
    print(f"  断连: {info['disconnected']}")
    print(f"  超时: {info['out_of_time']}")
    print(f"  到达距离: {info['dist_to_dest']:.2f}m")
    print(f"  最终SINR: {10 * np.log10(info['sinr']):.2f} dB")


if __name__ == '__main__':
    # train()
    train_fixed()
    # test()
