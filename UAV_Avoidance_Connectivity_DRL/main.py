from agent import ValueBasedAgent
from config import Config
from environment import GroundBaseStation, Environment
import numpy as np
from method import plot_trained_date, visualize_trajectory


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
        # 在每个训练周期（episode）中，运行 n 个随机初始化的训练场景，收集完整的轨迹数据
        for case in range(Config.NUM_CASES):
            state, _ = env.reset()

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
                    if case!=0 and episode!=0 and case * episode % 500 == 0:
                        visualize_trajectory(env.gbs_network.gbs_positions, env.pos_origin, env.pos_destination,
                                             np.array(env.trajectory), info["success"])
                    if info["success"]:
                        success_count += 1
                    break
            episode_reward_list.append(total_reward)

            # TODO: 计算状态价值
            # 获取状态价值
            # 原文：V_{i,0:Ti} <- updateValue(S_jn_{i,0:Ti}, R_{i,0:Ti}, ξ)
            values = []

            for i in range(len(trajectory_rewards)):
                v = sum([Config.GAMMA ** (j - i) * trajectory_rewards[j]
                         for j in range(i, len(trajectory_rewards))])
                values.append(v)

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
        episode_rewards.append(avg_reward)
        episode_success_rates.append(success_rate)
        if episode % 10 == 0:
            print(f"Episode {episode}/{Config.NUM_EPISODES}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Success Rate: {success_rate:.2%}, "
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


def test():
    gbs_network = GroundBaseStation()
    env = Environment(gbs_network)
    state_dim = 6 + Config.NEAR_GBS_N * 5
    sinr_state_dim = Config.NEAR_GBS_N * 3
    agent = ValueBasedAgent(state_dim, sinr_state_dim)
    agent.load_model("RLTCW-SP.pth")

    end_info = None
    env.reset()
    max_steps = 500
    for step in range(max_steps):
        action = agent.select_action(env, use_epsilon=False)
        state, reward, done, info = env.step(action)
        if done:
            end_info = info
            break
    trajectory = np.array(env.trajectory)
    visualize_trajectory(env.gbs_network.gbs_positions, env.pos_origin, env.pos_destination, trajectory,
                         end_info['success'])
    print(f"\n测试结果:")
    print(f"  成功: {end_info['is_success']}")
    print(f"  到达距离: {end_info['dist_to_dest']:.2f}m")
    print(f"  最终SINR: {10 * np.log10(end_info['sinr']):.2f} dB")


if __name__ == '__main__':
    train()
