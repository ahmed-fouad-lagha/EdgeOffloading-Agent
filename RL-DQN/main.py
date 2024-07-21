import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from dqn_agent import DQLAgent

def evaluate_agent(agent, env, episodes=100):
    total_rewards = []
    total_battery_consumed = []
    total_tasks_completed = []
    total_latency = []

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        episode_reward = 0
        battery_consumed = 0
        tasks_completed = 0
        latency = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            episode_reward += reward
            battery_consumed += (env.initial_battery_level - env.battery_level)
            tasks_completed += 1
            latency += 1
            state = next_state

        total_rewards.append(episode_reward)
        total_battery_consumed.append(battery_consumed)
        total_tasks_completed.append(tasks_completed)
        total_latency.append(latency)

    avg_reward = np.mean(total_rewards)
    avg_battery_consumed = np.mean(total_battery_consumed)
    avg_tasks_completed = np.mean(total_tasks_completed)
    avg_latency = np.mean(total_latency)

    print(f"Average Reward: {avg_reward}")
    print(f"Average Battery Consumed: {avg_battery_consumed}")
    print(f"Average Tasks Completed: {avg_tasks_completed}")
    print(f"Average Latency: {avg_latency}")

    # Plotting the rewards over episodes
    plt.plot(total_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards over Episodes')
    plt.show()

def test_agent(agent, state):
    state = np.reshape(state, [1, state_size])
    action = agent.act(state)
    action_name = "Offload" if action == 0 else "Execute Locally"
    print(f"Given state: {state}, Action chosen: {action_name}")

if __name__ == "__main__":
    env = Environment()
    state_size = 3  # Battery level, bandwidth, task size
    action_size = 2  # Offload or execute locally
    agent = DQLAgent(state_size, action_size)
    episodes = 5 # 1000
    batch_size = 32

    rewards = []

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        for time in range(200):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        rewards.append(total_reward)
        # Optionally, save the model weights
        agent.save("dqn_mcc_model.h5")

    # Plot the training rewards
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards over Episodes')
    plt.show()

    # Evaluate the agent
    evaluate_agent(agent, env, episodes=100)

    # Test the agent with a specific state
    test_state = np.array([20, 8, 5])  # Example state: 80% battery, 8 bandwidth, task size 5
    test_agent(agent, test_state)
