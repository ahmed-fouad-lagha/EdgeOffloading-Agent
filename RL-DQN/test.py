import os
print(os.getcwd())
import numpy as np
from dqn_agent import DQLAgent

# Test the agent with a specific state

def test_agent(agent, state):
    state = np.reshape(state, [1, agent.state_size])
    action = agent.act(state)
    action_name = "Offload" if action == 0 else "Execute Locally"
    print(f"Given state: {state}, Action chosen: {action_name}")

if __name__ == "__main__":
    # Load the trained agent model
    state_size = 3  # Battery level, bandwidth, task size
    action_size = 2  # Offload or execute locally
    agent = DQLAgent(state_size, action_size)
    agent.load("dqn_mcc_model.h5")  # Make sure the model weights are saved in this file during training

    # Test the agent with a specific state
    test_state1 = np.array([20, 8, 5])  # Example state: 20% battery, 8 bandwidth, task size 5
    test_agent(agent, test_state1)
    
    test_state1 = np.array([80, 2, 12])  # Example state: 80% battery, 2 bandwidth, task size 12
    test_agent(agent, test_state1)
