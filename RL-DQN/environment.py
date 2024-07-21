import numpy as np

class Environment:
    def __init__(self):
        self.initial_battery_level = 100
        self.initial_bandwidth = 10
        self.battery_level = self.initial_battery_level # Battery level in percentage
        self.bandwidth = self.initial_bandwidth # Network bandwidth in Mbps
        self.task_size = np.random.randint(1, 10) # Task size in bytes

    def reset(self):
        self.battery_level = self.initial_battery_level
        self.bandwidth = self.initial_bandwidth
        self.task_size = np.random.randint(1, 10)
        return np.array([self.battery_level, self.bandwidth, self.task_size])

    def step(self, action):
        if action == 0:
            reward, done = self.offload_task() # Offload to cloud
        else:
            reward, done = self.execute_task() # Execute locally
        self.task_size = np.random.randint(1, 10)
        next_state = np.array([self.battery_level, self.bandwidth, self.task_size])
        return next_state, reward, done

    def offload_task(self):
        self.battery_level -= 5 # Offloading consumes less battery and more bandwidth
        self.bandwidth -= 2
        reward = (self.battery_level / 10) - self.bandwidth
        if self.battery_level <= 0 or self.bandwidth <= 0:
            return reward-10, True
        return reward, False

    def execute_task(self):
        self.battery_level -= 10 # Local execution consumes more battery and no bandwidth
        reward = (self.battery_level / 10)
        if self.battery_level <= 0:
            return reward-10, True
        return reward, False

    def render(self):
        print(f"Battery Level: {self.battery_level}")
        print(f"Bandwidth: {self.bandwidth}")
        print(f"Task Size: {self.task_size}")
