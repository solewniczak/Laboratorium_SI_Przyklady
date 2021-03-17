import time

import numpy as np
import gym

env = gym.make('MountainCarContinuous-v0')
print(env.action_space, env.action_space.low, env.action_space.high)
print(env.observation_space, env.observation_space.low, env.observation_space.high)

observation = env.reset()
for t in range(1000):
    env.render()
    time.sleep(0.01)
    # print(observation)
    position, velocity = observation
    # action = env.action_space.sample() # akcja losowa
    action = np.array([1]) # maksymalna siła w prawą stronę
    observation, reward, done, info = env.step(action)
    if done:
        print(f"Episode finished after {t} time steps.")
        break
env.close()