import numpy as np
import gym

env = gym.make('MountainCarContinuous-v0')
print(env.action_space)
print(env.action_space.high)
print(env.action_space.low)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

observation = env.reset()
max_position = env.observation_space.low[0]
for t in range(1000):
    env.render()
    # print(observation)
    position, velocity = observation
    if position > max_position:
        max_position = position

    # action = env.action_space.sample()
    action = np.array([1])
    observation, reward, done, info = env.step(action)
    if done:
        print(f"Episode finished after {t} time steps.")
        print(f"Maximum position: {max_position}")
        break
env.close()