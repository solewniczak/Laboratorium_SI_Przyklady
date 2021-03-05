import numpy as np
import gym

env = gym.make('Pendulum-v0')
print(env.action_space)
print(env.action_space.high)
print(env.action_space.low)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

observation = env.reset()
for t in range(1000):
    env.render()
    print(observation)

    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        print(f"Episode finished after {t} time steps.")
        break
env.close()