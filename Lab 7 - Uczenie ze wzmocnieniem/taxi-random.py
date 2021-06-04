import time
import os

import gym

env = gym.make("Taxi-v3").env

print(f"actions: {env.action_space}")
print(f"states: {env.observation_space}")


### Test ###
test_epizodes = 10
avg_epochs_per_epizode = 0.0
avg_rewards_per_epizode = 0.0

for epizode_index in range(test_epizodes):
    epochs, rewards = 0, 0
    done = False
    state = env.reset()
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        epochs += 1
        rewards += reward
    avg_epochs_per_epizode += (epochs - avg_epochs_per_epizode) / (epizode_index + 1)
    avg_rewards_per_epizode += (rewards - avg_rewards_per_epizode) / (epizode_index + 1)


print(f'avg epochs per epizode: {avg_epochs_per_epizode}')
print(f'avg rewards per epizode: {avg_rewards_per_epizode}')

### Simulation ###
print('start simulation [Y/n]', end=' ')
while input() != 'n':
    done = False
    state = env.reset()
    while not done:
        os.system('cls' if os.name == 'nt' else 'clear')
        env.render()
        time.sleep(0.1)
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
    print('start next simulation [Y/n]', end=' ')