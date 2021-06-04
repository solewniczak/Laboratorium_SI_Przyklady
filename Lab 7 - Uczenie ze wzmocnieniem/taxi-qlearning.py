import os
import random
import time

import gym
import numpy as np

env = gym.make("Taxi-v3").env

print(f"actions: {env.action_space}")
print(f"states: {env.observation_space}")

### Train ###
print("training...")
start_time = time.time()

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1
train_epizodes = 100000

q_table = np.zeros([env.observation_space.n, env.action_space.n])

for i in range(train_epizodes):
    done = False
    state = env.reset()
    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state

print(f"traing time: {time.time() - start_time:.2f}s")

### Test ###
test_epizodes = 10
avg_epochs_per_epizode = 0.0
avg_rewards_per_epizode = 0.0

for epizode_index in range(test_epizodes):
    epochs, rewards = 0, 0
    done = False
    state = env.reset()
    while not done:
        action = np.argmax(q_table[state])
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
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)
    print('start next simulation [Y/n]', end=' ')