import random
import math
import time
from collections import deque

import gym
import torch
from torch import nn, optim

env = gym.make("CartPole-v1").env

print(f"actions: {env.action_space}")
print(f"states: {env.observation_space}")


def vectorize_state(state):
    state_vector = torch.from_numpy(state).float()
    return state_vector


class DQN(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=observation_space, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=action_space)
        )

    def forward(self, x_in):
        return self.model(x_in)


### Train ###
print("training...")
start_time = time.time()

# Hyperparameters
gamma = 0.8
epsion_start = 0.9
epsilon_end = 0.05
epsilon_decay = 200

# Training parameters
batch_size = 64
learning_rate = 0.01
memory_size = 1000000

train_epizodes = 100

observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

dqn = DQN(observation_space, action_space)

memory = deque(maxlen=memory_size)

loss_func = nn.MSELoss()
optimizer = optim.SGD(dqn.parameters(), lr=learning_rate)

for epizode in range(train_epizodes):
    done = False
    state = env.reset()
    epochs = 0
    while not done:
        epsilon = epsilon_end + (epsion_start - epsilon_end) * math.exp(-1. * epizode / epsilon_decay)
        if random.random() < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            state_vector = vectorize_state(state)
            q_values = dqn(state_vector)
            action = torch.argmax(q_values).item()

        next_state, reward, done, info = env.step(action)
        reward = reward if not done else -10 # Negative reward when episode done
        memory.append((state, action, next_state, reward))

        state = next_state
        epochs += 1

        # Learn q-function
        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)
            batch_state, batch_action, batch_next_state, batch_reward = zip(*batch)

            batch_state = torch.as_tensor(batch_state, dtype=torch.float)
            batch_action = torch.as_tensor(batch_action, dtype=torch.long)
            batch_next_state = torch.as_tensor(batch_next_state, dtype=torch.float)
            batch_reward = torch.as_tensor(batch_reward, dtype=torch.float)

            # current Q values are estimated by NN for all actions
            current_q_values = dqn(batch_state)
            current_action_q_values = current_q_values.gather(1, batch_action.unsqueeze(1)).squeeze()
            # expected Q values are estimated from actions which gives maximum Q value
            next_q_values = dqn(batch_next_state)
            max_next_q_values, _ = next_q_values.detach().max(1)
            expected_q_values = batch_reward + (gamma * max_next_q_values)

            # loss is measured from error between current and newly expected Q values
            loss = loss_func(current_action_q_values, expected_q_values)

            # backpropagation of loss to NN
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f'epizode: {epizode} epochs: {epochs} epsilon: {epsilon}')

print(f"traing time: {time.time() - start_time:.2f}s")

### Test ###
test_epizodes = 100
avg_epochs_per_epizode = 0.0
avg_rewards_per_epizode = 0.0

for epizode_index in range(test_epizodes):
    epochs, rewards = 0, 0
    done = False
    state = env.reset()
    while not done:
        state_vector = vectorize_state(state)
        q_values = dqn(state_vector)
        action = torch.argmax(q_values).item()

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
        env.render()
        time.sleep(0.1)

        state_vector = vectorize_state(state)
        q_values = dqn(state_vector)
        action = torch.argmax(q_values).item()

        state, reward, done, info = env.step(action)
    print('start next simulation [Y/n]', end=' ')