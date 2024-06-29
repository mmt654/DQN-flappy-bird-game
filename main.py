"""
    Deep Q Network AI Flappy Bird Project
    Author: Taylor M
    Date: 6/20/2024
    Est. Time: 5 hours
"""

# Import Libraries

import pygame
import sys
import os
from FlappyBird import FlappyBird
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import OrderedDict

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # define the layers
        self.layers = nn.Sequential(
            nn.Linear(9, 20), # Input Layer (state size: 9)
            nn.ReLU(),
            nn.Linear(20, 128), # First Hidden Layer
            nn.ReLU(),
            nn.Linear(128, 256), # Second Hidden Layer 
            nn.ReLU(),
            nn.Linear(256, 2) # output layer with 2 actions flap or not to flap
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

class ReplayMemory: 
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, expereince):
        if len(self.memory) < self.capacity:
            self.memory.append(expereince)
        else:
            self.memory[self.push_count % self.capacity] = expereince
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size
    
class EpsilonStrategy: # Using Epsilon Greedy Strategy 
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-current_step * self.decay)
    
class Agent:
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim = 1).to(self.device)
            
def plot(values, moving_avg_period):
    # this creates a seperate window that allows user to monitor agent's performance and game score per episode.
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    print("Episode", len(values), "\n", moving_avg_period, " episode moving avg:", moving_avg[-1])

def get_moving_average(period, values):
    # calculates the average game points recieved over a specified interval which is at 1000
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension = 0, size = period, step = 1).mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period - 1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()
    
def get_tensors(experiences):
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return t1, t2, t3, t4

class QValues:
    device = torch.device("cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

# Main Program
def main():
    #Hyperparameters 
    batch_size = 64
    gamma = 0.99
    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.000001
    target_update = 5000
    memory_size = 100000
    lr = 0.0001
    num_episodes = 100000
    weight_save = 100
    lr_update = 1000
    lr_decay = 0.995

    device = torch.device("cpu")
    strategy = EpsilonStrategy(eps_start, eps_end, eps_decay)
    agent = Agent(strategy, 2, device)
    memory = ReplayMemory(memory_size)

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.eval()
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

    if os.path.exists("./Optimizer_weight/weight.pt"):
        optimizer.load_state_dict(torch.load("Optimizer_weight/weight.pt"))
        print("Optimizer weights loaded successfully")

    score = []

    if os.path.exists("Policy_weight/weight.pt") and os.path.exists("Target_weight/weight.pt"):
        policy_net.load_state_dict(torch.load("Policy_weight/weight.pt"))
        target_net.load_state_dict(torch.load("Target_weight/weight.pt"))
        print("Policy and target weights loaded successfully")
    else:
        target_net.load_state_dict(policy_net.state_dict())
        torch.save(policy_net.state_dict(), "Policy_weight/weight.pt")
        torch.save(target_net.state_dict(), "Target_weight/weight.pt")
        torch.save(optimizer.state_dict(), "Optimizer_weight/weight.pt")
        print("Policy, target, and optimizer weights saved successfully for the first time")

    pygame.mixer.pre_init(frequency=44100, size=16, channels=1, buffer=512)
    pygame.init()
    
    # Enviorment Manager
    #FlappyBird.py
    env_manager = FlappyBird()

    for episode in range(num_episodes):

        env_manager.start_game(env_manager)

        state = torch.tensor([env_manager.get_state(env_manager)]).to(device).float()
        cumulative_reward = 0
        step_reward = []

        while True:
            action = agent.select_action(state, policy_net)
            reward, next_state = env_manager.step(env_manager, action)
            reward = torch.tensor([reward]).to(device).float()
            cumulative_reward += reward.item()
            step_reward.append(reward.item())
            next_state = torch.tensor([next_state]).to(device).float()
            memory.push(Experience(state, action, next_state, reward))
            state = next_state

            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states = get_tensors(experiences)

                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * gamma) + rewards

                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if env_manager.is_done(env_manager):
                score.append(env_manager.get_score(env_manager))
                plot(score, 1000)
                print (f"Epsiode: {episode + 1}, score: {env_manager.get_score(env_manager)}, Step rewards : {step_reward}Cumulative Reward: {cumulative_reward}") # used to print system messages to monitor rewards awarded each step and total amount rewarded for that episode
                print("Exploration : ", round(strategy.get_exploration_rate(agent.current_step) * 100), "%")
                if memory.can_provide_sample(batch_size):
                    print("Loss : ", loss.data)
                break
        # saving the weights based on certain intervals to help with training over multiple sessions    
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(target_net.state_dict(), "Target_weight/weight.pt")
            print("Target weights updated and saved successfully")

        if episode % weight_save == 0:
            torch.save(policy_net.state_dict(), "Policy_weight/weight.pt")
            torch.save(optimizer.state_dict(), "Optimizer_weight/weight.pt")
            print("Policy and optimizer weights saved successfully")

        if episode % lr_update == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
            print("Learning rate updated to: ", optimizer.param_groups[0]['lr'])

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
