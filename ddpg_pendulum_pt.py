import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import copy
from collections import namedtuple, deque
from utils import get_experiment_name
import sys

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """ Policy model. """
    def __init__(self, d_obs, d_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(d_obs, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, d_action)

        #self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        #self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        #self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    """ Q-value Model. """
    def __init__(self, d_obs, d_action):
        super(Critic, self).__init__()
        self.fcs1 = nn.Linear(d_obs, 400)
        self.fc2 = nn.Linear(400 + d_action, 300)
        self.fc3 = nn.Linear(300, 1)

        #self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        #self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        #self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs, action):
        xs = F.relu(self.fcs1(obs))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DDPGAgent():
    """DDPG Agent implementation."""
    def __init__(self, d_obs, d_action, buffer_size, batch_size, tau, gamma):
        self.d_obs = d_obs
        self.d_action = d_action
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma

        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.noise = OUNoise(d_action)

        # Policy network (current and target)
        self.actor = Actor(d_obs, d_action).to(device)
        self.actor_target = Actor(d_obs, d_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        # Q-value network (current and target)
        self.critic = Critic(d_obs, d_action).to(device)
        self.critic_target = Critic(d_obs, d_action).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=0)

    def add_to_buffer(self, state, action, reward, next_state, done):
        """Save experience in replay memory."""
        self.memory.add(state, action, reward, next_state, done)

    def action(self, obs, add_noise=True):
        """Returns action given observation using current policy."""
        obs = torch.from_numpy(obs).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs).cpu().data.numpy()
        self.actor.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def update(self):
        """Update policy and Q-value weights."""
        obs, actions, rewards, next_obs, dones = self.memory.sample()

        # Update Q-value network (critic).
        next_actions = self.actor_target(next_obs)
        Q_targets_next = self.critic_target(next_obs, next_actions)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic(obs, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Policy network (actor)
        pred_actions = self.actor(obs)
        actor_loss = -self.critic(obs, pred_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.target_update(self.critic, self.critic_target)
        self.target_update(self.actor, self.actor_target)

        return critic_loss, actor_loss

    def target_update(self, model, target_model):
        """Soft update of model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process.
       differential equation: dx_t =  theta (mu - x_t) dt + sigma dW_t
       mu: drift
       W_t: Wiener process
    """
    def __init__(self, size, mu=0., theta=0.15, sigma=0.1):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma # volatility
        self.delta_t = 0.01
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.delta_t + self.sigma * np.sqrt(self.delta_t) * np.array([random.gauss(0,1) for i in range(self.size)])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        samples = random.sample(self.memory, k=self.batch_size)

        obs = torch.from_numpy(np.vstack([e.state for e in samples if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in samples if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in samples if e is not None])).float().to(device)
        next_obs = torch.from_numpy(np.vstack([e.next_state for e in samples if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in samples if e is not None]).astype(np.uint8)).float().to(device)

        return (obs, actions, rewards, next_obs, dones)

    def __len__(self):
        return len(self.memory)

# Hyperparameters
n_episodes = 1000
log_step = 2000
buffer_size = int(1e5)
gamma = 0.99
tau = 1e-3
seed = 1
fill_buffer_at_start = 1000
batch_size = 5
max_time_steps = 300

# Non-Hyperparameters
episode_number = 0
step_number = 0
scores = []

# Environment
env = gym.make('Pendulum-v0')
d_action = env.action_space.shape[0]
d_obs = env.observation_space.shape[0]

random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)

# DDPG agent
agent = DDPGAgent(d_obs, d_action, buffer_size, batch_size, tau, gamma)

# train model
if True:
    # create experiment folder
    experiments_folder = os.path.join(os.getcwd(), 'results')
    experiment_name, experiment_folder = get_experiment_name(experiments_folder)

    for _ in range(n_episodes):
        episode_number += 1
        obs, done = env.reset(), False
        score, episode_length = 0, 0
        agent.noise.reset()

        while not done and episode_length <= max_time_steps:
            step_number += 1
            episode_length += 1

            action = agent.action(obs)
            next_obs, reward, done, _ = env.step(action)
            score += reward

            agent.add_to_buffer(obs, action, reward, next_obs, done)

            if step_number > fill_buffer_at_start:
                loss = agent.update()

                if step_number % log_step == 0:
                    print('ep {} st {} cr.ls {:.2f} ac.ls {:.2f} sc {:.2f}'.format(episode_number, step_number,
                                                                      loss[0], loss[1],
                                                                      np.mean(scores[-10:]) if len(scores) > 0 else 0.))
                    # save model
                    torch.save(agent.actor.state_dict(), os.path.join(experiment_folder, 'ckpt_actor.pth'))
                    torch.save(agent.critic.state_dict(), os.path.join(experiment_folder, 'ckpt_critic.pth'))

            obs = next_obs

            if done:
                break

            #env.render()

        scores.append(score)

# load model
else:
    agent.actor.load_state_dict(torch.load('results/drl.europe-west1-b.lithe-sunset-237906/results/1/ckpt_actor.pth'))
    agent.critic.load_state_dict(torch.load('results/drl.europe-west1-b.lithe-sunset-237906/results/1/ckpt_critic.pth'))

    obs = env.reset()
    for t in range(300):
        action = agent.action(obs, add_noise=False)
        env.render()
        obs, reward, done, _ = env.step(action)
        if done:
            break

env.close()

