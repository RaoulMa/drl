""" Agent trained via DDPG.
Continuous control with deep reinforcement learning,
Lillicrap et. al. [arXiv:1509.02971], September 2015
"""
import os
import sys
import numpy as np
import random
import gym
import copy
import tensorflow as tf
from collections import deque
from utils import get_experiment_name

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def sample(self):
        samples = random.sample(self.memory, k=self.batch_size)
        return zip(*samples)

class OUNoise:
    """Ornstein-Uhlenbeck process.
       Differential equation: dx_t =  theta (mu - x_t) dt + sigma dW_t
       mu: drift
       W_t: Wiener process
    """
    def __init__(self, size, mu=0., theta=0.15, sigma=0.1):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.delta_t = 0.01
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.delta_t + self.sigma * np.sqrt(self.delta_t) * np.array([random.gauss(0,1) for i in range(self.size)])
        self.state = x + dx
        return self.state

class Actor():
    def __init__(self, scope, d_obs, d_action):
        self.scope = scope
        with tf.variable_scope(scope):
            self.obs_ph = tf.placeholder(shape=(None, d_obs), dtype=tf.float32)
            output = tf.layers.dense(self.obs_ph, units=400, activation=tf.nn.relu)
            output = tf.layers.dense(output, units=300, activation=tf.nn.relu)
            self.output = tf.layers.dense(output, units=d_action, activation=tf.nn.tanh)
            self.trainable_variables = tf.trainable_variables(scope=scope)

class Critic():
    def __init__(self, scope, d_obs, d_action):
        self.scope = scope
        with tf.variable_scope(scope):
            self.obs_ph = tf.placeholder(shape=(None, d_obs), dtype=tf.float32)
            self.action_ph = tf.placeholder(shape=(None, d_action), dtype=tf.float32)
            output = tf.layers.dense(self.obs_ph, units=400, activation=tf.nn.relu)
            output = tf.concat([output, self.action_ph], axis=1)
            output = tf.layers.dense(output, units=300, activation=tf.nn.relu)
            self.output = tf.layers.dense(output, units=1, activation=None)
            self.action_grads = tf.gradients(self.output, self.action_ph)
            self.trainable_variables = tf.trainable_variables(scope=scope)

class DDPGAgent():
    def __init__(self, d_obs, d_action, buffer_size, batch_size, tau, gamma, experiment_folder):
        self.d_obs = d_obs
        self.d_action = d_action
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.global_step = 0
        self.experiment_folder = experiment_folder
        self.nn_name = 'ddpg'
        self.checkpoint_prefix = os.path.join(self.experiment_folder, self.nn_name + '_ckpt')

        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.noise = OUNoise(d_action)

        # Create policy networks (current and target)
        self.actor = Actor('actor', d_obs, d_action)
        self.actor_target = Actor('actor_target', d_obs, d_action)

        # Create Q-value networks (current and target)
        self.critic = Critic('critic', d_obs, d_action)
        self.critic_target = Critic('critic_target', d_obs, d_action)

        self.rewards_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
        self.done_ph = tf.placeholder(shape=(None,), dtype=tf.float32)

        # --- Update Q-value network (critic). --- #

        # Q-value for s_t, a_t
        self.Q_value = self.critic.output

        # target Q-value for s_(t+1), a_(t+1) predicted from actor_target
        self.Q_target = self.critic_target.output

        self.target = self.rewards_ph + self.gamma * (1 - self.done_ph) * tf.stop_gradient(tf.reduce_max(self.Q_target, axis=1))

        self.critic_loss = tf.losses.mean_squared_error(labels=tf.reshape(self.target,(-1,1)), predictions=self.Q_value)
        self.critic_train = tf.train.AdamOptimizer(1e-3).minimize(self.critic_loss,
                                                                  var_list=self.critic.trainable_variables)

        # --- Update Policy network (actor) --- #

        # loss with s_t and a_t predicted by actor
        self.actor_loss = - tf.reduce_mean(self.critic.output)

        # gradients of actor
        self.actor_grads = tf.gradients(ys=self.actor.output, xs=self.actor.trainable_variables,
                                        grad_ys=self.critic.action_grads)

        # divide gradients by batch_size and multiply by -1 for gradient ascent
        self.actor_grads = list(map(lambda x: tf.divide(x, - self.batch_size), self.actor_grads))

        self.actor_train = tf.train.AdamOptimizer(1e-4).apply_gradients(zip(self.actor_grads,
                                                                            self.actor.trainable_variables))

        # --- Update target networks --- #

        self.update_critic_target = tf.group(*[tf.assign(t_var, s_var) for t_var, s_var
                                               in zip(self.critic_target.trainable_variables,
                                                      self.critic.trainable_variables)])
        self.update_actor_target = tf.group(*[tf.assign(t_var, s_var) for t_var, s_var
                                               in zip(self.actor_target.trainable_variables,
                                                      self.actor.trainable_variables)])
        # saver
        self.saver = tf.train.Saver()

        self.init_session()

    def init_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def save(self):
        self.saver.save(self.sess, self.checkpoint_prefix, global_step=self.global_step)
        return self.saver

    def load(self):
        fnames = os.listdir(self.experiment_folder)
        ckpt_names = [fname.split('.', 1)[0] for fname in fnames if self.nn_name in fname and '.data' in fname]
        global_steps = [int(name.split('-',-1)[1]) for name in ckpt_names]
        latest_ckpt_name = ckpt_names[np.argmax(global_steps)]
        latest_ckpt = os.path.join(self.experiment_folder, latest_ckpt_name)
        self.saver.restore(self.sess, latest_ckpt)

    def add_to_buffer(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def get_trainable_variables(self, nn_name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=nn_name)

    def update_target_op(self, src_name, dst_name):
        main_vars = self.get_trainable_variables(src_name)
        target_vars = self.get_trainable_variables(dst_name)
        assign_ops = [tf.assign(target_var, main_var) for target_var, main_var in zip(target_vars, main_vars)]
        return tf.group(*assign_ops)

    def action(self, obs, add_noise=True):
        feed = {self.actor.obs_ph: obs.reshape(1,-1)}
        action = self.sess.run(self.actor.output, feed)
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def update(self):
        obs, actions, rewards, next_obs, dones = self.memory.sample()

        obs = np.vstack([np.squeeze(i) for i in obs])
        actions = np.vstack([np.squeeze(i) for i in actions])
        rewards = np.hstack(rewards)
        next_obs = np.vstack([np.squeeze(i) for i in next_obs])
        dones = np.hstack(dones)

        # Make a forward pass to compute the
        # - actions from actor for current observations s_t
        # - actions predicted from actor_target for next observations s_(t+1)
        feed = {self.actor.obs_ph: obs, self.actor_target.obs_ph: next_obs}
        pred_actions, pred_next_actions = self.sess.run([self.actor.output, self.actor_target.output], feed)

        # Update Q-value network (critic).
        feed = {self.critic_target.obs_ph: next_obs,
                self.critic_target.action_ph: pred_next_actions,
                self.critic.obs_ph: obs,
                self.critic.action_ph: actions,
                self.rewards_ph: rewards,
                self.done_ph: dones}
        critic_loss, _ = self.sess.run([self.critic_loss, self.critic_train], feed)

        # Update Policy network (actor)
        feed = {self.critic.obs_ph: obs, self.critic.action_ph: pred_actions, self.actor.obs_ph: obs}
        actor_loss, _ = self.sess.run([self.actor_loss, self.actor_train], feed)

        # Update target networks
        self.sess.run(self.update_critic_target)
        self.sess.run(self.update_actor_target)

        self.global_step+=1

        return critic_loss, actor_loss

# Hyperparameters
n_episodes = 1000
log_step = 2000
buffer_size = int(1e5)
gamma = 0.99
tau = 1e-3
seed = 1
fill_buffer_at_start = 1000
batch_size = 128
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
tf.set_random_seed(seed)
env.seed(seed)

# train model
if True:
    # create experiment folder
    experiments_folder = os.path.join(os.getcwd(), 'results')
    experiment_name, experiment_folder = get_experiment_name(experiments_folder)

    agent = DDPGAgent(d_obs, d_action, buffer_size, batch_size, tau, gamma, experiment_folder)

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
                    agent.save()

            obs = next_obs

            if done:
                break

            #env.render()

        scores.append(score)

# load model
else:
    experiment_folder = 'results/63'
    agent = DDPGAgent(d_obs, d_action, buffer_size, batch_size, tau, gamma, experiment_folder)
    agent.load()

    obs = env.reset()
    for t in range(300):
        action = agent.action(obs, add_noise=False)
        env.render()
        obs, reward, done, _ = env.step(action)
        if done:
            break

env.close()