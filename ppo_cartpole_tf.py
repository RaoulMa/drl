""" PPO Agent
    Proximal Policy Optimization Algorithms, Schulman et al. 2017
    arXiv:1707.06347v2  [cs.LG]  28 Aug 2017

    In PPO we keep the old and current policies. We sample trajectories
    from the old policy and update the current policy wrt to the clipped
    surrogate objective. If the ratio for one (s_t, a_t) pair is outside
    the allowed region, the objective gets clipped, which means that the
    corresponding gradient is zero. On-policy algorithm.
"""
import os
import numpy as np
import tensorflow as tf
import gym
import sys
import copy
from utils import get_experiment_name

class FNN():
    """Feed-forward neural network."""
    def __init__(self, d_input, d_hidden_layers, d_output, learning_rate,
                 experiment_folder=os.getcwd(), name='fnn'):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.learning_rate = learning_rate
        self.nn_name = name
        self.global_step = 0
        self.experiment_folder = experiment_folder
        self.checkpoint_prefix = os.path.join(self.experiment_folder, self.nn_name + '_ckpt')

        with tf.variable_scope(self.nn_name):

            kernel_initializer = tf.initializers.orthogonal()

            self.d_hidden_layers = d_hidden_layers
            self.d_layers = self.d_hidden_layers + [self.d_output]
            self.n_layers = len(self.d_layers)

            self.layers_ = []
            shape = (None, *self.d_input) if type(self.d_input).__name__ == 'tuple' else (None, self.d_input)
            self.observations = tf.placeholder(shape=shape, dtype=tf.float32)
            outputs = self.observations
            self.layers_.append(outputs)

            for i in range(self.n_layers):
                if i < (self.n_layers - 1):
                    outputs = tf.layers.dense(outputs,
                                              units=self.d_layers[i],
                                              activation=tf.nn.relu,
                                              kernel_initializer=kernel_initializer)
                else:
                    outputs = tf.layers.dense(outputs,
                                              units=self.d_layers[i],
                                              activation=None,
                                              kernel_initializer=kernel_initializer)
                self.layers_.append(outputs)
                self.outputs = outputs

            self.trainable_variables = tf.trainable_variables(scope=self.nn_name)

    def save(self):
        self.saver.save(self.session, self.checkpoint_prefix, global_step=self.global_step)
        return self.saver

    def load(self):
        fnames = os.listdir(self.experiment_folder)
        ckpt_names = [fname.split('.', 1)[0] for fname in fnames if self.nn_name in fname and '.data' in fname]
        global_steps = [int(name.split('-',-1)[1]) for name in ckpt_names]
        latest_ckpt_name = ckpt_names[np.argmax(global_steps)]
        latest_ckpt = os.path.join(self.experiment_folder, latest_ckpt_name)
        self.saver.restore(self.session, latest_ckpt)

class PPOAgent(FNN):
    """ PPO Agent
        Proximal Policy Optimization Algorithms, Schulman et al. 2017
        arXiv:1707.06347v2  [cs.LG]  28 Aug 2017

        In PPO we keep the old and current policies. We sample trajectories
        from the old policy and update the current policy wrt to the clipped
        surrogate objective. If the ratio for one (s_t, a_t) pair is outside
        the allowed region, the objective gets clipped, which means that the
        corresponding gradient is zero. On-policy algorithm.
    """
    def __init__(self, d_input, d_hidden_layers, d_output, learning_rate,
                 reward_discount_factor, clip_range, experiment_folder=os.getcwd(), name='ppo'):
        super().__init__(d_input, d_hidden_layers, d_output, learning_rate, experiment_folder, name)

        self.d_output = d_output
        self.learning_rate = learning_rate
        self.clip_range = clip_range
        self.reward_discount_factor = reward_discount_factor
        self.gae_lamda = 0.97
        self.experiment_folder = experiment_folder

        # dictionary to store one batch of sampled trajectories
        # as well as corresponding policies, advantages etc...
        self.sample = {}

        # baseline not sharing any layers with policy
        self.baseline = StateValueFunction(d_input,
                                           d_hidden_layers,
                                           1,
                                           learning_rate,
                                           reward_discount_factor,
                                           self.gae_lamda,
                                           experiment_folder, 'baseline')
        # logits, policy, log_policy
        self.logits = self.layers_[-1]
        self.policy = tf.nn.softmax(self.logits)
        self.log_policy = tf.nn.log_softmax(self.logits)

        # sampled actions
        self.sampled_actions = tf.placeholder(shape=(None, self.d_output), dtype=tf.float32)

        # log probabilities of sampled actions
        # log p(a_t|s_t, theta) for given (s_t, a_t) pairs
        self.log_policy_of_actions = tf.reduce_sum(self.log_policy * self.sampled_actions, axis=1)

        # sampled log probabilities of actions
        self.sampled_log_policy_of_actions = tf.placeholder(shape=(None,), dtype=tf.float32)

        # sampled advantages
        self.sampled_advantages = tf.placeholder(shape=(None,), dtype=tf.float32)

        # ratio
        # r_t(theta) = p(a_t|s_t, theta) / p(a_t|s_t, theta_old)
        self.ratio = tf.exp(self.log_policy_of_actions - self.sampled_log_policy_of_actions)

        # clipped policy objective that should be maximised
        self.clipped_ratio = tf.clip_by_value(self.ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        self.policy_loss = - tf.reduce_mean(tf.minimum(self.ratio * self.sampled_advantages,
                                                       self.clipped_ratio * self.sampled_advantages), axis=0)

        # entropy loss (exploration bonus)
        self.entropy_loss = tf.reduce_mean(tf.reduce_sum(self.policy * self.log_policy, axis=1), axis=0)

        # complete loss
        self.loss = self.policy_loss  # + 0.01 * entropy_loss
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                               var_list = self.get_trainable_variables(self.nn_name))

        # saver
        self.saver = tf.train.Saver()

        # initialise graph
        self.init_session()

    def init_session(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        # set baseline session
        if self.baseline is not None:
            self.baseline.init_session(self.saver, self.session)

    def close(self):
        self.session.close()

    def get_trainable_variables(self, nn_name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=nn_name)

    def action(self, obs):
        """ return action and policy """
        feed = {self.observations: obs.reshape(1,-1)}
        policy = self.session.run(self.policy, feed)[0] + 1e-6
        policy /= np.sum(policy)
        action = np.argmax(np.random.multinomial(1, policy))
        return action

    def update(self, obs_batch, action_batch, reward_batch, done_batch):
        """ one gradient step update """

        # cumulative rewards
        creward_batch = cumulative_rewards(reward_batch, self.reward_discount_factor)

        # flatten out episodes in batch
        obs = np.squeeze(np.vstack(obs_batch).astype(np.float32))
        actions = np.vstack(action_batch).astype(np.float32)
        crewards = np.hstack(creward_batch).astype(np.float32)

        # use advantage baseline
        advantages = crewards
        if self.baseline is not None:
            state_value_batch = []
            for i in range(len(obs_batch)):
                state_values = np.reshape(self.baseline.forward(np.vstack(obs_batch[i]).astype(np.float32)),[-1])
                state_value_batch.append(state_values)
            advantage_batch = advantage_values(obs_batch, reward_batch,
                                               done_batch, state_value_batch,
                                               self.baseline.reward_discount_factor,
                                               self.baseline.gae_lamda)
            advantages = np.hstack(advantage_batch).astype(np.float32)

        feed = {self.observations: obs,
                self.sampled_actions: actions,
                self.sampled_advantages: advantages}
        policy, log_policy, log_policy_of_actions = self.session.run([self.policy,
                                                                      self.log_policy,
                                                                      self.log_policy_of_actions], feed)
        # deep copy of sampled trajectories from old policy
        old_sample = copy.deepcopy(self.sample)

        # store current sample
        self.sample = {}
        self.sample['obs'] = obs
        self.sample['actions'] = actions
        self.sample['advantages'] = advantages
        self.sample['log_policy_of_actions'] = log_policy_of_actions

        # at start the old policy is equals the current policy
        if len(old_sample) == 0:
            old_sample = copy.deepcopy(self.sample)

        # train policy with old sample of trajectories
        feed = {self.observations: old_sample['obs'],
                self.sampled_actions: old_sample['actions'],
                self.sampled_advantages: old_sample['advantages'],
                self.sampled_log_policy_of_actions: old_sample['log_policy_of_actions']}
        loss, ratio, clipped_ratio, _, policy_loss, entropy_loss = self.session.run([self.loss,
                                                                                     self.ratio,
                                                                                     self.clipped_ratio,
                                                                                     self.train_op,
                                                                                     self.policy_loss,
                                                                                     self.entropy_loss], feed)
        self.global_step += 1

        # statistics
        stats = {}
        stats['crewards'] = crewards
        stats['obs'] = obs
        stats['baseline_value'] = np.mean(advantages)
        stats['ratio'] = np.mean(ratio)
        stats['clipped_ratio'] = np.mean(clipped_ratio)
        stats['n_clips'] = sum(diff > 10e-6 for diff in ratio - clipped_ratio)
        stats['policy_loss'] = policy_loss
        stats['entropy_loss'] = entropy_loss

        # monte carlo update of baseline
        if self.baseline is not None:
            baseline_loss, _ = self.baseline.mc_update(obs, crewards)
            stats['baseline_loss'] = baseline_loss

        return loss, _ , stats

class StateValueFunction(FNN):
    """ State-Value Function (critic) trained via Monte-Carlo """
    def __init__(self, d_input, d_hidden_layers, d_output, learning_rate,
                 reward_discount_factor, gae_lamda, experiment_folder=os.getcwd(), name='baseline'):
        super().__init__(d_input, d_hidden_layers, d_output, learning_rate, experiment_folder, name)
        self.reward_discount_factor = reward_discount_factor
        self.gae_lamda = gae_lamda
        self.nn_name = name

        self.outputs = self.layers_[-1]

        # loss
        self.crewards = tf.placeholder(shape=(None, 1), dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(labels=self.crewards, predictions=self.outputs)

        # only train baseline layer
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                               var_list=self.get_trainable_variables(self.nn_name))

    def forward(self, obs):
        feed = {self.observations: obs}
        outputs = self.session.run(self.outputs, feed)
        return outputs

    def init_session(self, saver=None, session=None):
        if session == None:
            self.saver = tf.train.Saver()
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
        else:
            self.session = session
            self.saver = saver

    def close(self):
        self.session.close()

    def get_trainable_variables(self, nn_name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=nn_name)

    def mc_update(self, obs, crewards):
        feed = {self.observations: obs, self.crewards: np.reshape(crewards,[-1,1])}
        loss, _ = self.session.run([self.loss, self.train_op], feed)
        self.global_step += 1
        return loss, _

def one_hot(value, n_classes):
    """ one-hot encoding """
    enc = np.zeros(n_classes, 'uint8')
    enc[value] = 1
    return enc

def cumulative_rewards(reward_batch, gamma):
    """ discounted cumulative sum of rewards for t = 1,2,...,T-1 """
    crewards = []
    for i in range(len(reward_batch)):
        gamma_mask = gamma ** np.arange(len(reward_batch[i]))
        cr = np.flip(np.cumsum(np.flip(gamma_mask * reward_batch[i]), axis=0)) / (gamma_mask + 1e-8)
        crewards.append(cr.tolist())
    return crewards

def advantage_values(obs_batch, reward_batch, done_batch, state_value_batch, gamma, lamda):
    """ generalised advantage estimate of the advantage function
    if lamda = 1: A_t = R_t - V(s_t)
    if lamda = 0: A_t = r_t+1 + gamma * V(s_t+1) - V(s_t)
    max. episode length: T
    obs_batch: s_1, s_2 ..., s_T-1 (i.e. without s_T)
    reward_batch: r_2, r_3, ..., r_T (r_2 is the reward obtained after taking action a_1 in s_1)
    done_batch: d_2, d_3, ..., d_T
    """
    n_episodes = len(obs_batch)
    ep_lengths = [len(obs_batch[i]) for i in range(len(obs_batch))]
    max_ep_length = max(ep_lengths)

    # obtain equal-sized arrays
    obs_arr = np.zeros((n_episodes, max_ep_length, len(obs_batch[0][0])))
    reward_arr = np.zeros((n_episodes, max_ep_length))
    state_value_arr = np.zeros((n_episodes, max_ep_length))
    advantage_arr = np.zeros((n_episodes, max_ep_length))
    done_arr = np.ones((n_episodes, max_ep_length))  # padding with ones

    for i in range(n_episodes):
        obs_arr[i, :ep_lengths[i]] = obs_batch[i]
        reward_arr[i, :ep_lengths[i]] = reward_batch[i]
        done_arr[i, :ep_lengths[i]] = done_batch[i]
        state_value_arr[i, :ep_lengths[i]] = state_value_batch[i]

    advantage_value = 0.  # A_T = 0
    next_state_value = 0.  # set V(s_T) = 0 since done = True

    for t in reversed(range(max_ep_length)):
        # only keep V(s_t+1) if done = False
        mask = 1.0 - done_arr[:, t]
        next_state_value = next_state_value * mask

        # td(0) error: delta_t = r_(t+1) + gamma * V(s_t+1) - V(s_t)
        delta = reward_arr[:, t] + gamma * next_state_value - state_value_arr[:, t]

        # advantage: A_t = delta_t + gamma * lamda * A_t+1
        advantage_value = delta + gamma * lamda * advantage_value
        advantage_arr[:, t] = advantage_value

        # V(s_t)
        next_state_value = state_value_arr[:, t]

    advantage_batch = [advantage_arr[i, :ep_lengths[i]] for i in range(n_episodes)]

    return advantage_batch

env = gym.make('CartPole-v0')
d_obs = env.observation_space.shape[0]
n_actions = env.action_space.n
gamma = 0.99
seed = 1
learning_rate = 1e-2
clip_range = 0.1
n_batches = 100
log_step = 10
batch_size = 8

np.random.seed(seed)
tf.set_random_seed(seed)
env.seed(seed)

# PPO agent
agent = PPOAgent(d_obs, [32], n_actions, learning_rate, gamma, clip_range, '', 'ppo')

episode_number, batch_number = 0, 0
for _ in range(n_batches):
    batch_number += 1
    obs_batch, reward_batch, action_batch, creward_batch, done_batch = [], [], [], [], []

    for _ in range(batch_size):
        ep_length = 0
        episode_number += 1

        obs_batch.append([])
        reward_batch.append([])
        action_batch.append([])
        creward_batch.append([])
        done_batch.append([])

        obs = env.reset()
        done = False
        while not done:
            ep_length += 1
            obs_batch[-1].append(obs)

            action = agent.action(obs)
            obs, r, done, info = env.step(action)
            action_one_hot = one_hot(action, n_actions)

            done_batch[-1].append(done)
            action_batch[-1].append(action_one_hot)
            reward_batch[-1].append(r)

        gamma_list = gamma ** np.arange(0, ep_length)

    loss, _, _ = agent.update(obs_batch, action_batch, reward_batch, done_batch)

    if batch_number % log_step  == 0:
        areturn = np.mean([np.sum(rewards) for rewards in reward_batch])
        print('bn {} ep {} loss {} return {}'.format(batch_number, episode_number, loss, areturn))

for _ in range(1):
    obs, done = env.reset(), False
    while not done:
        env.render()
        action = agent.action(obs)
        obs, _, done, _ = env.step(action)

agent.close()
env.close()




