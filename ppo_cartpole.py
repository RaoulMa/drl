""" PPO Agent
    Proximal Policy Optimization Algorithms, Schulman et al. 2017
    arXiv:1707.06347v2  [cs.LG]  28 Aug 2017

    In PPO we keep the old and current policies. We sample trajectories
    from the old policy and update the current policy wrt to the clipped
    surrogate objective. If the ratio for one (s_t, a_t) pair is outside
    the allowed region, the objective gets clipped, which means that the
    corresponding gradient is zero. On-policy algorithm.
"""
import numpy as np
import tensorflow as tf
import gym

class PPOAgent():
    def __init__(self, d_input, d_hidden_layers, d_output, learning_rate, reward_discount_factor, clip_range,
                 experiment_folder):

        self.d_output = d_output
        self.learning_rate = learning_rate
        self.clip_range = clip_range
        self.reward_discount_factor = reward_discount_factor

        # dictionary to store one batch of sampled trajectories
        # as well as corresponding policies, advantages etc...
        self.sample = {}

        self.baseline = None
        if 'advantage' in baseline_cfg.baseline:
            if 'shared' in baseline_cfg.baseline:
                # baseline sharing layers with policy except for the last layer
                self.baseline = StateValueFunction(self.layers_[-2].get_shape()[1],
                                                   baseline_cfg.d_hidden_layers,
                                                   1,
                                                   baseline_cfg.learning_rate,
                                                   reward_discount_factor,
                                                   baseline_cfg.gae_lamda,
                                                   self.layers_[:-1],
                                                   'dense',
                                                   activation,
                                                   experiment_folder, 'baseline')
            else:
                # baseline not sharing any layers with policy
                self.baseline = StateValueFunction(d_input,
                                                   baseline_cfg.d_hidden_layers,
                                                   1,
                                                   baseline_cfg.learning_rate,
                                                   reward_discount_factor,
                                                   baseline_cfg.gae_lamda,
                                                   [],
                                                   'dense',
                                                   activation,
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
        feed = {self.observations: obs}
        policy = self.session.run(self.policy, feed)[0] + 1e-6
        policy /= np.sum(policy)
        action = np.argmax(np.random.multinomial(1, policy))
        return action, policy

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
    """ State-Value Function trained via Monte-Carlo """
    def __init__(self, d_input, d_hidden_layers, d_output, learning_rate,
                 reward_discount_factor, gae_lamda,
                 prev_layers=[], nn_type='dense', activation='relu',
                 experiment_folder=os.getcwd(), name='state_value_function'):
        super().__init__(d_input, d_hidden_layers, d_output, learning_rate, prev_layers, nn_type,
                         activation, experiment_folder, name)
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

env = gym.make('CartPole-v0')
d_observation = env.observation_space.shape[0]
n_actions = env.action_space.n
gamma = 0.99
seed = 1

np.random.seed(seed)
tf.set_random_seed(seed)
env.seed(seed)

episode_number, batch_number = 0, 0

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for _ in range(600):
    batch_number += 1
    obs_batch, reward_batch, action_batch, creward_batch = [], [], [], []

    for _ in range(4):
        ep_length = 0
        episode_number += 1

        obs_batch.append([])
        reward_batch.append([])
        action_batch.append([])
        creward_batch.append([])

        obs = env.reset()
        done = False
        while not done:
            ep_length += 1
            obs_batch[-1].append(obs)

            feed = {observations_ph: np.reshape(obs, (1,-1))}
            policy_ = np.squeeze(sess.run(policy, feed))
            action = np.argmax(np.random.multinomial(1, policy_))
            obs, r, done, info = env.step(action)

            action_batch[-1].append(action)
            reward_batch[-1].append(r)

        gamma_list = gamma ** np.arange(0, ep_length)
        creward_batch.append(gamma_list * np.cumsum(reward_batch[-1][::-1])[::-1])

    obs_batch = np.vstack(obs_batch)
    action_batch = np.hstack(action_batch)
    creward_batch = np.hstack(creward_batch)

    #creward_batch = (creward_batch - np.mean(creward_batch)) / np.std(creward_batch)

    feed = {observations_ph: obs_batch, actions_ph: action_batch, crewards_ph: creward_batch}

    loss_, _ = sess.run([loss,train], feed)

    if batch_number % 50 == 0:
        areturn = np.mean([np.sum(rewards) for rewards in reward_batch])
        print('bn {} ep {} loss {} return {}'.format(batch_number, episode_number, loss_, areturn))

for _ in range(2):
    obs, done = env.reset(), False
    while not done:
        env.render()
        feed = {observations_ph: np.reshape(obs, (1, -1))}
        policy_ = np.squeeze(sess.run(policy, feed))
        action = np.argmax(np.random.multinomial(1, policy_))
        obs, _, done, _ = env.step(action)

sess.close()
env.close()




