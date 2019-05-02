import tensorflow as tf
import numpy as np
import gym
import collections
import random

env = gym.make('CartPole-v0')
seed = 1

tf.set_random_seed(seed)
np.random.seed(seed)
env.seed(seed)

n_actions= env.action_space.n
d_observation = env.observation_space.shape[0]

n_steps = 20000
log_step = 1000
batch_size = 32
fill_replay_buffer = 1000
replay_buffer = collections.deque(maxlen=10000)
target_update_freq = 100
gamma = 0.99
epsilon_start = 1.
epsilon_final = 0.01
epsilon_decay_range = 10000
step_number = 0

epsilon = epsilon_start

with tf.variable_scope('main'):
    observations_ph = tf.placeholder(shape=(None, d_observation), dtype=tf.float32)
    q_value = tf.layers.dense(observations_ph, units=32, activation=tf.nn.relu)
    q_value = tf.layers.dense(q_value, units=n_actions, activation=None)

with tf.variable_scope('target'):
    observations_target_ph = tf.placeholder(shape=(None, d_observation), dtype=tf.float32)
    q_value_target = tf.layers.dense(observations_target_ph, units=32, activation=tf.nn.relu)
    q_value_target = tf.layers.dense(q_value_target, units=n_actions, activation=None)

actions_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
actions_enc = tf.one_hot(actions_ph, axis=1, depth=n_actions)
rewards_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
done_ph = tf.placeholder(shape=(None,), dtype=tf.float32)

q_value_for_actions = tf.reduce_sum(q_value * actions_enc, axis=1)
target = rewards_ph + gamma * (1 - done_ph) * tf.stop_gradient(tf.reduce_max(q_value_target, axis=1))
loss = tf.reduce_mean((q_value_for_actions - target)**2)

train = tf.train.AdamOptimizer(1e-3).minimize(loss)

main_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main')
target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
assign_ops = [tf.assign(target_var, main_var) for target_var, main_var in zip(target_vars, main_vars)]
update_target_network = tf.group(*assign_ops)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def get_action(obs, eps=0.):
    if np.random.rand() > eps:
        feed = {observations_ph: obs.reshape(1,-1)}
        q_value_ = sess.run(q_value, feed)
        return np.argmax(q_value_, axis=1)[0]
    else:
        return np.random.randint(n_actions)

def sample_from_replay_buffer(size):
    history = random.sample(list(replay_buffer), min(size, len(replay_buffer)))
    return zip(*history)

sess.run(update_target_network)

done=True
for _ in range(fill_replay_buffer):
    if done:
        obs, reward, done = env.reset(), 0, False
    action = get_action(obs, 1)
    next_obs, reward, done, _ = env.step(action)
    replay_buffer.append([obs, action, reward, done, next_obs])

returns = [0]
episode_number = 0
for _ in range(n_steps):
    step_number += 1
    if done:
        episode_number += 1
        obs, reward, done = env.reset(), 0, False
        returns.append(reward)

    action = get_action(obs, epsilon)
    next_obs, reward, done, _ = env.step(action)
    returns[-1] += reward

    replay_buffer.append([obs, action, reward, done, next_obs])

    obs_batch, action_batch, reward_batch, done_batch, next_obs_batch = sample_from_replay_buffer(batch_size)

    obs_batch = np.vstack(obs_batch)
    action_batch = np.hstack(action_batch)
    reward_batch = np.hstack(reward_batch)
    done_batch = np.hstack(done_batch)
    next_obs_batch = np.vstack(next_obs_batch)

    feed = {observations_ph: obs_batch, actions_ph: action_batch, rewards_ph: reward_batch,
            done_ph: done_batch, observations_target_ph: next_obs_batch}

    loss_, _ = sess.run([loss, train], feed)

    obs = next_obs

    if step_number <= epsilon_decay_range:
        epsilon -= (epsilon_start - epsilon_final)/epsilon_decay_range

    if step_number % target_update_freq == 0:
        sess.run(update_target_network)

    if step_number % log_step == 0:
        print('ep {} step {} loss {:.2f} return {} eps {:.2f}'.format(episode_number, step_number,
                                                                      loss_, np.mean(returns[-10:]),
                                                                      epsilon))

sess.close()
env.close()














