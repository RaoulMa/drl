import numpy as np
import tensorflow as tf
import gym

env = gym.make('CartPole-v0')
d_observation = env.observation_space.shape[0]
n_actions = env.action_space.n
gamma = 0.99
seed = 1

np.random.seed(seed)
tf.set_random_seed(seed)
env.seed(seed)

observations_ph = tf.placeholder(shape=(None, d_observation), dtype=tf.float32)

outputs = tf.layers.dense(observations_ph, units=32, activation=tf.nn.relu)
outputs = tf.layers.dense(outputs, units=n_actions)

policy = tf.nn.softmax(outputs, axis=1)
log_policy = tf.nn.log_softmax(outputs, axis=1)

actions_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
actions_enc = tf.one_hot(actions_ph, axis=1, depth=n_actions)
crewards_ph = tf.placeholder(shape=(None,), dtype=tf.float32)

loss = - tf.reduce_mean(tf.reduce_sum(log_policy * actions_enc, axis=1) * crewards_ph, axis=0)

train = tf.train.AdamOptimizer(1e-3).minimize(loss)

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









