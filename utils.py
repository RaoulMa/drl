import numpy as np
import os

def get_experiment_name(experiments_folder):
    """ create new experiment directory """
    if not os.path.exists(experiments_folder):
        os.makedirs(experiments_folder)
    dir_names = subdir_names(experiments_folder)
    c = 0
    for i, dir_name in enumerate(dir_names):
        if dir_name.isdigit() and int(dir_name) > c:
            c = int(dir_name)
    experiment_name = str(c + 1)
    experiment_folder = os.path.join(experiments_folder, experiment_name)
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    return experiment_name, experiment_folder

def subdir_paths(dpath):
    return [os.path.join(dpath, o) for o in os.listdir(dpath) if os.path.isdir(os.path.join(dpath, o))]

def subdir_names(dpath):
    return [o for o in os.listdir(dpath) if os.path.isdir(os.path.join(dpath, o))]

def data_to_txt(data, fpath):
    with open(fpath, 'w') as f:
        f.write(data)

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




