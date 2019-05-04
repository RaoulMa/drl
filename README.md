### About
Implementation of DDPG, PPO, DQN and VPG agents to solve various OpenAI gym environments 
using Tensorflow. The focus is on clean and short code.  

This repository serves the purpose of self-teaching. The implementations are not particularly
clear, efficient, well tested or numerically stable. We advise against using this software for non-didactic
purposes.

This software is licensed under the MIT License.

### Installation and Usage
This code is based on [TensorFlow](https://www.tensorflow.org/). Install Python 3 with basic 
packages, then run these commands: 
```Shell
git clone -b master --single-branch https://github.com/RaoulMa/drl.git
python3 -m pip install --user --upgrade pip
python3 -m pip install --user -r requirements.txt 
```
Run one of the python scripts, e.g.
```Shell
python3 ddpg_pendulum_tf.py
```

### References

[1] Proximal Policy Optimization Algorithms, Schulman et al. [arXiv:1707.06347v2]  [cs.LG]  28 Aug 2017

[2] Playing Atari with Deep Reinforcement Learning, Mnih et al. [arXiv:1312.5602v1]  [cs.LG]  19 Dec 2013

[3] Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning, Williams, 1992

[4] Continuous control with deep reinforcement learning, Lillicrap et. al. [arXiv:1509.02971], September 2015