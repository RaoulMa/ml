import gym
import tensorflow as tf
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]

class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10, name='QNetwork'):
        with tf.variable_scope(name):
            self.inputs_ph = tf.placeholder(tf.float32, [None, state_size], name='inputs')

            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)

            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')

            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_ph, hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)

            self.output = tf.contrib.layers.fully_connected(self.fc2, action_size,
                                                            activation_fn=None)

            ### Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

train_episodes = 1000         # max number of episodes to learn from
max_steps = 200                # max steps in an episode
gamma = 0.99                   # future reward discount

# Exploration parameters
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Network parameters
hidden_size = 64               # number of units in each Q-network hidden layer
learning_rate = 0.0001         # Q-network learning rate

# Memory parameters
memory_size = 10000            # memory capacity
batch_size = 20                # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory

memory = Memory(max_size=memory_size)
mainQN = QNetwork(name='main', hidden_size=hidden_size, learning_rate=learning_rate)

env = gym.make('CartPole-v0')
env.reset()
state, reward, done, _ = env.step(env.action_space.sample())

# pre-training
for ii in range(pretrain_length):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

    if done:
        next_state = np.zeros(state.shape)
        memory.add((state, action, reward, next_state))
        env.reset()
        state, reward, done, _ = env.step(env.action_space.sample())
    else:
        memory.add((state, action, reward, next_state))
        state = next_state

# training
sess = tf.Session()
sess.run(tf.global_variables_initializer())
rewards_list = []

step = 0
for ep in range(1, train_episodes):
    total_reward = 0
    t = 0
    while t < max_steps:
        step += 1
        explore_p = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * step)
        if explore_p > np.random.rand():
            action = env.action_space.sample()
        else:
            feed = {mainQN.inputs_ph: state.reshape((1, *state.shape))}
            Qs = sess.run(mainQN.output, feed_dict=feed)
            action = np.argmax(Qs)

        next_state, reward, done, _ = env.step(action)

        total_reward += reward

        if done:
            next_state = np.zeros(state.shape)
            t = max_steps

            print('Episode: {}'.format(ep),
                  'Total reward: {}'.format(total_reward),
                  'Training loss: {:.4f}'.format(loss),
                  'Explore P: {:.4f}'.format(explore_p))
            rewards_list.append((ep, total_reward))

            memory.add((state, action, reward, next_state))

            env.reset()
            state, reward, done, _ = env.step(env.action_space.sample())

        else:
            memory.add((state, action, reward, next_state))
            state = next_state
            t += 1

        batch = memory.sample(batch_size)
        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        next_states = np.array([each[3] for each in batch])

        target_Qs = sess.run(mainQN.output, feed_dict={mainQN.inputs_ph: next_states})

        episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
        target_Qs[episode_ends] = (0, 0)

        targets = rewards + gamma * np.max(target_Qs, axis=1)

        loss, _ = sess.run([mainQN.loss, mainQN.opt],
                           feed_dict={mainQN.inputs_ph: states,
                                      mainQN.targetQs_: targets,
                                      mainQN.actions_: actions})

eps, rews = np.array(rewards_list).T
smoothed_rews = running_mean(rews, 10)
plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
plt.plot(eps, rews, color='grey', alpha=0.3)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

env.close()