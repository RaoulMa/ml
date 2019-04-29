import tensorflow as tf
import numpy as np
import gym

class ReplayBuffer:

    def __init__(self, d_obs, size):
        self.obs1_buf = np.zeros([size, d_obs], dtype=np.float32)
        self.obs2_buf = np.zeros([size, d_obs], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.int32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

if __name__ == '__main__':

    # settings
    env_name = 'CartPole-v0'

    seed = 1
    n_hidden_units = 32

    n_train_steps = 20000
    n_steps_before_training = 100
    batch_size = 32
    replay_buffer_size = 1000
    target_update_freq = 10
    test_run_freq = 1000

    learning_rate = 1e-3
    epsilon = 1
    epsilon_final = 0.0
    epsilon_decay_rate = 1/10000
    gamma = 0.99

    env, env_test = gym.make(env_name), gym.make(env_name)
    d_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    np.random.seed(seed)
    tf.set_random_seed(seed)
    env.seed(seed)

    replay_buffer = ReplayBuffer(d_observations, replay_buffer_size)

    # Q-value network
    with tf.variable_scope('main'):
        observations = tf.placeholder(shape=(None, d_observations), dtype=tf.float32)
        outputs = observations
        outputs = tf.layers.dense(outputs, units=n_hidden_units, activation=tf.tanh)
        qa_values = tf.layers.dense(outputs, units=n_actions, activation=None)

    # target Q-value network
    with tf.variable_scope('target'):
        observations_target = tf.placeholder(shape=(None, d_observations), dtype=tf.float32)
        outputs_target = observations_target
        outputs_target = tf.layers.dense(outputs_target, units=n_hidden_units, activation=tf.tanh)
        qa_values_target = tf.layers.dense(outputs_target, units=n_actions, activation=None)

    # loss
    actions = tf.placeholder(shape=(None,), dtype=tf.int32)
    rewards = tf.placeholder(shape=(None,), dtype=tf.float32)
    done_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    actions_enc = tf.one_hot(actions, n_actions)
    q_values = tf.reduce_sum(actions_enc * qa_values, axis=1)
    target = rewards + gamma * (1 - done_ph) * tf.stop_gradient(tf.reduce_max(qa_values_target, axis=1))
    loss = tf.reduce_mean((q_values - target) ** 2)

    # update op for target network
    main_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main')
    target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
    assign_ops = [tf.assign(target_var, main_var) for target_var, main_var in zip(target_vars, main_vars)]
    target_update_op = tf.group(*assign_ops)

    # make train op
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def get_action(observation, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(n_actions)
        else:
            qa_value = sess.run(qa_values, feed_dict={observations: observation.reshape(1,-1)})
            return np.argmax(qa_value)

    def test_runs(n_test_eps=10):
        ep_returns, ep_lengths = [], []
        for _ in range(n_test_eps):
            ep_return, ep_length = 0, 0
            observation, reward, done = env_test.reset(), 0, False
            while not done:
                env_test.render()
                observation, reward, done, _ = env_test.step(get_action(observation, epsilon_final))
                ep_return += reward
                ep_length += 1
            ep_returns.append(ep_return)
            ep_lengths.append(ep_length)
        return np.mean(ep_returns), np.mean(ep_lengths)

    observation, reward, done, ep_return, ep_length = env.reset(), 0, False, 0, 0
    batch_qa_values, batch_losses, ep_returns, ep_lengths = [], [], [], []

    for t in range(n_train_steps + n_steps_before_training):

        action = get_action(observation, epsilon)
        next_observation, reward, done, _ = env.step(action)

        replay_buffer.store(observation, action, reward, next_observation, done)
        observation = next_observation

        ep_return += reward
        ep_length += 1

        if done:
            ep_returns.append(ep_return)
            ep_lengths.append(ep_length)
            observation, reward, done, ep_return, ep_length = env.reset(), 0, False, 0, 0

        if t > n_steps_before_training:
            batch = replay_buffer.sample_batch(batch_size)
            feed = {observations: batch['obs1'], observations_target: batch['obs2'],
                    actions: batch['acts'], rewards: batch['rews'], done_ph: batch['done']}

            # check tensors
            if False:
                observations_ = sess.run(observations, feed)
                observations_target_ = sess.run(observations_target, feed)
                actions_ = sess.run(actions, feed)
                rewards_ = sess.run(rewards, feed)
                done_ = sess.run(done_ph, feed)

            loss_, q_values_, _ = sess.run([loss, q_values, train_op], feed)
            batch_losses.append(loss_)
            batch_qa_values.append(q_values_)

        if t % target_update_freq == 0:
            sess.run(target_update_op)

        epsilon = 1 + (epsilon_final - 1) * min(1, t * epsilon_decay_rate)

        if t > 0 and t % test_run_freq == 0 :
            test_return, test_length = test_runs()
            print(('step: %d \t loss: %.3f \t q-value: %.2f \t return: %.2f' \
                   + '\t length: %.f \t test_return: %.2f \t test_length: %.2f ' \
                   + '\t epsilon: %.3f') %
                  (t, np.mean(batch_losses[-10:]), np.mean(batch_qa_values[-10:]),
                   np.mean(ep_returns[-10:]), np.mean(ep_lengths[-10:]),
                   test_return, test_length , epsilon))

    sess.close()
    env.close()
    env_test.close()



