import numpy as np
import tensorflow as tf
import gym

if __name__ == '__main__':

    class dotdict(dict):
        __getattr__ = dict.__getitem__

    cfg = dotdict({
        'seed': 16,
        'policy_hidden_layer_dim': [32],
        'policy_learning_rate': 0.01,
        'batch_size': 16
    })

    env_name = 'CartPole-v0'
    env = gym.make(env_name)

    np.random.seed(cfg.seed)
    tf.set_random_seed(cfg.seed)
    env.seed(cfg.seed)

    d_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # parameters
    batch_size = 16
    n_hidden_units = 32
    learning_rate = 1e-2
    n_batches = 20
    gamma = 0.99

    # policy network
    observations = tf.placeholder(shape=(None, d_observations), dtype=tf.float32)
    outputs = observations
    outputs = tf.layers.dense(outputs, units=n_hidden_units, activation=tf.tanh)
    outputs = tf.layers.dense(outputs, units=n_actions, activation=None)
    policy = tf.nn.softmax(outputs)
    log_policy = tf.nn.log_softmax(outputs)

    # log probabilities for taken actions
    actions = tf.placeholder(shape=(None,), dtype=tf.int32)
    actions_enc = tf.one_hot(actions, n_actions)
    log_proba = tf.reduce_sum(actions_enc * log_policy, axis=1)

    # loss & train operation
    returns = tf.placeholder(shape=(None,), dtype=tf.float32)
    loss = - tf.reduce_mean(log_proba * returns, axis=0)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    n_steps = 0

    def train_one_batch():

        batch_observations, batch_rewards, batch_actions, batch_returns = [], [], [], []

        for i in range(0, batch_size):

            observation, reward, done = env.reset(), 0.0, False

            batch_observations.append([observation])
            batch_rewards.append([reward])
            batch_actions.append([])

            while not done:

                policy_ = sess.run(policy, {observations: observation.reshape(1,-1)})[0]

                action = np.argmax(np.random.multinomial(1, policy_))

                observation, reward, done, _ = env.step(action)

                batch_actions[-1].append(action)
                batch_rewards[-1].append(reward)
                if not done:
                    batch_observations[-1].append(observation)

            if True:
                # undiscounted returns
                returns_ = np.flip(np.cumsum(np.array(batch_rewards[-1][1:])), axis=0)
            else:
                # discounted returns
                rewards_ = batch_rewards[-1]
                gamma_list = (gamma ** np.arange(len(rewards_)))
                dreturns =  [sum(rewards_[i+1:] * gamma_list[:len(rewards_)-i-1]) for i in range(len(rewards_)-1)]
                returns_ = dreturns

            batch_returns.append(returns_)

        feed_observations = np.array([item for sublist in batch_observations for item in sublist])
        feed_returns = np.array([item for sublist in batch_returns for item in sublist])
        feed_actions = np.array([item for sublist in batch_actions for item in sublist])

        # standardise returns
        feed_returns = (feed_returns - np.mean(feed_returns)) / (np.std(feed_returns) + 1e-8)

        feed = {observations: feed_observations, actions: feed_actions, returns: feed_returns}

        # check tensors
        if False:
            observations_ = sess.run(observations, feed)
            actions_ = sess.run(actions, feed)
            returns_ = sess.run(returns, feed)
            outputs_ = sess.run(outputs, feed)
            policy_ = sess.run(policy, feed)
            actions_enc_ = sess.run(actions_enc, feed)
            log_proba_ = sess.run(log_proba, feed)
            loss_ = sess.run(loss, feed)

        loss_, _ = sess.run([loss, train_op], feed)

        return loss_, batch_observations, batch_actions, batch_returns

    for i in range(n_batches):
        stats = train_one_batch()
        loss_ = stats[0]
        ereturn = np.mean(np.array([ret[0] for ret in stats[3]]))
        n_steps += sum([len(sublist) for sublist in stats[2]])

        print('Batch {} Step {} Loss {:.2f} eReturn {:.2f}'.format(i, n_steps, loss_, ereturn))

    observation = env.reset()
    counter = 0

    for _ in range(1000):
        env.render()
        policy_ = sess.run(policy, {observations: observation.reshape(1, -1)})[0]
        action = np.argmax(np.random.multinomial(1, policy_))
        observation, _, done, _ = env.step(action)
        counter += 1
        if done:
            env.reset()
            print('reset after {} steps'.format(counter))
            counter = 0

    env.close()

    sess.close()












