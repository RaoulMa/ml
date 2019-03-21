import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import tensorflow.contrib as tfc
import tensorflow.contrib.eager as tfe
import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from gym.spaces import Discrete
import collections
import seaborn as sns
import pandas as pd

tfe.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)

def get_experiment_name(experiments_folder):
    if not os.path.exists(experiments_folder):
        os.makedirs(experiments_folder)
    dir_names = [o for o in os.listdir(experiments_folder) if os.path.isdir(os.path.join(experiments_folder, o))]
    c = 0
    for i, dir_name in enumerate(dir_names):
        if dir_name.isdigit() and int(dir_name) > c:
            c = int(dir_name)
    experiment_name = str(c + 1)
    experiment_folder = os.path.join(experiments_folder, experiment_name)
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    return experiment_name, experiment_folder

data_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'data')
experiments_folder = os.path.join(os.getcwd(), 'results')
experiment_name, experiment_folder = get_experiment_name(experiments_folder)
log_dir = experiment_folder

class dotdict(dict):
    __getattr__ = dict.__getitem__

class PolicyNN(tf.keras.Model):
    def __init__(self, hidden_layer_dim=[32], output_dim=2):
        super(PolicyNN, self).__init__()

        self.hidden_layer_dim = hidden_layer_dim
        self.output_dim = output_dim
        self.hidden_layers = []

        for i in range(len(self.hidden_layer_dim)):
            h = tf.layers.Dense(self.hidden_layer_dim[i], activation=tf.nn.relu)
            self.hidden_layers.append(h)

        self.output_layer = tf.layers.Dense(output_dim, activation=None)

    def flat_weights(self):
        variables = self.trainable_variables
        flat_variables = tf.concat([tf.reshape(v, [-1]) for v in variables], axis=0)
        return flat_variables[tf.newaxis]

    def forward(self, observation):
        h = observation
        for i in range(len(self.hidden_layer_dim)):
            h = self.hidden_layers[i](h)
        predict_logit = h[:, 0]
        action_logits = self.output_layer(h)
        action_distr = tfd.Categorical(logits=action_logits)
        action = tf.squeeze(action_distr.sample())
        return action, predict_logit

    def loss(self, hypothesis1, hypothesis2):
        logit1 = hypothesis1.forward(self.flat_weights())
        distr1 = tfd.Bernoulli(logits=logit1)
        logit2 = hypothesis2.forward(self.flat_weights())
        distr2 = tfd.Bernoulli(logits=logit2)
        loss = - tfd.kl_divergence(distr1, distr2)
        return loss

    def grads(self, hypothesis1, hypothesis2):
        with tfe.GradientTape() as tape:
            loss = self.loss(hypothesis1, hypothesis2)
        return tape.gradient(loss, self.variables)


class HypothesisNN(tf.keras.Model):
    def __init__(self, hidden_layer_dim=[32, 32], output_dim=1):
        super(HypothesisNN, self).__init__()

        self.hidden_layer_dim = hidden_layer_dim
        self.output_dim = output_dim
        self.hidden_layers = []

        for i in range(len(self.hidden_layer_dim)):
            h = tf.layers.Dense(self.hidden_layer_dim[i], activation=tf.nn.relu)
            self.hidden_layers.append(h)

        self.output_layer = tf.layers.Dense(self.output_dim, activation=None)

    def forward(self, policy_weights):
        h = policy_weights
        for i in range(len(self.hidden_layer_dim)):
            h = self.hidden_layers[i](h)
        logit = self.output_layer(h)
        return logit

    def loss(self, policy_weights, predict_logit):
        predicted_predict_logit = self.forward(policy_weights)
        distr1 = tfd.Bernoulli(logits=predicted_predict_logit)
        distr2 = tfd.Bernoulli(logits=predict_logit)
        loss = tf.reduce_mean(tfd.kl_divergence(distr1, distr2))
        return loss

    def grads(self, policy_weights, predict_logit):
        with tfe.GradientTape() as tape:
            loss = self.loss(policy_weights, predict_logit)
        return tape.gradient(loss, self.variables)


def smooth(arr, window):
    return np.convolve(arr, np.ones(window) / window, 'same')


def plot_return(values, xlabel, ylabel):
    plt.figure()
    sns.set(style='darkgrid')

    sns.tsplot(data=values, value=ylabel,
               time=pd.Series(np.arange(1, values.shape[1] + 1), name=xlabel),
               estimator=np.mean)

    fname = '{0}_{1}.png'.format(ylabel.replace(' ', '_'), xlabel)
    plt.savefig(os.path.join(experiment_folder, fname), dpi=300)


cfg = dotdict({
    'seed': 16,
    'policy_hidden_layer_dim': [32],
    'evaluator_hidden_layer_dim': [32, 32],
    'policy_learning_rate': 0.01,
    'hypothesis_learning_rate': 0.01
})

tf.reset_default_graph()

env_name = 'MountainCar-v0'
env = gym.make(env_name)

tf.set_random_seed(cfg.seed)
np.random.seed(cfg.seed)
env.seed(cfg.seed)

action_dim = env.action_space.n
obs_dim = env.observation_space.shape[0]

policy = PolicyNN(cfg.policy_hidden_layer_dim, action_dim)
hypothesis1 = HypothesisNN(cfg.evaluator_hidden_layer_dim)
hypothesis2 = HypothesisNN(cfg.evaluator_hidden_layer_dim)

policy_optimizer = tf.train.AdamOptimizer(learning_rate=cfg.policy_learning_rate)
hypothesis1_optimizer = tf.train.AdamOptimizer(learning_rate=cfg.hypothesis_learning_rate)
hypothesis2_optimizer = tf.train.AdamOptimizer(learning_rate=cfg.hypothesis_learning_rate)

history = collections.deque(maxlen=1000)
returns_list = []
max_pos_list = []


def simulate(save=True):
    done = False
    sum_rewards = 0.0
    obs = env.reset()
    max_pos = obs[0]
    while not done:
        obs_tensor = tf.constant(obs)[tf.newaxis]  # (1,2)
        action, predict_logit = policy.forward(obs_tensor)
        obs, r, done, _ = env.step(action.numpy())
        max_pos = max(max_pos, obs[0])
        sum_rewards += r
    max_pos_list.append(max_pos)

    if save:
        history.append((policy.flat_weights(), [predict_logit]))
        returns_list.append(sum_rewards)

    return sum_rewards, predict_logit


simulate(False)

for i in range(1000):

    # update policy weights to maximimze hypothesis uncertainty
    grads = policy.grads(hypothesis1, hypothesis2)
    policy_optimizer.apply_gradients(zip(grads, policy.variables))

    # sample trajectory
    _, predict_logit = simulate()

    history_batch = random.sample(history, min(128, len(history))) + list(history)[-128:]
    policy_weights_batch, predict_logit_batch = zip(*history_batch)

    policy_weights_batch = tf.stack(policy_weights_batch)
    predict_logit_batch = tf.stack(predict_logit_batch)

    # update hypothesis weights to minimize hypothesis uncertainty
    grads = hypothesis1.grads(policy_weights_batch, predict_logit_batch)
    hypothesis1_optimizer.apply_gradients(zip(grads, hypothesis1.variables))

    grads = hypothesis2.grads(policy_weights_batch, predict_logit_batch)
    hypothesis2_optimizer.apply_gradients(zip(grads, hypothesis2.variables))

    if i % 100 == 0:
        # sample trajectory
        _, pred_logit = simulate(False)
        hypo1_pred = hypothesis1.forward(policy.flat_weights()).numpy()
        hypo2_pred = hypothesis2.forward(policy.flat_weights()).numpy()

        print('{} pos {} predict_logit {} hypo1 {} hypo2 {}'.format(i, np.mean(max_pos_list[-10:]),
                                                                    pred_logit, hypo1_pred, hypo2_pred))

        plot_return(np.array([max_pos_list]), 'iterations_{}'.format(i), 'maximal position')

    if False:
        history_batch = random.sample(history, min(128, len(history))) + list(history)[-128:]
        policy_weights_batch, returns_batch = zip(*history_batch)

        policy_weights_batch = tf.stack(policy_weights_batch)
        returns_batch = tf.stack(returns_batch)

        grads = evaluator.grads(policy_weights_batch, returns_batch)
        evaluator_optimizer.apply_gradients(zip(grads, evaluator.variables))

        grads = policy.grads(evaluator)
        policy_optimizer.apply_gradients(zip(grads, policy.variables))

        if i % 100 == 0:

            avg_predicted_return, avg_observed_return, avg_evaluator_loss = 0., 0., 0.

            for n in range(10):
                predicted_return = - policy.loss(evaluator)[0][0]
                observed_return = tf.constant(simulate(False))
                policy_weights = policy.trainable_flat_variables()[tf.newaxis]
                evaluator_loss = evaluator.loss(policy_weights, tf.reshape(observed_return, (1, 1)))

                avg_predicted_return = 1 / (n + 1) * (n * avg_predicted_return + predicted_return.numpy())
                avg_observed_return = 1 / (n + 1) * (n * avg_observed_return + observed_return.numpy())
                avg_evaluator_loss = 1 / (n + 1) * (n * avg_evaluator_loss + evaluator_loss.numpy())

            print('{}: pred._return {:.2f} obs._return {:.2f} evaluator_loss {:.2f}'.format(
                i, avg_predicted_return, avg_observed_return, avg_evaluator_loss))

            if i % 100 == 0 and i > 0:
                plot_return(np.array([returns_list]), 'iterations_{}'.format(i), 'returns')

