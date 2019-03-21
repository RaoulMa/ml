import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import os

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

class dotdict(dict):
    __getattr__ = dict.__getitem__

class QNN(nn.Module):
    def __init__(self, input_dim, hidden_layer_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_dim = hidden_layer_dim

        self.layers = nn.ModuleList()
        layer_dim = [input_dim] + hidden_layer_dim + [output_dim]
        self.loss = nn.MSELoss()

        for i in range(len(layer_dim) - 1):
            self.layers.append(nn.Linear(layer_dim[i], layer_dim[i + 1], bias=False))
            if i < (len(layer_dim) - 2):
                self.layers.append(nn.Tanh())  # activation function

    def forward(self, x):
        model = torch.nn.Sequential(*self.layers)
        return model(x)

    def update(self, q, q_target, optimizer):
        loss = self.loss(q, q_target)
        self.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

cfg = dotdict({
    'seed': 16,
    'qnn_hidden_layer_dim': [200],
    'qnn_learning_rate': 0.001,
    'reward_discount_factor': 1.,
    'n_episodes': 100,
    'epsilon': 0.3
})

writer = SummaryWriter(experiment_folder)
env = gym.make('MountainCar-v0')
env.seed(cfg.seed)
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

action_dim = env.action_space.n
observation_dim = env.observation_space.shape[0]

epsilon = cfg.epsilon
gamma = cfg.reward_discount_factor
max_position = -0.4
n_successes = 0
ext_return_list, int_return_list, loss_list  = [], [], []

qnn = QNN(observation_dim, cfg.qnn_hidden_layer_dim, action_dim)
qnn_optimizer = optim.SGD(qnn.parameters(), lr=cfg.qnn_learning_rate)
scheduler = optim.lr_scheduler.StepLR(qnn_optimizer, step_size=1, gamma=0.9)

for episode in range(cfg.n_episodes):

    ext_return, int_return, loss = 0., 0., 0.
    obs, done = env.reset(), False

    while not done:

        # q-value in current state
        obs_t = Variable(torch.from_numpy(obs).type(torch.FloatTensor))
        q = qnn.forward(obs_t)

        if np.random.rand(1) < epsilon:
            action = np.random.randint(0, 3)
        else:
            _, action = torch.max(q, -1)
            action = action.item()

        obs_next, ext_reward, done, _ = env.step(action)

        # q-value for next state
        obs_next_t = Variable(torch.from_numpy(obs_next).type(torch.FloatTensor))
        q_next = qnn.forward(obs_next_t)
        max_q_next, _ = torch.max(q_next, -1)

        # to do: compute intrinsic reward from predictor network
        # for the moment just set to zero
        int_reward = 0

        # target network
        q_target = q.clone().detach()
        q_target[action] = int_reward + torch.mul(max_q_next.detach(), gamma)

        # update q-value network
        loss_t = qnn.update(q, q_target, qnn_optimizer)

        loss += loss_t.item()
        ext_return += ext_reward
        int_return += int_reward

        if obs_next[0] > max_position:
            max_position = obs_next[0]
            writer.add_scalar('data/max_position', max_position, episode)

        obs = obs_next

    if obs_next[0] >= 0.5:

        epsilon *= .99
        writer.add_scalar('data/epsilon', epsilon, episode)

        scheduler.step()
        writer.add_scalar('data/learning_rate', qnn_optimizer.param_groups[0]['lr'], episode)

        n_successes += 1
        writer.add_scalar('data/cumulative_success', n_successes, episode)
        writer.add_scalar('data/successes', 1, episode)

    elif obs_next[0] < 0.5:

        writer.add_scalar('data/success', 0, episode)

    loss_list.append(loss)
    ext_return_list.append(ext_return)
    int_return_list.append(int_return)

    writer.add_scalar('data/episode_loss', loss, episode)
    writer.add_scalar('data/int_episode_return', int_return, episode)
    writer.add_scalar('data/ext_episode_return', ext_return, episode)

    if episode % 10 == 0:
        print('episode {} loss {:.2f} ext.return {:.2f} int.return {:.2f} epsilon {:.2f} successes {} max.position {:.2f}'.format(
            episode, np.mean(loss_list[-10:]), np.mean(ext_return_list[-10:]), np.mean(int_return_list[-10:]), epsilon,
            n_successes, max_position))

writer.close()

# simulate one episode
obs, reward, done = env.reset(), 0., False
while not done:
    q = qnn.forward(Variable(torch.from_numpy(obs).type(torch.FloatTensor)))
    _, action = torch.max(q, -1)
    obs, _, done, _ = env.step(action.item())
    env.render()
env.close()