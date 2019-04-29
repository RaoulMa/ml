import tensorflow as tf
import numpy as np
import gym
import sys

env = gym.make('CartPole-v0')
seed = 1

tf.set_random_seed(seed)

