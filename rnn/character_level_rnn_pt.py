import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

with open('anna.txt', 'r') as f:
    text = f.read()

chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: i for i,ch in int2char.items()}
encoded = np.array([char2int[ch] for ch in text])



