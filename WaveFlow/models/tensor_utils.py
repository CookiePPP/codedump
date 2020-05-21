import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def permute_height(x, reverse=False, bipart=False, shift=False, inverse_shift=False):
    x = [x[:,:,i] for i in range(x.shape[2])]
    if bipart and reverse:
        half = len(x)//2
        x = x[:half][::-1] + x[half:][::-1] # reverse H halfs [0,1,2,3,4,5,6,7] -> [3,2,1,0] + [7,6,5,4] -> [3,2,1,0,7,6,5,4]
    elif reverse:
        x = x[::-1] # reverse entire H [0,1,2,3,4,5,6,7] -> [7,6,5,4,3,2,1,0]
    if shift:
        x = [x[-1],] + x[:-1] # shift last H into first position [0,1,2,3,4,5,6,7] -> [7,0,1,2,3,4,5,6]
    elif inverse_shift:
        x = x[1:] + [x[0],]   # shift first H into last position [0,1,2,3,4,5,6,7] -> [1,2,3,4,5,6,7,0]
    return torch.stack(x, dim=2)