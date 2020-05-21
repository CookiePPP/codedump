import torch
import numpy as np
from math import log, pi


class WaveFlowLoss(torch.nn.Module):
    def __init__(self, hparams, elementwise_mean=True):
        super(WaveFlowLoss, self).__init__()
        self.sigma = hparams.sigma
        self.sigma2 = self.sigma ** 2
        self.sigma2_2 = self.sigma2 * 2
        self.mean = elementwise_mean
    
    def forward(self, model_outputs):
        raise NotImplementedError # This doesn't train properly atm. Need to fix it (at *some* point).
        
        z, logdet = model_outputs # [B, ...], logdet
        
        loss = z.pow(2).sum(1) / self.sigma2_2 - logdet
        loss = loss.mean()
        if self.mean:
            loss = loss / z.size(1)
        k = 0.5 * np.log(2 * np.pi) + np.log(self.sigma)

        return loss + k