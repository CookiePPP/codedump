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
        z, logdet = model_outputs # [B, ...], logdet
        B, T = z.shape
        
        # TODO: which log_p? the commented line is the one from the paper but the below is the correct Gaussian pdf..
        # log_p_sum += ((-0.5) * (log(2.0 * pi) + 2.0 * z.pow(2)).sum())
        log_p_sum =    ((-0.5) * (log(2.0 * pi) + z.pow(2)).sum())
        
        logdet = logdet / (B * T)
        log_p = log_p_sum / (B * T)
        
        log_p, logdet = torch.mean(log_p), torch.mean(logdet)
        
        loss = -(log_p + logdet)
        return loss