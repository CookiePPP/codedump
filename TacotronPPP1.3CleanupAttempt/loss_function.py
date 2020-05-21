import torch
from torch import nn
from utils import get_mask_from_lengths


class Tacotron2Loss(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2Loss, self).__init__()
        self.pos_weight = torch.tensor(hparams.gate_positive_weight)
        self.loss_func = hparams.loss_func
        self.alignment_loss_scaler = hparams.alignment_loss_weight
        self.alignment_encoderwise_mean = hparams.alignment_encoderwise_mean
        self.reduction = 'sum' if hparams.mean_without_padding else 'mean'
    
    def forward(self, model_output, targets):
        mel_target, gate_target, output_lengths, alignment_target, *_ = targets
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)
        
        mel_out, mel_out_postnet, gate_out, alignment_out = model_output
        gate_out = gate_out.view(-1, 1)
        
        # get mel loss
        if self.loss_func == 'MSELoss':
            mel_loss = nn.MSELoss(reduction=self.reduction)(mel_out, mel_target) + \
                nn.MSELoss(reduction=self.reduction)(mel_out_postnet, mel_target)
        elif self.loss_func == 'SmoothL1Loss':
            mel_loss = nn.SmoothL1Loss(reduction=self.reduction)(mel_out, mel_target) + \
                nn.SmoothL1Loss(reduction=self.reduction)(mel_out_postnet, mel_target)
        if self.reduction == 'sum':
            total_elems = output_lengths.sum() * mel_out.shape[1] # TODO: check this is n_mel_channel dim
            mel_loss = mel_loss / total_elems # get mean without paddings
        
        # get gate_loss
        gate_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(gate_out, gate_target)

        loss = mel_loss + gate_loss
        
        # (optional) get alignment loss from PAG target
        if alignment_target is not None:
            if self.alignment_encoderwise_mean: # this bit needs some cleanup
                alignment_loss = nn.MSELoss()(alignment_out.transpose(1,2), alignment_target)
            else:
                alignment_loss = nn.MSELoss(reduction='sum')(alignment_out.transpose(1,2), alignment_target)
                alignment_loss = alignment_loss / (alignment_target.shape[0]*alignment_target.shape[2])
                alignment_loss = alignment_loss * self.alignment_loss_scaler
            loss += alignment_loss
        
        return loss, gate_loss, alignment_loss
