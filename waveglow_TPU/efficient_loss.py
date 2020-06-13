import torch
import torch.nn.functional as F

# PreEmpthasis & DeEmpthasis taken from https://github.com/AppleHolic/pytorch_sound/blob/master/pytorch_sound/models/sound.py#L64
class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size()) == 3, 'The number of dimensions of input tensor must be 3!'
        # reflect padding to match lengths of in/out
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter)


# "Efficient" Loss Function
class WaveGlowLoss(torch.nn.Module):
    def __init__(self, sigma=1., loss_empthasis=0.0, elementwise_mean=True):
        super().__init__()
        self.sigma2 = sigma ** 2
        self.sigma2_2 = self.sigma2 * 2
        self.mean = elementwise_mean
        self.loss_empthasis = loss_empthasis
        if self.loss_empthasis:
            self.empth = PreEmphasis(float(self.loss_empthasis))
    
    def db_loss(self, z, logdet): # DB loss I wanna try
        batch_size, segment_size = z.shape
        z_two = z.new_full(z.shape, 2.0)
        
        pos_mask = [z>=0]
        pos_log = z_two[pos_mask].pow(z[pos_mask].log10()) * z[pos_mask]
        neg_mask = [z<0]
        neg_log = -z_two[neg_mask].pow((-z[neg_mask]).log10()) * z[neg_mask]
        
        #print("pos_log.sum() =", pos_log.sum())
        #print("neg_log.sum() =", neg_log.sum())
        
        loss = ( (pos_log.sum() + neg_log.sum()) / self.sigma2 ) - logdet.sum()
        return loss/batch_size
    
    def forward(self, model_outputs):
        z, logdet = model_outputs # [B, ...], logdet
        #loss = 0.5 * z.pow(2).sum(1) / self.sigma2 - logdet
        
        z = z.float()
        logdet = logdet.float()
        
        if self.loss_empthasis:
            z = self.empth(z)
        
        #loss = self.db_loss(z, logdet) # new DB based version
         
        loss = z.pow(2).sum(1) / self.sigma2_2 - logdet # safe original
        
        loss = loss.mean()
        if self.mean:
            loss = loss / z.size(1)
        
        return loss
