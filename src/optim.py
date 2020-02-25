import torch
import logging
import sys
from torch import nn
from torch.autograd import Variable

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        logging.debug('built NoamOpt')
        
    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def state_dict(self):
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "factor": self.factor,
            "model_size": self.model_size,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)


##################################################################
### Criterions ###################################################
##################################################################

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size #vocab size
        #self.true_dist = None #???
        logging.info('built criterion (label smoothing)')
        
    def forward(self, x, target): 
        #x is [batch_size*max_len, vocab] 
        #target is [batch_size*max_len]
        assert x.size(1) == self.size
        true_dist = x.data.clone() #[batch_size*max_len, vocab]
        true_dist.fill_(self.smoothing / (self.size - 2)) #true_dist is filled with value=smoothing/(size-2)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) #moves value=confidence to tensor true_dist and indiceds target_data dim=1
        true_dist[:, self.padding_idx] = 0 #prob mass on padding_idx is 0
        mask = torch.nonzero(target.data == self.padding_idx) # device=x.device ???
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        #self.true_dist = true_dist #???
        return self.criterion(x, true_dist) #total loss of this batch (not normalized)


class Align(nn.Module):
    def __init__(self):
        super(Align, self).__init__()
        logging.info('built criterion (align)')
        
    def forward(self, S_st, y, mask_s, mask_t):
        #print('S_st',S_st.shape)
        #print('y',y.shape)
        #print('mask_s',mask_s.shape)
        #print('mask_t',mask_t.shape)
        #print(S_st[0])
        #print(y[0])
        #print(mask_s[0])
        #print(mask_t[0])

        ### considering S_st * y:
        # different sign (success)
        # same sign (mistake)
        error = torch.log(1.0 + torch.exp(S_st * y))
        mask_s = mask_s.unsqueeze(-1) #[bs, ls, 1]
        mask_t = mask_t.unsqueeze(-2) #[bs, 1, lt]
        batch_error = torch.sum(error * mask_s * mask_t) ### discard error of padded words
        return batch_error #total loss of this batch (not normalized)



##################################################################
### Compute losses ###############################################
##################################################################

class ComputeLossMLM:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, h, y): 
        x_hat = self.generator(h) # project x softmax #[bs,sl,V]
        x_hat = x_hat.contiguous().view(-1, x_hat.size(-1)) #[bs*sl,V]
        y = y.contiguous().view(-1) #[bs*sl]
        loss = self.criterion(x_hat, y) 
        return loss #not normalized


class ComputeLossALI:
    def __init__(self, criterion, step_ali, opt=None):
        self.criterion = criterion
        self.align_scale = step_ali['align_scale']
        self.opt = opt

    def __call__(self, h_st, y, mask_st): 
        #h_st [bs, ls+lt+2, es] embeddings of source and target words after encoder (<cls> s1 s2 ... sI <pad>* <sep> t1 t2 ... tJ <pad>*)
        #y [bs, ls, lt] alignment matrix (only words are considered neither <cls> nor <sep>)
        #mask_s [bs,ls]
        #mask_t [bs,lt]
        ls = y.shape[1]
        lt = y.shape[2]
        hs = h_st[:,1:ls+1,:]
        ht = h_st[:,ls+2:,:]
        mask_s = mask_st[:,1:ls+1].type(torch.float64)
        mask_t = mask_st[:,ls+2:,].type(torch.float64)
        S_st = torch.bmm(hs, torch.transpose(ht, 2, 1)) * self.align_scale #[bs, sl, es] x [bs, es, tl] = [bs, sl, tl]
        if torch.isnan(S_st).any():
            logging.info('nan detected in alignment matrix (S_st) ...try reducing align_scale')
        loss = self.criterion(S_st,y,mask_s,mask_t)
        return loss #not normalized

