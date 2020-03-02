import torch
import logging
import sys
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

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
        logging.debug('built criterion (label smoothing)')
        
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
        error_batch = self.criterion(x, true_dist)
        npred = (target != self.padding_idx).sum()
        ypred = torch.argmax(x,dim=1)
        nok = ((ypred == target) & (target != self.padding_idx)).sum()
        return error_batch, nok, npred #total loss of this batch (not normalized)


class Align(nn.Module):
    def __init__(self):
        super(Align, self).__init__()
        logging.debug('built criterion (align)')
        
    def forward(self, DP_st, y, mask_s, mask_t):
        assert DP_st.shape == y.shape
        assert len(DP_st.shape) == 3
        assert len(mask_s.shape) == 2 #[bs, ls]
        assert len(mask_t.shape) == 2 #[bs, lt]

        ### considering DP_st * y:
        # different sign (success)
        # same sign (error)
        mask = ((DP_st>0.0) | (y<0.0)) * mask_s.unsqueeze(-1) * mask_t.unsqueeze(-2) ### predicted(DP_st>0.0) or aligned(y<0.0) and not masked
        error = torch.log(1.0 + torch.exp(DP_st*y)) #[bs,ls,lt]
        batch_error = torch.sum(error*mask) # sum over errors of this batch (all word pairs not masked)
        nok = (DP_st*y*mask < 0.0).sum()
        npred = (mask == 1.0).sum()
        return batch_error, nok, npred #total loss of this batch (not normalized)

class Cosine(nn.Module):
    def __init__(self,margin=0.0):
        super(Cosine, self).__init__()
        logging.debug('built criterion (cosine)')

    def forward(self, DP, y):
        assert DP.shape == y.shape #DP [bs], y [bs]
        assert len(DP.shape) == 1  
        error = torch.log(1.0 + torch.exp(DP*y))
        batch_error = torch.sum(error) 
        npred = y.numel()
        nok = (DP*y < 0.0).sum()
        return batch_error, nok, npred #total loss of this batch (not normalized)

##################################################################
### Compute losses ###############################################
##################################################################

class ComputeLossMLM:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        logging.info('built ComputeLossMLM')

    def __call__(self, h, y): 
        x_hat = self.generator(h) # project x softmax #[bs,sl,V]
        x_hat = x_hat.contiguous().view(-1, x_hat.size(-1)) #[bs*sl,V]
        y = y.contiguous().view(-1) #[bs*sl]
        loss, nok, npred = self.criterion(x_hat, y) 
        return loss, nok, npred #sum of loss over batch


class ComputeLossALI:
    def __init__(self, criterion, step_ali, opt=None):
        self.criterion = criterion
        self.scale = step_ali['scale']
        self.norm = step_ali['norm']
        self.opt = opt
        logging.info('built ComputeLossALI scale={} norm={}'.format(self.scale, self.norm))

    def __call__(self, h_st, y, ls, st_mask): 
        #ls is max leng of source words
        assert len(h_st.shape) == 3 #[bs, ls+lt+2, es] embeddings of source and target words (<cls> s1 s2 ... sI <pad>* <sep> t1 t2 ... tJ <pad>*)
        assert len(y.shape) == 3 #[bs, ls, lt] alignment matrix (only words are considered neither <cls> nor <sep>)
        assert len(st_mask.shape) == 2 #st_mask [bs,ls+lt+2]
        assert st_mask.shape[0:2] == h_st.shape[0:2]
        s, t, hs, ht, s_mask, t_mask = sentence_embedding(h_st, st_mask, ls, norm_h=self.norm) 
        #s [bs, es]
        #t [bs, es]
        #hs [bs, ls, es]
        #ht [bs, lt, es]
        #s_mask [bs, ls]
        #t_mask [bs, lt]
        DP_st = torch.bmm(hs, torch.transpose(ht, 2, 1)) * self.scale #[bs, sl, es] x [bs, es, tl] = [bs, sl, tl] (cosine similarity after normalization)
        if torch.isnan(DP_st).any():
            logging.info('nan detected in alignment matrix (DP_st)')
        loss, nok, npred = self.criterion(DP_st,y,s_mask,t_mask)
        return loss, nok, npred #sum of word-pair loss over batch

class ComputeLossCOS:
    def __init__(self, criterion, step_cos, opt=None):
        self.criterion = criterion
        self.pooling = step_cos['pooling']
        self.norm = step_cos['norm']
        self.opt = opt
        logging.info('built ComputeLossCOS pooling={} norm={}'.format(self.pooling, self.norm))

    def __call__(self, h_st, y, ls, st_mask): 
        #ls is max leng of source words
        assert len(y.shape) == 1 #[bs] uneven 1.0 if uneven, -1.0 if parallel
        assert len(h_st.shape) == 3 #[bs, ls+lt+2, es] embeddings of source and target words (<cls> s1 s2 ... sI <pad>* <sep> t1 t2 ... tJ <pad>*)
        assert len(st_mask) == 2 #[bs,ls+lt+2]
        assert st_mask.shape[0:2] == h_st.shape[0:2]
        s, t, hs, ht, s_mask, t_mask = sentence_embedding(h_st, st_mask, ls, pooling=self.pooling, norm_st=self.norm)
        #s [bs, es]
        #t [bs, es]
        #hs [bs, ls, es]
        #ht [bs, lt, es]
        #s_mask [bs, ls]
        #t_mask [bs, lt]
        DP = torch.bmm(s.unsqueeze(-2), t.unsqueeze(-1)).squeeze(2).squeeze(1) #[bs, 1, es] X [bs, es, 1] = [bs, 1, 1] => [bs]
        if torch.isnan(DP).any():
            logging.info('nan detected in unevent vector (DP)')
        loss, nok, npred = self.criterion(DP, y) 
        return loss, nok, npred #sum of sent loss over batch


def sentence_embedding(h_st, st_mask, ls, pooling='mean', norm_st=False, norm_h=False):
    assert len(h_st.shape) == 3 #[bs, ls+lt+2, es] embeddings of source and target words (<cls> s1 s2 ... sI <pad>* <sep> t1 t2 ... tJ <pad>*)
    assert len(st_mask.shape) == 2 #[bs, ls+lt+2]
    assert st_mask.shape[0:2] == h_st.shape[0:2]
    hs = h_st[:,1:ls+1,:] #[bs, ls, es]
    ht = h_st[:,ls+2:,:] #[bs, lt, es]
    s_mask = st_mask[:,1:ls+1].type(torch.float64).unsqueeze(-1) #[bs, ls, 1]
    t_mask = st_mask[:,ls+2:,].type(torch.float64).unsqueeze(-1) #[bs, lt, 1]
    if pooling == 'cls':
        s = h_st[:, 0, :] # take embedding of <cls>
        t = h_st[:,ls+1,:] # take embedding of <sep>
    else:
        if pooling == 'max':
            s, _ = torch.max(hs*s_mask + (1.0-s_mask)*-999.9, dim=1) #-999.9 should be -Inf but it produces an nan when multiplied by 0.0
            t, _ = torch.max(ht*t_mask + (1.0-t_mask)*-999.9, dim=1) #-999.9 should be -Inf but it produces an nan when multiplied by 0.0
        elif pooling == 'mean':
            s = torch.sum(hs*s_mask, dim=1) / torch.sum(s_mask, dim=1)
            t = torch.sum(ht*t_mask, dim=1) / torch.sum(t_mask, dim=1)
        else:
            logging.error('bad pooling method: {}'.format(self.pooling))
    s_mask = s_mask.squeeze(-1) #[bs, es]
    t_mask = t_mask.squeeze(-1) #[bs, es]
    if norm_h:
        hs = F.normalize(hs,p=2,dim=2,eps=1e-12) #all embeddings are normalized
        ht = F.normalize(ht,p=2,dim=2,eps=1e-12) #all embeddings are normalized
    if norm_st:
        s = F.normalize(s,p=2,dim=1,eps=1e-12) #[bs, es]
        t = F.normalize(t,p=2,dim=1,eps=1e-12) #[bs, es]
    return s, t, hs, ht, s_mask, t_mask


