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
        #self.true_dist = None
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
        #self.true_dist = true_dist
        return self.criterion(x, true_dist) #total loss of this batch (not normalized)


class CosineSIM(nn.Module):
    def __init__(self, margin=0.0):
        super(CosineSIM, self).__init__()
        self.criterion = nn.CosineEmbeddingLoss(margin=margin, size_average=None, reduce=None, reduction='sum')
        logging.info('built criterion (cosine)')
        
    def forward(self, s1, s2, target):
        #i use -target since target is: 1.0 (divergent) or -1.0 (parallel)
        #and i need: 1.0 (cosine of same vectors) or -1.0 (cosine of distant vectors)
        return self.criterion(s1, s2, -target) #total loss of this batch (not normalized)


class AlignSIM(nn.Module):
    def __init__(self):
        super(AlignSIM, self).__init__()
        logging.info('built criterion (align)')
        
    def forward(self, aggr, y, mask_t):
        #print('aggr',aggr[0])
        #print('y',y[0])
        sign = torch.ones(aggr.size(), device=y.device) * y.unsqueeze(-1) #[bs,lt] (by default ones builds on CPU)
        #aggr sign aggr*sign loss
        #-------------------------------
        # >0    -1 (par)  <0   ~0
        # <0    -1        >0   >0 (large)
        # >0    +1 (div)  >0   >0 (large)
        # <0    +1        <0   ~0
        ##read like: when aggr > 0 (target related to source) and -sign is -1 (parallel) the loss is very small 
        #i change the sign since i used -1 (uneven) +1 (parallel)
        error = torch.log(1.0 + torch.exp(aggr * sign)) #equation (3) error of each tgt word
        #print('error',error[0])
        sum_error = torch.sum(error * mask_t, dim=1) #error of each sentence in batch
        #print('sum_error',sum_error[0])
        batch_error = torch.sum(sum_error)
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


class ComputeLossSIM:
    def __init__(self, criterion, pooling, R, align_scale, opt=None):
        self.criterion = criterion
        self.pooling = pooling
        self.align_scale = align_scale
        self.opt = opt
        self.R = R


    def __call__(self, hs, ht, slen, tlen, y, mask_s, mask_t): 
        #hs [bs, sl, es] embeddings of source words after encoder (<cls> <bos> s1 s2 ... sI <eos> <pad> ...)
        #ht [bs, tl, es] embeddings of target words after encoder (<cls> <bos> t1 t2 ... tJ <eos> <pad> ...)
        #slen [bs] length of source sentences (I) in batch
        #tlen [bs] length of target sentences (J) in batch
        #y [bs] parallel(-1.0)/divergent(1.0) value of each sentence pair
        #mask_s [bs,sl]
        #mask_t [bs,tl]
        #mask_st [bs,sl,tl]
        mask_s = mask_s.unsqueeze(-1).type(torch.float64)
        mask_t = mask_t.unsqueeze(-1).type(torch.float64)

        if self.pooling == 'max':
            s, _ = torch.max(hs*mask_s + (1.0-mask_s)*-999.9, dim=1) #-999.9 should be -Inf but it produces an nan when multiplied by 0.0
            t, _ = torch.max(ht*mask_t + (1.0-mask_t)*-999.9, dim=1) 
            loss = self.criterion(s, t, y)

        elif self.pooling == 'mean':
            s = torch.sum(hs * mask_s, dim=1) / torch.sum(mask_s, dim=1)
            t = torch.sum(ht * mask_t, dim=1) / torch.sum(mask_t, dim=1)
            loss = self.criterion(s, t, y)

        elif self.pooling == 'cls':
            s = hs[:, 0, :] # take embedding of first token <cls>
            t = ht[:, 0, :] # take embedding of first token <cls>
            loss = self.criterion(s, t, y)

        elif self.pooling == 'align':
            S_st = torch.bmm(hs, torch.transpose(ht, 2, 1)) * self.align_scale #[bs, sl, es] x [bs, es, tl] = [bs, sl, tl]            
            if torch.isnan(S_st).any():
                logging.info('nan detected in alignment matrix (S_st) ...try reducing align_scale')
            #print('S_st',S_st[0])
            aggr_t = self.aggr(S_st,mask_s) #equation (2) #for each tgt word, consider the aggregated matching scores over the source sentence words
            loss = self.criterion(aggr_t,y,mask_t.squeeze())

        else:
            logging.error('bad pooling method {}'.format(self.pooling))
            sys.exit()

        return loss #not normalized


    def aggr(self,S_st,mask_s): #foreach tgt word finds the aggregation over all src words
#        print('mask_s',mask_s[0])
        exp_rS = torch.exp(S_st * self.R)
#        print('exp_rS',exp_rS[0])
        sum_exp_rS = torch.sum(exp_rS * mask_s,dim=1) #sum over all source words (source words nor used are masked)
#        print('sum_exp_rS',sum_exp_rS[0])
        log_sum_exp_rS_div_R = torch.log(sum_exp_rS) / self.R
#        print('log_sum_exp_rS_div_R',log_sum_exp_rS_div_R[0])
        return log_sum_exp_rS_div_R




