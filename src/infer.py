import io
import gzip
import numpy as np
import torch
import logging
import time
import random
import sys
import glob
import os
import pyonmttok
from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from src.dataset import Vocab, DataSet, OpenNMTTokenizer, batch
from src.trainer import sequence_mask
from src.model import make_model
from src.optim import NoamOpt, LabelSmoothing, CosineSIM, AlignSIM, ComputeLossMLM, ComputeLossSIM


class Infer():

    def __init__(self, opts):
        self.dir = opts.dir
        self.vocab = Vocab(opts.cfg['vocab'])
        self.cuda = opts.cfg['cuda']
        V = len(self.vocab)
        N = opts.cfg['num_layers']
        d_model = opts.cfg['hidden_size']
        d_ff = opts.cfg['feedforward_size']
        h = opts.cfg['num_heads']
        dropout = opts.cfg['dropout']
        self.align_scale = 0.001
        self.token = OpenNMTTokenizer(**opts.cfg['token'])
        self.pooling = opts.pooling
        self.normalize = True

        self.model = make_model(V, N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)
        if self.cuda:
            self.model.cuda()

        ### load checkpoint
        files = sorted(glob.glob(self.dir + '/checkpoint.???????.pth')) 
        if len(files):
            file = files[-1] ### last is the newest
            checkpoint = torch.load(file)
            self.model.load_state_dict(checkpoint['model'])
            logging.info('loaded checkpoint {}'.format(file))
        else:
            logging.error('no checkpoint available')
            sys.exit()


    def __call__(self, file):
        logging.info('Start testing')
        files = file.split(',')

        fsrc = files[0]
        if fsrc.endswith('.gz'): fs = gzip.open(fsrc, 'rb')
        else: fs = io.open(fsrc, 'r', encoding='utf-8', newline='\n', errors='ignore')

        if len(files)>1:
            ftgt = files[1]
            if ftgt.endswith('.gz'): ft = gzip.open(ftgt, 'rb')
            else: ft = io.open(ftgt, 'r', encoding='utf-8', newline='\n', errors='ignore')

        self.data = []
        self.model.eval()
        with torch.no_grad():
            for lsrc in fs:

                src = [s for s in self.token.tokenize(lsrc)]
                idx_src = [self.vocab[s] for s in src]
                idx_src.insert(0,self.vocab.idx_bos)
                idx_src.append(self.vocab.idx_eos)
                idx_src.insert(0,self.vocab.idx_cls)
                batch_src = np.array([idx_src])
                batch_src_len = np.array([len(idx_src)])
                x1 = torch.from_numpy(batch_src) #[batch_size, max_len] the original words with padding
                x1_mask = torch.as_tensor((x1 != self.vocab.idx_pad)).unsqueeze(-2) #[batch_size, 1, max_len]
                mask_s = torch.from_numpy(sequence_mask(batch_src_len,mask_n_initials=2))
                if self.cuda:
                    x1 = x1.cuda()
                    x1_mask = x1_mask.cuda()
                    mask_s = mask_s.cuda()
                h1 = self.model.forward(x1,x1_mask)
                mask_s = mask_s.unsqueeze(-1).type(torch.float64)

                if len(files)>1:
                    ltgt = ft.readline()
                    tgt = [t for t in self.token.tokenize(ltgt)]
                    idx_tgt = [self.vocab[t] for t in tgt]
                    idx_tgt.insert(0,self.vocab.idx_bos)
                    idx_tgt.append(self.vocab.idx_eos)
                    idx_tgt.insert(0,self.vocab.idx_cls)
                    batch_tgt = np.array([idx_tgt])
                    batch_tgt_len = np.array([len(idx_tgt)])
                    x2 = torch.from_numpy(batch_tgt) #[batch_size, max_len] the original words with padding
                    x2_mask = torch.as_tensor((x2 != self.vocab.idx_pad)).unsqueeze(-2) #[batch_size, 1, max_len]
                    mask_t = torch.from_numpy(sequence_mask(batch_tgt_len,mask_n_initials=2))
                    if self.cuda:
                        x2 = x2.cuda()
                        x2_mask = x2_mask.cuda()
                        mask_t = mask_t.cuda()
                    h2 = self.model.forward(x2,x2_mask)
                    mask_t = mask_t.unsqueeze(-1).type(torch.float64)

            
                if self.pooling == 'max':
                    s, _ = torch.max(h1*mask_s + (1.0-mask_s)*-999.9, dim=1) #-999.9 should be -Inf but it produces an nan when multiplied by 0.0
                    if len(files)>1:
                        t, _ = torch.max(h2*mask_t + (1.0-mask_t)*-999.9, dim=1) #-999.9 should be -Inf but it produces an nan when multiplied by 0.0
                elif self.pooling == 'mean':
                    s = torch.sum(h1 * mask_s, dim=1) / torch.sum(mask_s, dim=1)
                    if len(files)>1:
                        t = torch.sum(h2 * mask_t, dim=1) / torch.sum(mask_t, dim=1)
                elif self.pooling == 'cls':
                    s = h1[:, 0, :] # take embedding of first token <cls>
                    if len(files)>1:
                        t = h2[:, 0, :] # take embedding of first token <cls>
                elif self.pooling == 'align' and len(files) == 2:
                    #h1 [bs, sl, es] embeddings of source words after encoder (<cls> <bos> s1 s2 ... sI <eos> <pad> ...)
                    #h2 [bs, tl, es] embeddings of target words after encoder (<cls> <bos> t1 t2 ... tJ <eos> <pad> ...)
                    S_st = torch.bmm(h1, torch.transpose(h2, 2, 1)) * self.align_scale #[bs, sl, es] x [bs, es, tl] = [bs, sl, tl]            
                    ### scale S_st to <=10
#                    mask_s = mask_s.type(torch.float64)
#                    S_st_masked_s = S_st * mask_s #[bs, ls, lt] * [bs, ls, 1]
#                    S_st_masked_st = S_st_masked_s * mask_t.transpose(2,1) #[bs, ls, lt] * [bs, 1, lt]
#                    max_div_10 = torch.max(S_st_masked_st.pow(2)).pow(0.5) / 1.0 ### largest number will be 10.0
#                    S_st = S_st / max_div_10
                else:
                    logging.error('bad pooling method: {}'.format(self.pooling))

                if len(files)==1:
                    sentence = torch.Tensor.cpu(self.norm(s)).detach().numpy()[0]
                    print(' '.join([str(tok) for tok in sentence]))
                elif len(files)>1:
                    if self.pooling == 'align':
                        #i add mean pooling
                        s = torch.sum(h1 * mask_s, dim=1) / torch.sum(mask_s, dim=1)
                        t = torch.sum(h2 * mask_t, dim=1) / torch.sum(mask_t, dim=1)
                        sim = F.cosine_similarity(s, t, dim=1, eps=1e-12)
                        ### and i show the alignments
                        align = []
                        sim = sim[0].cpu().detach().numpy()
                        align.append(['{:.4f}'.format(sim)] + src) #mean pooling is added here
                        for t in range(len(tgt)):
                            row = []
                            for s in range(len(src)):
                                row.append('{:.2f}'.format(S_st[0,2+s,t+2]))
                            align.append([tgt[t]] + row)
                        #print(np.matrix(align))
                        #s = [[str(e) for e in row] for row in align]
                        lens = [max(map(len, col)) for col in zip(*align)]
                        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
                        table = [fmt.format(*row) for row in align]
                        print('\n'.join(table))
                    else:
                        sim = F.cosine_similarity(s, t, dim=1, eps=1e-12)
                        print(torch.Tensor.cpu(sim).detach().numpy()[0])

        logging.info('End testing')

    def norm(self,x):
        if not self.normalize:
            return x
        return F.normalize(x,p=2,dim=1,eps=1e-12)
