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
from src.dataset import Vocab, Dataset, OpenNMTTokenizer, batch
from src.trainer import sequence_mask
from src.model import make_model
from src.optim import NoamOpt, LabelSmoothing, Align, Cosine, ComputeLossMLM, ComputeLossALI, ComputeLossCOS


class Infer():

    def __init__(self, opts):
        self.dir = opts.dir
        self.cuda = opts.cfg['cuda']
        self.vocab = Vocab(opts.dir+'/vocab')
        V = len(self.vocab)
        N = opts.cfg['num_layers']
        d_model = opts.cfg['hidden_size']
        d_ff = opts.cfg['feedforward_size']
        h = opts.cfg['num_heads']
        dropout = opts.cfg['dropout']

        self.model = make_model(V, N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)
        if self.cuda:
            self.model.cuda()
        self.load_checkpoint()

        self.align_scale = opts.align_scale
        self.batch_size = opts.batch_size
        self.pooling = opts.pooling
        self.matrix = opts.matrix
        self.token = None
#        self.token = OpenNMTTokenizer(**opts.cfg['token'])

    def load_checkpoint(self):
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
        self.data_test = Dataset(self.token,self.vocab,max_length=0,is_infinite=False)
        if len(files) == 2:
            self.data_test.add2files(files[0],files[1])
        elif len(files) == 3:
            self.data_test.add3files(files[0],files[1],files[2])
        else:
            logging.error('2 or 3 files must be passed for inference')
            sys.exit()

        self.data_test.build_batches(self.batch_size,p_swap=0.0,p_uneven=0.0)


        with torch.no_grad():
            self.model.eval() ### avoids dropout
            for batch in self.data_test:
                xy = torch.from_numpy(np.append(batch.sidx, batch.tidx, axis=1)) #xy [batch_size, max_len] contains the original words after concat(x,y) [input for ALI]                
                mask_xy = torch.as_tensor((xy != self.vocab.idx_pad)) #mask_xy [batch_size, max_len] True for x or y words in xy; false for <pad> (<cls>/<sep> included)
                if self.cuda:
                    xy = xy.cuda()
                    mask_xy = mask_xy.cuda()
                h_xy = self.model.forward(xy, mask_xy.unsqueeze(-2))
                ls = batch.maxlsrc-1 ### maxlength of source sequence without <cls>
                lt = batch.maxltgt-1 ### maxlength of target sequence without <sep>
                hs = h_xy[:,1:ls+1,:] #[bs, ls, es]
                ht = h_xy[:,ls+2:,:] #[bs, lt, es]
                mask_s = mask_xy[:,1:ls+1].type(torch.float64).unsqueeze(-1) #[bs, ls, 1]
                mask_t = mask_xy[:,ls+2:,].type(torch.float64).unsqueeze(-1) #[bs, lt, 1]
                if self.pooling == 'max':
                    s, _ = torch.max(hs*mask_s + (1.0-mask_s)*-999.9, dim=1) #-999.9 should be -Inf but it produces an nan when multiplied by 0.0
                    t, _ = torch.max(ht*mask_t + (1.0-mask_t)*-999.9, dim=1) #-999.9 should be -Inf but it produces an nan when multiplied by 0.0
                elif self.pooling == 'mean':
                    s = torch.sum(hs*mask_s, dim=1) / torch.sum(mask_s, dim=1)
                    t = torch.sum(ht*mask_t, dim=1) / torch.sum(mask_t, dim=1)
                elif self.pooling == 'cls':
                    s = h_xy[:, 0, :] # take embedding of <cls>
                    t = h_xy[:,ls+1,:] # take embedding of <sep>
                else:
                    logging.error('bad pooling method: {}'.format(self.pooling))
                s = F.normalize(s,p=2,dim=1,eps=1e-12).unsqueeze(-2) #[bs, 1, es]
                t = F.normalize(t,p=2,dim=1,eps=1e-12).unsqueeze(-1) #[bs, es, 1]
                print('s',s.shape)
                print('t',t.shape)
#                sim = F.cosine_similarity(s, t, dim=1, eps=1e-12).cpu().detach().numpy()
                DP = torch.bmm(s, t).squeeze(0).squeeze(0)
                print('DP',DP.shape)
                ### output
                if self.matrix:
                    hs = F.normalize(hs,p=2,dim=2,eps=1e-12)
                    ht = F.normalize(ht,p=2,dim=2,eps=1e-12)
                    DP_st = torch.bmm(hs, torch.transpose(ht, 2, 1)) * self.align_scale #[bs, sl, es] x [bs, es, tl] = [bs, sl, tl]            

                for b in range(len(DP)):
                    if self.matrix:
                        print_matrix(DP_st[b], batch.src[b], batch.tgt[b], DP[b], batch.indexs[b])
                    else:
                        print("{}\t{:.6f}\t{}\t{}".format(batch.indexs[b],DP[b],' '.join(batch.src[b]),' '.join(batch.tgt[b])))

        logging.info('End testing')

def print_matrix(DP_st, src, tgt, DP, index):
    align = []
    align.append(['{:.6f}'.format(DP)] + src)
    for t in range(len(tgt)):
        row = []
        for s in range(len(src)):
            row.append('{:.2f}'.format(DP_st[s,t]))
        align.append([tgt[t]] + row)
    #print(np.matrix(align))
    #s = [[str(e) for e in row] for row in align]
    lens = [max(map(len, col)) for col in zip(*align)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in align]
    print(index)
    print('\n'.join(table))


def norm(x,d):
    return F.normalize(x,p=2,dim=d,eps=1e-12)
