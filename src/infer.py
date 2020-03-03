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
from src.trainer import sequence_mask, format_batch
from src.model import make_model
from src.optim import NoamOpt, LabelSmoothing, Align, Cosine, ComputeLossMLM, ComputeLossALI, ComputeLossCOS, sentence_embedding


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
        self.data_test = Dataset(self.token,self.vocab,max_length=0,batch_size=self.batch_size,p_swap=0.0,p_uneven=0.0,is_infinite=False,max_sentences_per_file=0)
        if len(files) == 2:
            self.data_test.add2files(files[0],files[1])
        elif len(files) == 3:
            self.data_test.add3files(files[0],files[1],files[2])
        else:
            logging.error('2 or 3 files must be passed for inference')
            sys.exit()

        self.data_test.build_batches()


        with torch.no_grad():
            self.model.eval() ### avoids dropout
            for batch in self.data_test:
                st, st_mlm, st_mlm_ref, st_mask, st_matrix, st_uneven = format_batch(self.vocab, self.cuda, batch)
                h_st = self.model.forward(st, st_mask.unsqueeze(-2))
                s, t, hs, ht, s_mask, t_mask = sentence_embedding(h_st, st_mask, batch.maxlsrc-1, self.pooling, norm_st=True, norm_h=True)
                DP = torch.bmm(s.unsqueeze(-2), t.unsqueeze(-1)).squeeze(-1).squeeze(-1).cpu().detach().numpy() #[bs, 1, 1] => [bs]
                if self.matrix:
                    DP_st = torch.bmm(hs, torch.transpose(ht, 2, 1)) #[bs, sl, es] x [bs, es, tl] = [bs, sl, tl]            

                ### output
                for b in range(len(DP)):
                    if self.matrix:
                        print_matrix(DP_st[b], batch.src[b], batch.tgt[b], DP[b], batch.indexs[b])
                    else:
                        print("{}\t{:.6f}\t{}\t{}".format(batch.indexs[b],DP[b],' '.join(batch.src[b]),' '.join(batch.tgt[b])))

                    my_enc_layer=2
                    my_head = 6
                    my_attn = self.model.encoder.layers[my_enc_layer].self_attn.attn[0, my_head].cpu().detach().numpy()
                    print(my_attn.shape)
                    print(list(my_attn))

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
