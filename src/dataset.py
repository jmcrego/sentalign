import io
import os
import sys
import gzip
import torch
import logging
import pyonmttok
import numpy as np
import json
import six
import random
from copy import deepcopy
#from torch.nn.utils.rnn import pad_sequence
from random import shuffle
from collections import defaultdict

idx_pad = 0
idx_unk = 1
idx_bos = 2
idx_eos = 3
idx_msk = 4
idx_sep = 5
idx_cls = 6
str_pad = '<pad>'
str_unk = '<unk>'
str_bos = '<bos>'
str_eos = '<eos>'
str_msk = '<msk>'
str_sep = '<sep>'
str_cls = '<cls>'

####################################################################
### OpenNMTTokenizer ###############################################
####################################################################
class OpenNMTTokenizer():

  def __init__(self, **kwargs):
    if 'mode' not in kwargs:
       logging.error('error: missing mode in tokenizer')
       sys.exit()
    mode = kwargs["mode"]
    del kwargs["mode"]
    self.tokenizer = pyonmttok.Tokenizer(mode, **kwargs)
    logging.info('built tokenizer mode={} {}'.format(mode,kwargs))

  def tokenize(self, text):
    tokens, _ = self.tokenizer.tokenize(text)
    return tokens

  def detokenize(self, tokens):
    return self.tokenizer.detokenize(tokens)


####################################################################
### Vocab ##########################################################
####################################################################
class Vocab():

    def __init__(self, file):
        self.tok_to_idx = {} 
        self.idx_to_tok = [] 
        self.idx_to_tok.append(str_pad)
        self.tok_to_idx[str_pad] = len(self.tok_to_idx) #0
        self.idx_to_tok.append(str_unk)
        self.tok_to_idx[str_unk] = len(self.tok_to_idx) #1
        self.idx_to_tok.append(str_bos)
        self.tok_to_idx[str_bos] = len(self.tok_to_idx) #2
        self.idx_to_tok.append(str_eos)
        self.tok_to_idx[str_eos] = len(self.tok_to_idx) #3
        self.idx_to_tok.append(str_msk)
        self.tok_to_idx[str_msk] = len(self.tok_to_idx) #4
        self.idx_to_tok.append(str_sep)
        self.tok_to_idx[str_sep] = len(self.tok_to_idx) #5
        self.idx_to_tok.append(str_cls)
        self.tok_to_idx[str_cls] = len(self.tok_to_idx) #6

        self.idx_pad = idx_pad
        self.idx_unk = idx_unk
        self.idx_bos = idx_bos
        self.idx_eos = idx_eos
        self.idx_msk = idx_msk
        self.idx_sep = idx_sep
        self.idx_cls = idx_cls

        with io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            for line in f:
                tok = line.strip()
                if tok not in self.tok_to_idx:
                    self.idx_to_tok.append(tok)
                    self.tok_to_idx[tok] = len(self.tok_to_idx)
        logging.info('read Vocab ({} entries) file={}'.format(len(self),file))

    def is_reserved(self, s):
        return s<7

    def __len__(self):
        return len(self.idx_to_tok)

    def __iter__(self):
        for tok in self.idx_to_tok:
            yield tok

    def __contains__(self, s): ### implementation of the method used when invoking : entry in vocab
        if type(s) == int: ### testing an index
            return s>=0 and s<len(self)
        ### testing a string
        return s in self.tok_to_idx

    def __getitem__(self, s): ### implementation of the method used when invoking : vocab[entry]
        if type(s) == int: ### input is an index, i want the string
            if s not in self:
                logging.error("key \'{}\' not found in vocab".format(s))
                sys.exit()
            return self.idx_to_tok[s]
            ### input is a string, i want the index
        if s not in self: 
            return idx_unk
        return self.tok_to_idx[s]

####################################################################
### batch ##########################################################
####################################################################

class batch():
    def __init__(self,p_swap=0.0):
        self.src = []
        self.tgt = []
        self.sidx = [] #batch of [<cls>, <s1>, ..., <sI>, [<pad>]*]
        self.tidx = [] #batch of [<sep>, <t1>, ..., <tJ>, [<pad>]*]
        self.lsrc = [] #[I1, I2, ...] size of src sentences including <cls>
        self.ltgt = [] #[J1, J2, ...] size of src sentences including <sep>
        self.a = [] #batch of alignment pairs [[0,0], [1,1], ...]
        self.indexs = [] #[i1, i2, ...] position in the original file
        self.p_swap = p_swap

    def __len__(self):
        return len(self.sidx)

    def add(self, index, idx, snt):
        self.indexs.append(index)
        if len(idx) == 3:
            (sidx, tidx, ali) = idx
        elif len(idx) == 2:
            (sidx, tidx) = idx
            ali = []
        (src, tgt) = snt

        if len(tidx) > 1 and random.random() < self.p_swap:
            aidx = list(tidx)
            tidx = list(sidx)
            sidx = list(aidx)
            aux = list(tgt)
            tgt = list(src)
            src = list(tgt)
            do_swap = True
        else:
            do_swap = False

        self.src.append(src)
        self.tgt.append(tgt)
        sidx.insert(0,idx_cls)
        self.lsrc.append(len(sidx)) #lsrc is the position of sI
        self.sidx.append(sidx) ### [<cls>, <s1>, ..., <sI>]

        if len(tidx) > 0:
            tidx.insert(0,idx_sep)
            self.ltgt.append(len(tidx)) #ltgt is the position of tJ
            self.tidx.append(tidx) ### [<sep>, <t1>, ..., <tJ>]

        if len(ali) > 0:
            pairs = []
            for st in ali:
                s,t = map(int, st.split('-'))
                if do_swap:
                    pairs.append([t,s])
                else:
                    pairs.append([s,t])
            self.a.append(pairs)

    def pad(self):
        #convert self.lsrc/self.ltgt into np array
        self.lsrc = np.array(self.lsrc)
        self.maxlsrc = np.max(self.lsrc)
        if len(self.tidx) > 0:
            self.ltgt = np.array(self.ltgt)
            self.maxltgt = np.max(self.ltgt)
        bs = len(self.sidx)
        ### pad source/target sentences
        for b in range(bs):
            self.sidx[b] += [idx_pad]*(self.maxlsrc-len(self.sidx[b])) 
            if len(self.tidx) > 0:
                self.tidx[b] += [idx_pad]*(self.maxltgt-len(self.tidx[b]))
        ### build alignment matrix
        if len(self.a) > 0:
            self.ali = np.full((bs, self.maxlsrc-1, self.maxltgt-1), 1.0) #initially all pairs are not aligned (divergent), do not consider <cls>, <sep>
            for b in range(bs):
                for (s, t) in self.a[b]:
                    self.ali[b,s,t] = -1.0 #is aligned
        #convert self.sidx/self.tidx/self.ali into np array
        self.sidx = np.array(self.sidx)
        if len(self.tidx) > 0:
            self.tidx = np.array(self.tidx)
        if len(self.a) > 0:
            self.ali = np.array(self.ali)

    def dump(self):
        print('indexs')
        print(self.indexs)
        print('[bs, ls, lt]')
        print('[{}, {}, {}]'.format(len(self.sidx),self.maxlsrc,self.maxltgt))
        print('src')
        print(self.src)
        print('tgt')
        print(self.tgt)
        print('lsrc (np)')
        print(self.lsrc)
        print('maxlsrc (np)')
        print(self.maxlsrc)
        print('ltgt (np)')
        print(self.ltgt)
        print('maxltgt (np)')
        print(self.maxltgt)
        print('sidx (np)')
        print(self.sidx)
        print('tidx (np)')
        print(self.tidx)
        print('a')
        print(self.a)
        print('ali (np)')
        print(self.ali)


####################################################################
### Dataset ########################################################
####################################################################

class Dataset():

    def __init__(self, token, vocab, max_length=0, is_infinite=False):
        self.token = token
        self.vocab = vocab
        self.max_length = max_length
        self.is_infinite = is_infinite
        self.idx = []
        self.snt = [] #only to debug

    def tokenize(self,l):
        if self.token is not None:
            tok = self.token.tokenize(l.strip(' \n'))
        else:
            tok = l.strip(' \n').split()
        return tok

    def tok2idx(self, tok):
        idx = [self.vocab[x] for x in tok]
        return idx

    def add3files(self, fsrc, ftgt, fali):
        if fsrc.endswith('.gz'): 
            fs = gzip.open(fsrc, 'rb')
        else: 
            fs = io.open(fsrc, 'r', encoding='utf-8', newline='\n', errors='ignore')
        if ftgt.endswith('.gz'): 
            ft = gzip.open(ftgt, 'rb')
        else: 
            ft = io.open(ftgt, 'r', encoding='utf-8', newline='\n', errors='ignore')
        if fali.endswith('.gz'): 
            fa = gzip.open(fali, 'rb')
        else: 
            fa = io.open(fali, 'r', encoding='utf-8', newline='\n', errors='ignore')


        ntoks_src = 0
        nunks_src = 0
        ntoks_tgt = 0
        nunks_tgt = 0
        nsent = 0
        nfilt = 0
        for ls, lt, la in zip(fs,ft,fa):
            ssnt = self.tokenize(ls)
            sidx = self.tok2idx(ssnt)
            tsnt = self.tokenize(lt)
            tidx = self.tok2idx(tsnt)
            if self.max_length > 0 and (len(sidx) > self.max_length or len(tidx) > self.max_length):
                nfilt += 1
                continue
            ### align
            alig = la.strip(' \n').split()
            nsent += 1
            nunks_src += sidx.count(idx_unk)
            nunks_tgt += tidx.count(idx_unk) 
            ntoks_src += len(sidx)
            ntoks_tgt += len(tidx)
            self.idx.append([sidx,tidx,alig])
            self.snt.append([ssnt,tsnt])
#            if len(self.idx) > 1000:
#                break
        logging.info('found {} sentences ({} filtered), {}/{} tokens ({:.3f}/{:.3f} %OOVs) in files: [{},{},{}]'.format(nsent,nfilt,ntoks_src,ntoks_tgt,100.0*nunks_src/ntoks_src,100.0*nunks_tgt/ntoks_tgt,fsrc,ftgt,fali))


    def add2files(self, fsrc, ftgt):
        if fsrc.endswith('.gz'): 
            fs = gzip.open(fsrc, 'rb')
        else: 
            fs = io.open(fsrc, 'r', encoding='utf-8', newline='\n', errors='ignore')
        if ftgt.endswith('.gz'): 
            ft = gzip.open(ftgt, 'rb')
        else: 
            ft = io.open(ftgt, 'r', encoding='utf-8', newline='\n', errors='ignore')

        ntoks_src = 0
        nunks_src = 0
        ntoks_tgt = 0
        nunks_tgt = 0
        nsent = 0
        nfilt = 0
        for ls, lt in zip(fs,ft):
            ### src tokenize
            if self.token is not None:
                ssnt = self.token.tokenize(ls.strip(' \n'))
            else:
                ssnt = ls.strip(' \n').split()
            ### src vocab
            sidx = [self.vocab[s] for s in ssnt]
            ### tgt tokenize
            if self.token is not None:
                tsnt = self.token.tokenize(lt.strip(' \n'))
            else:
                tsnt = lt.strip(' \n').split()
            ### tgt vocab
            tidx = [self.vocab[t] for t in tsnt]
            if self.max_length > 0 and (len(sidx) > self.max_length or len(tidx) > self.max_length):
                nfilt += 1
                continue
            nsent += 1
            nunks_src += sidx.count(idx_unk)
            nunks_tgt += tidx.count(idx_unk) 
            ntoks_src += len(sidx)
            ntoks_tgt += len(tidx)
            self.idx.append([sidx,tidx])
            self.snt.append([ls,lt])
        logging.info('found {} sentences ({} filtered), {}/{} tokens ({:.3f}/{:.3f} %OOVs) in files: [{},{}]'.format(nsent,nfilt,ntoks_src,ntoks_tgt,100.0*nunks_src/ntoks_src,100.0*nunks_tgt/ntoks_tgt,fsrc,ftgt))


    def add1file(self, fsrc):
        if fsrc.endswith('.gz'): 
            fs = gzip.open(fsrc, 'rb')
        else: 
            fs = io.open(fsrc, 'r', encoding='utf-8', newline='\n', errors='ignore')

        ntoks = 0
        nunks = 0
        nsent = 0
        nfilt = 0
        for ls in fs:
            ### src tokenize
            if self.token is not None:
                ssnt = self.token.tokenize(ls.strip(' \n'))
            else:
                ssnt = ls.strip(' \n').split()
            ### src vocab
            sidx = [self.vocab[s] for s in ssnt]
            if self.max_length > 0 and len(sidx) > self.max_length:
                nfilt += 1
                continue
            nsent += 1
            nunks += sidx.count(idx_unk) 
            ntoks += len(sidx)
            self.idx.append([sidx])
            self.snt.append([ssnt])
        logging.info('found {} sentences ({} filtered), {} tokens ({:.3f}% OOVs) in files: [{}]'.format(nsent,nfilt,ntoks,100.0*nunks/ntoks,fsrc))


    def build_batches(self, batch_size, p_swap=0.0):
        self.batches = []
        self.batch_size = batch_size
        if len(self.idx) == 0:
            logging.error('no examples available to build batches')
            sys.exit()

        if len(self.idx[0]) > 1:
            logging.debug('sorting data by source/target sentence length to minimize padding over {} examples'.format(len(self.idx)))
            data_len1 = [len(x[0]) for x in self.idx]
            data_len2 = [len(x[1]) for x in self.idx]
            indexs = np.lexsort((data_len1,data_len2))
        else:
            logging.debug('sorting data by source sentence length to minimize padding over {} examples'.format(len(self.idx)))
            data_len1 = [len(x[0]) for x in self.idx]
            indexs = np.argsort(data_len1)

        currbatch = batch(p_swap) 
        for i in range(len(indexs)):
            index = indexs[i]

            currbatch.add(index,self.idx[index],self.snt[index])
            if len(currbatch) == self.batch_size or i == len(indexs)-1: ### record new batch
                currbatch.pad()
                self.batches.append(deepcopy(currbatch))
                currbatch = batch()
        logging.info('built {} batches'.format(len(self.batches)))

    def __len__(self):
        return len(self.idx)


    def __iter__(self):

        indexs = [i for i in range(len(self.batches))]
        while True: 

            logging.debug('shuffling batches')
            shuffle(indexs)

            for index in indexs:
                yield self.batches[index]

            if not self.is_infinite:
                break










