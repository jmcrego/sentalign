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
    def __init__(self):
        self.src = [] #source tokens (strings)
        self.tgt = [] #target tokens (strings)
        self.sidx = [] #batch of [<cls>, <s1>, ..., <sI>, [<pad>]*]
        self.tidx = [] #batch of [<sep>, <t1>, ..., <tJ>, [<pad>]*]
        self.lsrc = [] #[I1, I2, ...] size of src sentences including <cls>
        self.ltgt = [] #[J1, J2, ...] size of src sentences including <sep>
        self.link = [] #batch of alignment pairs [[0,0], [1,1], ...] or empty
        self.parallel = [] #for each sentence pair indicates if it is parallel:True or divergent:False
        self.indexs = [] #position in the original file

    def __len__(self):
        return len(self.indexs)

    def add(self, index, data):
        self.indexs.append(index)
        if len(data) == 7:
            (src,tgt,sidx,tidx,link,is_swap,is_uneven) = data
        else:
            logging.error('bad number of fields in example index={}'.format(index))
            sys.exit()

        sidx.insert(0,idx_cls)
        tidx.insert(0,idx_sep)
        self.src.append(src)
        self.tgt.append(tgt)
        self.lsrc.append(len(sidx)) #length of source sentence considering <cls>
        self.ltgt.append(len(tidx)) #length of target sentence considering <sep>
        self.sidx.append(sidx) ### [<cls>, <s1>, ..., <sI>]
        self.tidx.append(tidx) ### [<sep>, <t1>, ..., <tJ>]
        self.link.append(link) ### sequence of links (may be empty)
        self.parallel.append(not is_uneven)

    def pad(self):
        #convert self.lsrc/self.ltgt into np array
        self.lsrc = np.array(self.lsrc)
        self.ltgt = np.array(self.ltgt)
        #compute maxlsrc/maxltgt
        self.maxlsrc = np.max(self.lsrc)
        self.maxltgt = np.max(self.ltgt)
        ### pad source/target sentences
        for b in range(len(self.indexs)):
            self.sidx[b] += [idx_pad]*(self.maxlsrc-len(self.sidx[b])) 
            self.tidx[b] += [idx_pad]*(self.maxltgt-len(self.tidx[b]))
        ### build alignment matrix
        self.matrix = np.zeros((len(self.indexs), self.maxlsrc-1, self.maxltgt-1), dtype=bool) #initially all pairs are not aligned (False), do not considers <cls>, <sep>
        for b in range(len(self.link)):
            for (s, t) in self.link[b]:
                self.matrix[b,s,t] = True
        #convert self.sidx/self.tidx/self.parallel into np array
        self.sidx = np.array(self.sidx)
        self.tidx = np.array(self.tidx)
        self.parallel = np.array(self.parallel)


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
        print('link')
        print(self.link)
        print('matrix (np)')
        print(self.matrix)
        print('parallel (np)')
        print(self.parallel)


####################################################################
### Dataset ########################################################
####################################################################

class Dataset():

    def __init__(self, token, vocab, max_length=0, batch_size=32, p_swap=0.0, p_uneven=0.0, is_infinite=False, max_sentences_per_file=0):
        self.token = token
        self.vocab = vocab
        self.max_length = max_length
        self.is_infinite = is_infinite
        self.batch_size = batch_size
        self.p_swap = p_swap
        self.p_uneven = p_uneven
        self.max_sentences_per_file = max_sentences_per_file
        self.data = []

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
        n_uneven = 0
        n_swap = 0
        prev_lt = ""
        for ls, lt, la in zip(fs,ft,fa):
            if fsrc.endswith('.gz'): 
                ls = ls.decode('utf8')
            if ftgt.endswith('.gz'): 
                lt = lt.decode('utf8')
            if fali.endswith('.gz'): 
                la = la.decode('utf8')

            is_uneven = False
            if self.p_uneven > 0.0 and random.random() < self.p_uneven:
                if len(prev_lt):
                    curr_lt = lt
                    lt = prev_lt
                    prev_lt = curr_lt
                    is_uneven = True
                else:
                    prev_lt = lt

            is_swap = False
            if self.p_swap > 0.0 and random.random() < self.p_swap:
                ssnt = self.tokenize(lt)
                tsnt = self.tokenize(ls)
                is_swap = True
            else:
                ssnt = self.tokenize(ls)
                tsnt = self.tokenize(lt)
            sidx = self.tok2idx(ssnt)
            tidx = self.tok2idx(tsnt)
            link = self.get_links(la,is_swap,is_uneven) #alig = la.strip(' \n').split()
            ### filtering
            if self.max_length > 0 and (len(sidx) > self.max_length or len(tidx) > self.max_length):
                nfilt += 1
                continue

            nsent += 1
            nunks_src += sidx.count(idx_unk)
            nunks_tgt += tidx.count(idx_unk) 
            ntoks_src += len(sidx)
            ntoks_tgt += len(tidx)
            if is_uneven:
                n_uneven += 1
            if is_swap:
                n_swap += 1
            self.data.append([ssnt,tsnt,sidx,tidx,link,is_swap,is_uneven])
            if self.max_sentences_per_file > 0 and nsent >= self.max_sentences_per_file: 
                break

        logging.info('found {} sentences ({} filtered, {} uneven, {} swap), {}/{} tokens ({:.3f}/{:.3f} %OOVs) in files: [{},{},{}]'.format(nsent,nfilt,n_uneven,n_swap,ntoks_src,ntoks_tgt,100.0*nunks_src/ntoks_src,100.0*nunks_tgt/ntoks_tgt,fsrc,ftgt,fali))


    def get_links(self, la, is_swap, is_uneven):
        link = []
        if not is_uneven:
            for s_t in la.strip(' \n').split():
                (s,t) = map(int, s_t.split('-'))
                if is_swap:
                    link.append([t,s])
                else:
                    link.append([s,t])
        return link

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
            if fsrc.endswith('.gz'): 
                ls = ls.decode('utf8')
            if ftgt.endswith('.gz'): 
                lt = lt.decode('utf8')
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
            ### filtering
            if self.max_length > 0 and (len(sidx) > self.max_length or len(tidx) > self.max_length):
                nfilt += 1
                continue
            nsent += 1
            nunks_src += sidx.count(idx_unk)
            nunks_tgt += tidx.count(idx_unk) 
            ntoks_src += len(sidx)
            ntoks_tgt += len(tidx)
            self.data.append([ssnt,tsnt,sidx,tidx,[],False,False])
        logging.info('found {} sentences ({} filtered), {}/{} tokens ({:.3f}/{:.3f} %OOVs) in files: [{},{}]'.format(nsent,nfilt,ntoks_src,ntoks_tgt,100.0*nunks_src/ntoks_src,100.0*nunks_tgt/ntoks_tgt,fsrc,ftgt))


    def build_batches(self):
        self.batches = []
        if len(self.data) == 0:
            logging.error('no examples available to build batches')
            sys.exit()

        logging.debug('sorting data by source/target sentence length to minimize padding over {} examples'.format(len(self.data)))
        data_len1 = [len(x[0]) for x in self.data]
        data_len2 = [len(x[1]) for x in self.data]
        indexs = np.lexsort((data_len1,data_len2))

        currbatch = batch() 
        for index in indexs:
            currbatch.add(index,self.data[index])

            if len(currbatch) >= self.batch_size: ### batch filled
                currbatch.pad()
                self.batches.append(currbatch)
                currbatch = batch()

        if len(currbatch): ### record last batch
            currbatch.pad()
            self.batches.append(currbatch)

        logging.info('built {} batches'.format(len(self.batches)))
        del self.data


    def __iter__(self):

        indexs = [i for i in range(len(self.batches))]
        while True: 

            logging.debug('shuffling batches')
            shuffle(indexs)

            for index in indexs:
                yield self.batches[index]

            if not self.is_infinite:
                break










