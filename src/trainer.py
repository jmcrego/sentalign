import numpy as np
import torch
import logging
import time
import random
import sys
import glob
import os
from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from src.dataset import Vocab, Dataset, OpenNMTTokenizer, batch
from src.model import make_model
from src.optim import NoamOpt, LabelSmoothing, Align, Cosine, ComputeLossMLM, ComputeLossALI, ComputeLossCOS


def sequence_mask(lengths, mask_n_initials=0):
    bs = len(lengths)
    l = lengths.max()
    msk = np.cumsum(np.ones([bs,l],dtype=int), axis=1).T #[l,bs] (transpose to allow combine with lenghts)
    mask = (msk <= lengths-1) ### i use lenghts-1 because the last unpadded word is <eos> and i want it masked too
    if mask_n_initials:
        mask &= (msk > mask_n_initials)
    return mask.T #[bs,l]

class stats():

    def __init__(self):
        self.n_steps = defaultdict(int)
        self.sum_loss = defaultdict(float)
        self.n_ok = defaultdict(int)
        self.n_pred = defaultdict(int)
        self.start = time.time()

    def add(self,name,loss,nok=0,npred=0):
        self.n_steps[name] += 1
        self.sum_loss[name] += loss
        self.n_ok[name] += nok
        self.n_pred[name] += npred

    def report(self,n_steps,trn_val_tst):
        res = []
        res.append('loss={:.4f} {:.2f} steps/sec'.format(self.sum_loss['loss']/self.n_steps['loss'],self.n_steps['loss']/(time.time()-self.start)))
        res.append(self.name2report('MLM'))
        res.append(self.name2report('ALI'))
        res.append(self.name2report('COS'))
        logging.info('{} n_steps: {}\t{}'.format(trn_val_tst,n_steps,'\t'.join(res)))
        self.start = time.time()
        ### all losses are averaged per n_preds and per n_steps

    def name2report(self,name):
        if name not in self.sum_loss:
            return ''
        loss = self.sum_loss[name]/self.n_steps[name]
        npred = self.n_pred[name]
        acc = 0.0
        if self.n_pred[name] > 0:
            acc = 1.0*self.n_ok[name]/self.n_pred[name]
        self.n_steps[name] = 0
        self.sum_loss[name] = 0.0
        self.n_ok[name] = 0
        self.n_pred[name] = 0
        return '({}: loss:{:.4f}, Acc:{:.4f}/{})'.format(name,loss,acc,npred)


class Trainer():

    def __init__(self, opts):
        self.dir = opts.dir
        self.vocab = Vocab(opts.dir+'/vocab')
        self.n_steps_so_far = 0
        self.report_every_steps = opts.train['report_every_steps']
        self.validation_every_steps = opts.train['validation_every_steps']
        self.checkpoint_every_steps = opts.train['checkpoint_every_steps']
        self.keep_last_n = opts.train['keep_last_n']
        self.batch_size = opts.train['batch_size']
        self.max_length = opts.train['max_length']
        self.swap_bitext = opts.train['swap_bitext'] 
        self.uneven_bitext = opts.train['uneven_bitext'] 
        self.step_mlm = opts.train['steps']['mlm']
        self.step_ali = opts.train['steps']['ali']
        self.step_cos = opts.train['steps']['cos']
        self.max_sentences_per_file = opts.train['max_sentences_per_file']

        V = len(self.vocab)
        N = opts.cfg['num_layers']
        d_model = opts.cfg['hidden_size']
        d_ff = opts.cfg['feedforward_size']
        h = opts.cfg['num_heads']
        dropout = opts.cfg['dropout']
        factor = opts.cfg['factor']
        label_smoothing = opts.cfg['label_smoothing']
        warmup_steps = opts.cfg['warmup_steps']
        lrate = opts.cfg['learning_rate']
        beta1 = opts.cfg['beta1']
        beta2 = opts.cfg['beta2']
        eps = opts.cfg['eps']
        self.cuda = opts.cfg['cuda']

        self.model = make_model(V, N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)
        if self.cuda:
            self.model.cuda()

        self.optimizer = NoamOpt(d_model, factor, warmup_steps, torch.optim.Adam(self.model.parameters(), lr=lrate, betas=(beta1, beta2), eps=eps))
        self.crit_mlm = LabelSmoothing(size=V, padding_idx=self.vocab.idx_pad, smoothing=label_smoothing)
        self.crit_ali = Align()
        self.crit_cos = Cosine()

        if self.cuda:
            self.crit_mlm.cuda()
            self.crit_ali.cuda()
            self.crit_cos.cuda()

        self.load_checkpoint() #loads if exists
        self.computeloss_mlm = ComputeLossMLM(self.model.generator, self.crit_mlm, self.optimizer)
        self.computeloss_ali = ComputeLossALI(self.crit_ali, self.step_ali, self.optimizer)
        self.computeloss_cos = ComputeLossCOS(self.crit_cos, self.step_cos, self.optimizer)

        logging.info('read Train data')
        self.data_train = Dataset(None,self.vocab,max_length=self.max_length,batch_size=self.batch_size,p_swap=self.swap_bitext,p_uneven=self.uneven_bitext,is_infinite=True,max_sentences_per_file=self.max_sentences_per_file)
        for (fs,ft,fa) in opts.train['train']:
            self.data_train.add3files(fs,ft,fa)
        logging.info('build Train batches')
        self.data_train.build_batches()

        logging.info('read Valid data')
        self.data_valid = Dataset(None,self.vocab,max_length=self.max_length,batch_size=self.batch_size,p_swap=self.swap_bitext,p_uneven=self.uneven_bitext,is_infinite=False,max_sentences_per_file=0)
        for (fs,ft,fa) in opts.train['valid']:
            self.data_valid.add3files(fs,ft,fa)
        logging.info('build Valid batches')
        self.data_valid.build_batches()


    def __call__(self):
        logging.info('Start train n_steps_so_far={}'.format(self.n_steps_so_far))
        ts = stats()
        for batch in self.data_train:
            self.model.train()
            ok, loss, loss_mlm, loss_ali, loss_cos = self.get_loss(batch,ts)
            if not ok:
                continue
            ### gradient computation / model update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.n_steps_so_far += 1
            ###
            ### report
            ###
            if self.report_every_steps > 0 and self.n_steps_so_far % self.report_every_steps == 0:
                ts.report(self.n_steps_so_far,'[Train]')
            ###
            ### save checkpoint
            ###
            if self.checkpoint_every_steps > 0 and self.n_steps_so_far % self.checkpoint_every_steps == 0:
                self.save_checkpoint()
            ###
            ### validation
            ###
            if self.validation_every_steps > 0 and self.n_steps_so_far % self.validation_every_steps == 0:
                vs = stats()
                with torch.no_grad():
                    self.model.eval() ### avoids dropout
                    for batch in self.data_valid:
                        ok, loss, loss_mlm, loss_ali, loss_cos = self.get_loss(batch,vs)
                        if not ok:
                            continue
                    vs.report(self.n_steps_so_far,'[Valid]')
            ###
            ### stop training (never)
            ###
            if self.n_steps_so_far >= 9999999999:
                break

        self.save_checkpoint()
        logging.info('End train')


    def get_loss(self,batch,ds):
        st, st_mlm, st_mlm_ref, st_mask, st_matrix, st_uneven = format_batch(self.vocab, self.cuda, batch, self.step_mlm, self.step_ali, self.step_cos) 
        #st         [bs, ls+lt] contains the original words after concat(x,y)                       [input for ALI]
        #st_matrix  [bs, ls, lt] the alignment between src/tgt (<cls>/<sep> not included)           [reference for ALI]
        #st_mlm     [bs, ls+lt] contains the original words concat(x,y), some masked                [input for MLM]
        #st_mlm_ref [bs, ls+lt] contains the original value of masked words; <pad> for the rest     [reference for MLM]
        #st_mask    [bs, ls+lt] True for x or y words in xy; false for <pad> (<cls>/<sep> included) [mask in MLM and ALI forward step]
        #st_uneven  [bs] 1.0 if uneven, -1.0 if parallel
        #batch.dump()
        #print(st_mlm)
        #print(st_mlm_ref)
        loss = 0.0
        loss_mlm = 0.0
        loss_ali = 0.0
        loss_cos = 0.0
        if self.step_mlm['w'] > 0.0: ### (MLM)
            #st_mlm, st_mlm_ref, st_mask
            #npred_mlm = (st_mlm_ref != self.vocab.idx_pad).sum() ### counts number of elements not <pad> (to be predicted)
            h_st = self.model.forward(st_mlm, st_mask.unsqueeze(-2))
            if torch.isnan(h_st).any():
                logging.error('nan detected in model forward (h_st) [MLM]')
                logging.error('{}'.format(h_st))
                sys.exit()
            batch_loss_mlm, nok_mlm, npred_mlm = self.computeloss_mlm(h_st, st_mlm_ref)
            if npred_mlm == 0: 
                logging.info('batch with nothing to predict')
                return False, loss, loss_mlm, loss_ali, loss_cos
            loss_mlm = batch_loss_mlm / npred_mlm
            ds.add('MLM',loss_mlm,nok_mlm,npred_mlm)
            loss += self.step_mlm['w'] * loss_mlm

        if self.step_ali['w'] > 0.0 or self.step_cos['w'] > 0.0:
            #st, st_mask
            h_st = self.model.forward(st, st_mask.unsqueeze(-2))
            if torch.isnan(h_st).any():
                logging.error('nan detected in model forward (h_st) [ALI/COS]')
                logging.error('{}'.format(h_st))
                sys.exit()
            if self.step_ali['w'] > 0.0: ### (ALI)
                #st_matrix
                #npred_ali = np.dot(batch.lsrc,batch.ltgt)
                batch_loss_ali, nok_ali, npred_ali = self.computeloss_ali(h_st, st_matrix, batch.maxlsrc-1, st_mask)
                loss_ali = batch_loss_ali / npred_ali
                ds.add('ALI',loss_ali,nok_ali,npred_ali)
                loss += self.step_ali['w'] * loss_ali
            if self.step_cos['w'] > 0.0: ### (COS)
                #st_uneven 1.0 if uneven, -1.0 if parallel
                #npred_cos = h_st.shape[0]
                batch_loss_cos, nok_cos, npred_cos = self.computeloss_cos(h_st, st_uneven, batch.maxlsrc-1, st_mask)
                loss_cos = batch_loss_cos / npred_cos
                ds.add('COS',loss_cos,nok_cos,npred_cos)
                loss += self.step_cos['w'] * loss_cos

        ds.add('loss',loss)
#?        torch.cuda.empty_cache()
        return True, loss, loss_mlm, loss_ali, loss_cos


    def load_checkpoint(self):
        files = sorted(glob.glob(self.dir + '/checkpoint.???????.pth')) 
        if len(files):
            file = files[-1] ### last is the newest
            checkpoint = torch.load(file)
            self.n_steps_so_far = checkpoint['n_steps_so_far']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.model.load_state_dict(checkpoint['model'])
            logging.info('loaded checkpoint {}'.format(file))
        else:
            logging.info('no checkpoint available')

            
    def save_checkpoint(self):
        file = '{}/checkpoint.{:07d}.pth'.format(self.dir,self.n_steps_so_far)
        state = {
            'n_steps_so_far': self.n_steps_so_far,
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.state_dict()
        }
        torch.save(state, file)
        logging.info('saved checkpoint {}'.format(file))
        files = sorted(glob.glob(self.dir + '/checkpoint.???????.pth')) 
        while len(files) > self.keep_last_n:
            f = files.pop(0)
            os.remove(f) ### first is the oldest
            logging.debug('removed checkpoint {}'.format(f))


    def average(self):
        files = sorted(glob.glob(self.dir + '/checkpoint.???????.pth')) 
        if len(files) == 0:
            logging.info('no checkpoint available')
            return
        #read models
        models = []
        for file in files:
            m = self.model.clone()
            checkpoint = torch.load(file)
            m.load_state_dict(checkpoint['model'])
            models.append(m)
            logging.info('read {}'.format(file))
        #create mout 
        mout = self.model.clone()
        for ps in zip(*[m.params() for m in [mout] + models]):
            p[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))
        fout = self.dir + '/checkpoint.averaged.pth'
        state = {
            'n_steps_so_far': mout.n_steps_so_far,
            'optimizer': mout.optimizer.state_dict(),
            'model': mout.model.state_dict()
        }
        #save mout into fout
        torch.save(state, fout)
        logging.info('averaged {} models into {}'.format(len(files), fout))


def format_batch(vocab, cuda, batch, step_mlm=None, step_ali=None, step_cos=None):
    ### when steps are None means inference
    st = torch.from_numpy(np.append(batch.sidx, batch.tidx, axis=1))
    #st [batch_size, max_len] contains the original words after concat(x,y) [input for ALI]
    st_mask = torch.as_tensor((st != vocab.idx_pad))
    #st_mask [batch_size, max_len] True for x or y words in st; false for <pad> (<cls>/<sep> included)

    if step_mlm is not None and step_mlm['w'] > 0.0:
        p_mask = step_mlm['p_mask']
        r_same = step_mlm['r_same']
        r_rand = step_mlm['r_rand']

        st_mlm = torch.ones_like(st, dtype=torch.int64) * st
        #st_mlm [batch_size, maxlsrc+maxltgt] contains the original words concat(x,y), some will be masked            [input for MLM]
        st_mlm_ref = torch.ones_like(st, dtype=torch.int64) * vocab.idx_pad 
        #st_mlm_ref [batch_size, maxlsrc+maxltgt] contains the original value of masked words; <pad> for the rest     [reference for MLM]
        for i in range(st.shape[0]):
            for j in range(st.shape[1]):
                if not vocab.is_reserved(st[i,j]):
                    r = random.random()          # float in range [0.0, 1,0)
                    if r < p_mask:               ### masked token that will be predicted
                        st_mlm_ref[i,j] = st[i,j]# save the original value to be used as reference
                        ### replace original token by another. float in range [0.0, 1,0)
                        q = random.random()      # float in range [0.0, 1,0)
                        if q < r_same:           # use the same original token (st[i,j])
                            pass
                        elif q < r_same+r_rand:  # take a random token (among all vocab words in range [7, |vocab|])
                            st_mlm[i,j] = random.randint(7,len(vocab)-1)
                        else:                    # use <msk>
                            st_mlm[i,j] = vocab.idx_msk
    else:
        st_mlm = []
        st_mlm_ref = []


    if step_ali is not None and step_ali['w'] > 0.0:
        st_matrix = (torch.as_tensor(batch.matrix) * -2.0) + 1.0 #-1.0 for aligned pairs; 1.0 for not aligned
        #matrix  [bs,ls,lt] the alignment between src/tgt [reference for ALI]
    else:
        st_matrix = []

    if step_cos is not None and step_cos['w'] > 0.0:
        st_uneven = (torch.as_tensor(batch.parallel) * -2.0) + 1.0 #-1.0 parallel; 1.0 uneven
        #uneven  [bs]
    else:
        st_uneven = []

    if cuda:
        st = st.cuda()
        st_mask = st_mask.cuda()
        if step_mlm is not None and step_mlm['w'] > 0.0:
            st_mlm = st_mlm.cuda()
            st_mlm_ref = st_mlm_ref.cuda()
        if step_ali is not None and step_ali['w'] > 0.0:
            st_matrix = st_matrix.cuda()
        if step_cos is not None and step_cos['w'] > 0.0:
            st_uneven = st_uneven.cuda()

    return st, st_mlm, st_mlm_ref, st_mask, st_matrix, st_uneven




