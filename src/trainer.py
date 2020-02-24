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
from src.optim import NoamOpt, LabelSmoothing, Align, ComputeLossMLM, ComputeLossALI


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
        self.n_preds = 0
        self.sum_loss = 0.0
        self.start = time.time()

    def add_batch(self,batch_loss,n_predicted):
        self.sum_loss += batch_loss
        self.n_preds += n_predicted

    def report(self,n_steps,step,trn_val_tst,cuda):
        if False and cuda:
            torch.cuda.empty_cache()
            device = torch.cuda.current_device()
            Gb_reserved = torch.cuda.memory_reserved(device=device) / 1073741824
            Gb_used = torch.cuda.get_device_properties(device=device).total_memory / 1073741824

        logging.info("{} step: {} ({}) Loss: {:.6f} Pred/sec: {:.1f}".format(trn_val_tst, n_steps, step, self.sum_loss/self.n_preds, self.n_preds/(time.time()-self.start)))
        #logging.info('{}'.format(torch.cuda.memory_summary(device=device, abbreviated=False)))
        self.n_preds = 0
        self.sum_loss = 0.0
        self.start = time.time()


class Trainer():

    def __init__(self, opts):
        self.dir = opts.dir
        self.vocab = Vocab(opts.dir+'/vocab')
        self.n_steps_so_far = 0
        self.report_every_steps = opts.train['report_every_steps']
        self.validation_every_steps = opts.train['validation_every_steps']
        self.checkpoint_every_steps = opts.train['checkpoint_every_steps']
        self.batch_size = opts.train['batch_size']
        self.max_length = opts.train['max_length']
        self.swap_bitext = opts.train['swap_bitext'] 
        self.step_mlm = opts.train['steps']['mlm']
        self.step_ali = opts.train['steps']['ali']


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

        if self.cuda:
            self.crit_mlm.cuda()
            self.crit_ali.cuda()

        self.load_checkpoint() #loads if exists
        self.computeloss_mlm = ComputeLossMLM(self.model.generator, self.crit_mlm, self.optimizer)
        self.computeloss_ali = ComputeLossALI(self.crit_ali, self.step_ali, self.optimizer)

        logging.info('read Train data')
        self.data_train = Dataset(None,self.vocab,max_length=self.max_length,is_infinite=True)
        for (fs,ft,fa) in opts.train['train']:
            self.data_train.add3files(fs,ft,fa)
        self.data_train.build_batches(self.batch_size[0],self.swap_bitext)

        logging.info('read Valid data')
        self.data_valid = Dataset(None,self.vocab,max_length=self.max_length,is_infinite=False)
        for (fs,ft,fa) in opts.train['valid']:
            self.data_valid.add3files(fs,ft,fa)
        self.data_valid.build_batches(self.batch_size[1],self.swap_bitext)


    def __call__(self):

        logging.info('Start train n_steps_so_far={}'.format(self.n_steps_so_far))
        ts = stats()
        for batch in self.data_train:
            self.model.train()
            xy, xy_mask, xy_refs, mask_xy, mask_x, mask_y, matrix, npred_mlm, npred_ali = self.format_batch(batch, self.step_mlm, self.step_ali) 
            #xy      [batch_size, ls+lt] contains the original words after concat(x,y)                          [input for ALI]
            #xy_mask [batch_size, ls+lt] contains the original words concat(x,y), some are be masked            [input for MLM]
            #xy_refs [batch_size, ls+lt] contains the original value of masked words; <pad> for the rest        [reference for MLM]
            #matrix  [bs,ls-1,lt-1] the alignment between src/tgt (<cls>/<sep> not included)                    [reference for ALI]
            #mask_xy [batch_size, ls+lt] True for x or y words in xy; false for <pad> (<cls>/<sep> included)
            #mask_x  [batch_size, ls+lt] True for x words in xy; false for rest (<cls> not included)
            #mask_y  [batch_size, ls+lt] True for y words in xy; false for rest (<sep> not included)
            loss = 0.0
            if self.step_mlm['w'] > 0.0: ### (MLM)
                h_xy = self.model.forward(xy_mask, mask_xy.unsqueeze(-2))
                if npred_mlm == 0: 
                    logging.info('batch with nothing to predict')
                    continue
                batch_loss_mlm = self.computeloss_mlm(h_xy, xy_refs)
                loss_mlm = batch_loss_mlm / npred_mlm
                loss += self.step_mlm['w'] * loss_mlm
                print(loss_mlm)

            if self.step_ali['w'] > 0.0 and False: ### (ALI)
                h_xy = self.model.forward(xy, mask_xy)
                h_x = h_xy[:,:maxslen,:]
                h_y = h_xy[:,maxslen:,:]
                batch_loss_ali = self.computeloss_ali(h_xy, matrix, mask_x, mask_y)
                loss_ali = batch_loss_ali / npred_ali
                loss += self.step_ali['w'] * loss_ali

            ### gradient computation / model update
            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step()

            self.n_steps_so_far += 1
            #ts.add_batch(loss,batch_loss_mlm,npred_mlm,batch_loss_ali,npred_ali)
            ###
            ### report
            ###
            if self.report_every_steps > 0 and self.n_steps_so_far % self.report_every_steps == 0:
                ts.report(self.n_steps_so_far,step,'[Train]',self.cuda)

            ###
            ### saved
            ###
            if self.checkpoint_every_steps > 0 and self.n_steps_so_far % self.checkpoint_every_steps == 0:
                self.save_checkpoint()
            ###
            ### validation
            ###
            if self.validation_every_steps > 0 and self.n_steps_so_far % self.validation_every_steps == 0:
                self.validation()
            ###
            ### stop training (never)
            ###
            if self.n_steps_so_far >= 9999999999:
                break

        self.save_checkpoint()
        logging.info('End train')

    def validation(self):
        ds = stats()
        with torch.no_grad():
            self.model.eval() ### avoids dropout
            for batch in self.data_valid:
                if not self.sim_run: ### pre-training (MLM)
                    step = 'mlm'
                    x, x_mask, y_mask = self.mlm_batch_cuda(batch)
                    n_predictions = torch.sum((y_mask != self.vocab.idx_pad)).data
                    if n_predictions == 0: #nothing to predict
                        logging.info('batch with nothing to predict')
                        continue
                    h = self.model.forward(x,x_mask)
                    batch_loss = self.computeloss(h, y_mask)
                else: ### fine-tunning (SIM)
                    step = 'sim'
                    x1, x2, l1, l2, x1_mask, x2_mask, y, mask_s, mask_t = self.sim_batch_cuda(batch) 
                    n_predictions = x1.size(0)
                    h1 = self.model.forward(x1,x1_mask)
                    h2 = self.model.forward(x2,x2_mask)
                    batch_loss = self.computeloss(h1, h2, l1, l2, y, mask_s, mask_t)
                ds.add_batch(batch_loss,n_predictions)
            ds.report(self.n_steps_so_far,step,'[Valid]',self.cuda)

    def format_batch(self, batch, step_mlm, step_ali):
        xy = torch.from_numpy(np.append(batch.sidx, batch.tidx, axis=1))
        #xy [batch_size, max_len] contains the original words after concat(x,y) [input for ALI]
        npred_mlm = 0
        npred_ali = 0
        if step_mlm['w'] > 0.0:
            p_mask = step_mlm['p_mask']
            r_same = step_mlm['r_same']
            r_rand = step_mlm['r_rand']

            xy_mask = torch.ones_like(xy, dtype=torch.int64) * xy                 #[batch_size, max_len] contains the original words concat(x,y). some will be masked
            #xy_mask [batch_size, max_len] contains the original words concat(x,y), some will be masked            [input for MLM]
            xy_refs = torch.ones_like(xy, dtype=torch.int64) * self.vocab.idx_pad #[batch_size, max_len] will contain the original value of masked words in xy <pad> for the rest
            #xy_refs [batch_size, max_len] contains the original value of masked words; <pad> for the rest         [reference for MLM]
            mask_xy = torch.as_tensor((xy != self.vocab.idx_pad))                 #[batch_size, max_len] contains true for words, False for <pad>
            #mask_xy [batch_size, max_len] True for x or y words in xy; false for <pad> (<cls>/<sep> included)
            for i in range(xy.shape[0]):
                for j in range(xy.shape[1]):
                    if not self.vocab.is_reserved(xy[i,j]):
                        r = random.random()          # float in range [0.0, 1,0)
                        if r < p_mask:               ### masked token that will be predicted
                            npred_mlm += 1
                            xy_refs[i,j] = xy[i,j]   # save the original value to be used as reference
                            q = random.random()      # float in range [0.0, 1,0)
                            if q < r_same:           ### same
                                pass
                            elif q < r_same+r_rand:  ### random (among all vocab words in range [7, |vocab|])
                                xy_mask[i,j] = random.randint(7,len(self.vocab)-1)
                            else:                    # <msk>
                                xy_mask[i,j] = self.vocab.idx_msk
        else:
            xy_mask = []
            xy_refs = []
            mask_xy = []


        if step_ali['w'] > 0.0:
            matrix = torch.as_tensor(batch.ali)
            #matrix  [bs,ls,lt] the alignment between src/tgt [reference for ALI]
            mask_x = torch.zeros_like(xy, dtype=torch.bool)
            #mask_x  [batch_size, max_len] True for x words in xy; false for rest (<cls> not included)
            mask_y = torch.zeros_like(xy, dtype=torch.bool)
            #mask_y  [batch_size, max_len] True for y words in xy; false for rest (<sep> not included)
            for b in range(mask_x.shape[0]):
                for i in range(1,batch.lsrc[b]): ### do not include <cls>
                    mask_x[b,i] = True
                for j in range(1,batch.ltgt[b]): ### do not include <sep>
                    mask_y[b,batch.maxlsrc+j] = True
            npred_ali += matrix.numel()
        else:
            matrix = []
            mask_x = []
            mask_y = []

        if self.cuda:
            xy = xy.cuda()
            if step_mlm['w'] > 0.0:
                xy_mask = xy_mask.cuda()
                xy_refs = xy_refs.cuda()
                mask_xy = mask_xy.cuda()
            if step_ali['w'] > 0.0:
                mask_x = mask_x.cuda()
                mask_y = mask_y.cuda()
                matrix = matrix.cuda()

        return xy, xy_mask, xy_refs, mask_xy, mask_x, mask_y, matrix, npred_mlm, npred_ali


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
        while len(files) > self.average_last_n:
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





