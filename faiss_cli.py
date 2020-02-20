# -*- coding: utf-8 -*-
import logging
import sys
import io
import gzip
import faiss
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from faiss import normalize_L2

class Infile:

    def __init__(self, file, d, norm=True,file_str=None):
        self.vec = []
        self.txt = []

        if file.endswith('.gz'): 
            f = gzip.open(file, 'rt')
        else:
            f = io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')

        for l in f:
            l = l.rstrip().split(' ')
            if len(l) != d:
                logging.error('found {} floats instead of {}'.format(len(l),d))
                sys.exit()
            self.vec.append(l)

        self.vec = np.array(self.vec).astype('float32')
        if norm:
            faiss.normalize_L2(self.vec)

        if file_str is None:
            return

        if file_str.endswith('.gz'): 
            f = gzip.open(file_str, 'rt')
        else:
            f = io.open(file_str, 'r', encoding='utf-8', newline='\n', errors='ignore')

        for l in f:
            self.txt.append(l.rstrip())

        if len(self.txt) != len(self.vec):
            logging.error('diff num of entries {} <> {} in files {} and {}'.format(len(self.vec),len(self.txt),file, file_str))
            sys.exit()

    def __len__(self):
        return len(self.vec)

    def txts(self):
        return len(self.txt)>0


def results(D,I,k,db,query,verbose):
    n_ok = [0.0] * k
    for i in range(len(I)):
        ### Accuracy
        for j in range(k):
            if i in I[i,0:j+1]:
                n_ok[j] += 1.0
        ### output
        out = []
        if verbose:
            out.append(str(i))
            if query.txts():
                out[-1] += " {}".format(query.txt[i])
            for j in range(len(I[i])):
                out.append("{}:{:.4f}".format(I[i,j],D[i,j]))
                if db.txts():
                    out[-1] += " {}".format(db.txt[I[i,j]])
            print('\n\t'.join(out))
        else:
            out.append(str(i))
            out.append("{} {}".format(I[i],D[i]))
            if query.txts():
                out.append(query.txt[i])
            if db.txts():
                out.append(db.txt[I[i,0]])
            print('\t'.join(out))

    n_ok = ["{:.3f}".format(n/len(query)) for n in n_ok]
    print('Done k-best Acc = [{}] over {} examples'.format(', '.join(n_ok),len(query)))


class IndexFaiss:

    def __init__(self, file, d, file_str=None):
        self.db = Infile(file, d, norm=True, file_str=file_str)
        self.index = faiss.IndexFlatIP(d) #inner product (needs L2 normalization over db and query vectors)
        self.index.add(self.db.vec) # add all normalized vectors to the index
        logging.info("read {} vectors".format(self.index.ntotal))


    def Query(self,file,d,k,file_str,verbose):
        query = Infile(file, d, norm=True, file_str=file_str)
        D, I = self.index.search(query.vec, k)
        results(D,I,k,self.db,query,verbose)

'''
class Index:

    def __init__(self, file, d, file_str=None):
        self.db = Infile(file, d, norm=True, file_str=file_str)
        logging.info("read {} vectors".format(len(self.db)))


    def Query(self,file,d,k,file_str,verbose):
        #cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        query = Infile(file, d, norm=True, file_str=file_str)
        db = torch.from_numpy(self.db.vec)
        D = []
        I = []
        for i in range(len(query)):
            q =  torch.from_numpy(query.vec[i]).unsqueeze(0)
            dist = F.cosine_similarity(db,q)
            dist_sorted, index_sorted = torch.sort(dist, descending=True)
            D.append(dist_sorted[:k].numpy())
            I.append(index_sorted[:k].numpy())
        results(np.array(D),np.array(I),k,self.db,query,verbose)
'''

if __name__ == '__main__':

    fdb = None
    fquery = None
    fdb_str = None
    fquery_str = None
    d = 512
    k = 1
    verbose = False
    name = sys.argv.pop(0)
    usage = '''usage: {} -db FILE -query FILE [-db_str FILE] [-query_str FILE] [-d INT] [-k INT] [-v]
    -db        FILE : file to index 
    -db_str    FILE : file to index 
    -query     FILE : file with queries
    -query_str FILE : file with queries
    -d          INT : vector size (default 512)
    -k          INT : k-best to retrieve (default 1)
    -v              : verbose output (default False)
    -h              : this help
'''.format(name)

    while len(sys.argv):
        tok = sys.argv.pop(0)
        if tok=="-h":
            sys.stderr.write("{}".format(usage))
            sys.exit()
        elif tok=="-v":
            verbose = True
        elif tok=="-db" and len(sys.argv):
            fdb = sys.argv.pop(0)
        elif tok=="-db_str" and len(sys.argv):
            fdb_str = sys.argv.pop(0)
        elif tok=="-query" and len(sys.argv):
            fquery = sys.argv.pop(0)
        elif tok=="-query_str" and len(sys.argv):
            fquery_str = sys.argv.pop(0)
        elif tok=="-k" and len(sys.argv):
            k = int(sys.argv.pop(0))
        elif tok=="-d" and len(sys.argv):
            d = int(sys.argv.pop(0))
        elif tok=="-query_is_db":
            query_is_db = True
        else:
            sys.stderr.write('error: unparsed {} option\n'.format(tok))
            sys.stderr.write("{}".format(usage))
            sys.exit()


    if fdb is not None:
        indexdb = IndexFaiss(fdb,d,fdb_str)


    if fquery is not None:
        indexdb.Query(fquery,d,k,fquery_str,verbose)



