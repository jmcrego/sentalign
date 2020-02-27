import logging
import yaml
import sys
import os
import random
import torch
from shutil import copyfile
from src.tools import create_logger
from src.dataset import Vocab, Dataset, OpenNMTTokenizer
from src.model import make_model
from src.trainer import Trainer
from src.infer import Infer


class Argv():

    def __init__(self, argv):
        self.prog = argv.pop(0)
        self.usage = '''usage: {} -dir DIR [-learn YAML] [-infer YAML] [-pooling STRING] [-config YAML] [-seed INT] [-log FILE] [-loglevel LEVEL]
   -dir        DIR : checkpoint directory (must not exist when learning from scratch)

 Learning:
   -learn     YAML : train config file (learning mode)
   -config    YAML : modeling/optim config file (needed when learning from scratch)

 Inference:
   -infer     YAML : test config file (inference mode)
   -pooling STRING : inference pooling method, use 'max', 'mean' or 'cls' (default mean)
   -batch_size INT : batch size use on inference (default 32)
   -scale    FLOAT : alignment matrix scale (default 0.001)
   -matrix         : show laignment matrix

 Average:
   -average        : average models in DIR

   -seed       INT : seed value (default 12345)
   -log       FILE : log file (default stderr)
   -loglevel LEVEL : use 'debug', 'info', 'warning', 'critical' or 'error' (default info) 
   -h              : this help

* The script needs:
  + pytorch:   conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
  + pyonmttok: pip install pyonmttok
  + pyyaml:    pip install PyYAML
* Use -learn YAML (or -infer YAML) for learning (or inference) modes
* When learning from scratch:
  + The directory -dir DIR is created
  + source and target vocabs/bpe files are copied to DIR (cannot be further modified)
  + config file -config YAML is copied to DIR (cannot be further modified)
'''.format(self.prog)
        self.fcfg = None
        self.flog = None
        self.log_level = 'info'
        self.dir = None
        self.flearn = None
        self.finfer = None
        self.seed = 12345
        self.pooling = 'mean'
        self.matrix = False
        self.batch_size = 32
        self.align_scale = 0.001
        while len(argv):
            tok = argv.pop(0)
            if   (tok=="-config"   and len(argv)): self.fcfg = argv.pop(0)
            elif (tok=="-learn"    and len(argv)): self.flearn = argv.pop(0)
            elif (tok=="-log"      and len(argv)): self.flog = argv.pop(0)
            elif (tok=="-loglevel" and len(argv)): self.log_level = argv.pop(0)
            elif (tok=="-dir"      and len(argv)): self.dir = argv.pop(0)
            elif (tok=="-infer"    and len(argv)): self.finfer = argv.pop(0)
            elif (tok=="-pooling"  and len(argv)): self.pooling = argv.pop(0)
            elif (tok=="-batch_size" and len(argv)): self.batch_size = int(argv.pop(0))
            elif (tok=="-scale"    and len(argv)): self.align_scale = float(argv.pop(0))
            elif (tok=="-matrix"):                 self.matrix = True
            elif (tok=="-seed"     and len(argv)): self.seed = int(argv.pop(0))
            elif (tok=="-h"):
                sys.stderr.write("{}".format(self.usage))
                sys.exit()
            else:
                sys.stderr.write('error: unparsed {} option\n'.format(tok))
                sys.stderr.write("{}".format(self.usage))
                sys.exit()

        create_logger(self.flog,self.log_level)
                
        if self.dir is None:
            logging.error('error: missing -dir option')
            #sys.stderr.write("{}".format(self.usage))
            sys.exit()

        if self.dir.endswith('/'): 
            self.dir = self.dir[:-1]
        self.dir = os.path.abspath(self.dir)
        logging.info('dir={}'.format(self.dir))

        if self.finfer is not None and self.flearn is not None:
            logging.warning('-learn FILE not used on inference mode')

        if self.finfer is None and self.flearn is None:
            logging.error('either -learn or -infer options must be used')
            sys.exit()

        if self.finfer is not None and not os.path.exists(self.dir):
            logging.error('running inference with empty model dir')
            sys.exit()

        if os.path.exists(self.dir):
            if self.fcfg is not None:
                logging.warning('-config FILE not used ({}/config.yml)'.format(self.dir))
            self.fcfg = self.dir + "/config.yml"
        else:
            if self.fcfg is None:
                logging.error('missing -config option')
                sys.exit()


def create_experiment(opts):
    with open(opts.fcfg) as file:
        mycfg = yaml.load(file, Loader=yaml.FullLoader)
        logging.debug('Read config : {}'.format(mycfg))
    os.mkdir(opts.dir)
    ### copy vocab/config.yml files to opts.dir
    copyfile(mycfg['vocab'], opts.dir+"/vocab")
    logging.info('copied {} => {}'.format(mycfg['vocab'], opts.dir+"/vocab"))
    #delete vocab from config file (it is always dir/vocab)
    del mycfg['vocab']
    ### dump mycfg into opts.dir/config.yml
    with open(opts.dir+"/config.yml", 'w') as file:
        _ = yaml.dump(mycfg, file)
    opts.fcfg = opts.dir+"/config.yml"
    logging.info('Created experiment: {}'.format(opts.dir))


if __name__ == "__main__":
    
    #create_logger(None,'debug')
    #cfg_token = { 'bpe_model_path': './data/joint_enfr.30k.bpe', 'mode': 'conservative', 'joiner_annotate': False }
    #cfg_token = { 'mode': 'conservative', 'joiner_annotate': False }
    #token = OpenNMTTokenizer(**cfg_token)
    #token = None
    #vocab = Vocab('./data/vocab')
    #data = Dataset(token,vocab,max_length=0,is_infinite=False)
    #data.add3files('./data/clean.news-commentary-v14.en.trn.tokc','./data/clean.news-commentary-v14.fr.trn.tokc','./data/clean.news-commentary-v14.en.trn.tokc.en2fr.gdfa')
    #data.add3files('./data/clean.GNOME.en-fr.en.trn.tokc','./data/clean.GNOME.en-fr.fr.trn.tokc','./data/clean.GNOME.en-fr.en.trn.tokc.en2fr.gdfa')
    #data.build_batches(4)
    #sys.exit()

    opts = Argv(sys.argv)

    if opts.seed > 0:
        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts.seed)
        logging.debug('random.seed set to {}'.format(opts.seed))

    if not os.path.exists(opts.dir):
        create_experiment(opts)

    with open(opts.dir+"/config.yml") as file:
        opts.cfg = yaml.load(file, Loader=yaml.FullLoader)
        logging.debug('Read config : {}'.format(opts.cfg))

    if opts.finfer is not None:
        infer = Infer(opts)
        infer(opts.finfer)
    else:
        with open(opts.flearn) as file:
            opts.train = yaml.load(file, Loader=yaml.FullLoader)
            logging.debug('Read config for learning : {}'.format(opts.train))
        trainer = Trainer(opts)
        trainer()

    logging.info('Done!')





