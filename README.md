# sentalign

Sentence Alignment through Transformer

Prerequisites:
```
pip3 install torch torchvision
pip3 install pyonmttok
pip3 install pyyaml
```

Corpora:
```
```

## Usage:
```
usage: sentalign.py -dir DIR [-learn YAML] [-infer YAML] [-config YAML] [-seed INT] [-log FILE] [-loglevel LEVEL]
   -dir        DIR : checkpoint directory (must not exist when learning from scratch)
   -infer     YAML : test config file (inference mode)
   -learn     YAML : train config file (learning mode)
   -config    YAML : modeling/optim config file (needed when learning from scratch)

   -seed       INT : seed value (default 12345)
   -log       FILE : log file (default stderr)
   -loglevel LEVEL : use 'debug', 'info', 'warning', 'critical' or 'error' (default info)
   -h              : this help

* The script needs pyonmttok installed (pip install pyonmttok)
* Use -learn YAML (or -infer YAML) for learning (or inference) modes
* When learning from scratch:
  + The directory -dir DIR is created
  + source and target vocabs/bpe files are copied to DIR (cannot be further modified)
  + config file -config YAML is copied to DIR (cannot be further modified)
  ```

Learn example:
```
CUDA_VISIBLE_DEVICES=0 python3 sentalign.py -dir exp1 -learn cfg/train.yml -model cfg/model.yml -optim cfg/optim.yml
```

cat cfg/model.yml
```
tokenization:
	src: { bpe_model_path: ./data/joint_enfr.30k.bpe, mode: conservative, joiner_annotate: true }
	tgt: { bpe_model_path: ./data/joint_enfr.30k.bpe, mode: conservative, joiner_annotate: true }
vocab: ./data/en.vocab
example_format:
	src: { cls: true, bos: true, eos: true }
        sep: True
	tgt: { cls: false, bos: true, eos: true }
num_layers: 6
hidden_size: 512
feedforward_size: 2048
num_heads: 8
emb_size: 512
dropout: 0.1
cuda: true
```

cat cfg/optim.yml
```
learning_rate: 0.0001
beta1: 0.9
beta2: 0.98
eps: 0.000000001
warmup_steps: 10000
factor: 1.0
smoothing: 0.1
```

cat cfg/train.yml
```
valid:
	src: [ ./data/ECB.val.en.gz, ./data/EMEA.val.en.gz, ./data/KDE4.val.en.gz ]
   	tgt: [ ./data/ECB.val.fr.gz, ./data/EMEA.val.fr.gz, ./data/KDE4.val.fr.gz ]
train:
	src: [ ./data/ECB.trn.en.gz, ./data/EMEA.trn.en.gz, ./data/KDE4.trn.en.gz ]
        tgt: [ ./data/ECB.trn.fr.gz, ./data/EMEA.trn.fr.gz, ./data/KDE4.trn.fr.gz ]
max_length: 160
batch_size: 32
train_steps: 1000000
para_step: { prob: 0.15, same: 0.1, rnd: 0.1 }
tran_step: { prob: 0.15, same: 0.1, rnd: 0.1 }
mono_step: { prob: 0.15, same: 0.1, rnd: 0.1 }
checkpoint_every_steps: 1000
validation_every_steps: 5000
report_every_steps: 100
```