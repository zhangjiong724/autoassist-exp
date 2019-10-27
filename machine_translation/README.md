# AutoAssist Pytorch NMT Implementation
Code for machine translation accompanying the NeurIPS 2019 paper [AutoAssist: A Framework to Accelerate Training of Deep Neural Networks](https://arxiv.org/pdf/1905.03381.pdf).
This folder contains code for machine translation experiments. The code is modified from the Facebook [fairseq](https://github.com/pytorch/fairseq) repository.

# Prerequisites
    - Python (v3.7)
    - PyTorch (v1.0.1)
    - h5py, numpy, scipy

## Data preprocessing 

```
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
--trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
--destdir data-bin/iwslt14.tokenized.de-en
```

## Quick Start
For a quick start, please execute ```scripts/run_*```:
```
    $ bash ./scripts/run_base # for the baseline SGD model
    $ bash ./scripts/run_spl # for the self paced learning model
    $ bash ./scripts/run_autoassist # for the AutoAssist model
```


## Training models

As an example, to train transformer model on iwslt14 dataset with 4 GPUs:
```
python train.py data-bin/iwslt14.tokenized.de-en \
								--arch transformer_iwslt_de_en\
                                --optimizer adam --lr 0.0005  \
                                --label-smoothing 0.1 --dropout 0.1 --max-tokens 3000 \
                                --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
                                --criterion label_smoothed_cross_entropy --max-update 300000 --max-epoch 100\
                                --warmup-updates 4000 --warmup-init-lr '1e-07' \
                                --adam-betas '(0.9, 0.98)' \
                                --distributed-world-size 4 \
                                --log-format simple --log-interval 10 
```


## Translation 
To translate with saved checkpoint "ckpt.pt":

```
python generate.py data-bin/iwslt14.tokenized.de-en  \
  --path ckpt.pt \
  --beam 5 --batch-size 128 --remove-bpe 
```

## Evaluation 
To evaluate trained model:

```
bash ./scripts/eval_BLEU [checkpoint_dir] [data_dir]
```

