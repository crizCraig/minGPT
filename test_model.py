"""
Temporary benchmarking script while integrating Lightning, will remove before merge to master
"""

import os
import time
import math
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.backends.cudnn as cudnn

from loguru import logger as log

from bench import CharDataset
from mingpt.constants import ROOT_DIR, SAVE_DIR

from mingpt.model import GPT
from mingpt.lr_decay import WarmupCosineLearningRateDecay
from mingpt.utils import sample

# -----------------------------------------------------------------------------
if int(os.environ.get('USE_LIGHTNING', 0)):
    import pytorch_lightning as pl
else:
    import mingpt.fake_lightning as pl
# -----------------------------------------------------------------------------

log.add(ROOT_DIR + "/logs/{time}.log")
BLOCK_SIZE = 128

def main():
    torch.backends.cudnn.benchmark = True  # autotune kernels
    log.info("sampling:")
    # context = "anarchism originated as a term of"
    # context = "O God, O God!"
    input_prompt = "Romeo and Julie"
    train_dataset = CharDataset(open('train_shakespeare.txt', 'r').read(), 128)  # one line of poem is roughly 50 characters
    model = GPT.load_from_checkpoint('/home/c2/src/minGPT/checkpoints/lightning_logs/version_3/checkpoints/epoch=1-step=29033.ckpt',
                                     vocab_size=65, block_size=128, n_layer=8, n_head=8, n_embd=256)
    x = torch.tensor([train_dataset.stoi[s] for s in input_prompt], dtype=torch.long)[None, ...]
    if next(model.parameters()).is_cuda:
        x = x.cuda()
    y = sample(model, x, 200, temperature=2.0, sample=True, top_k=None)[0]
    completion = ''.join([train_dataset.itos[int(i)] for i in y])
    log.info(completion)


if __name__ == '__main__':
    main()
