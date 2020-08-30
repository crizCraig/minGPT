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


class Text8Dataset(Dataset):
    """
    e.g. Text8 dataset is often used: http://mattmahoney.net/dc/textdata.html
    Vocabulary is lowercase English characters and space for total of 27.
    Training data: First 90M characters.
    Validation data: First 5M characters out of the last 10M characters.
    Testing data: Last 5M characters.
    """

    def __init__(self, data_path, block_size, crop=None, override_vocab=None):

        # load the data and crop it appropriately
        with open(data_path, 'r') as f:
            if crop is None:
                data = f.read()
            else:
                f.seek(crop[0])
                data = f.read(crop[1])

        # build a vocabulary from data or inherit it
        vocab = sorted(list(set(data))) if override_vocab is None else override_vocab
        data_size, vocab_size = len(data), len(vocab)
        log.info('data of crop %s has %d characters, vocab of size %d.' % (str(crop), data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(vocab) }
        self.itos = { i:ch for i,ch in enumerate(vocab) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        # attempt to fetch a chunk of (block_size + 1) items, but (block_size) will work too
        chunk = self.data[idx*self.block_size : min(len(self.data), (idx+1)*self.block_size + 1)]
        # map the string into a sequence of integers
        ixes = [self.stoi[s] for s in chunk]
        # if stars align (last idx and len(self.data) % self.block_size == 0), pad with -100, to skip training at the last position
        if len(ixes) < self.block_size + 1:
            assert len(ixes) == self.block_size # i believe this is the only way this could happen, make sure
            ixes.append(-100)
        dix = torch.tensor(ixes, dtype=torch.long)
        return dix[:-1], dix[1:]


class CharDataset(Dataset):

    def __init__(self, data, block_size, percent_random=0, all_same=False):  # block_size=128
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)  # vocab_size=65, data_size=1115390
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.percent_random = percent_random
        self.all_same = all_same

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size
        random_chunk = []
        if self.percent_random:
            random_size = block_size * self.percent_random // 100
            block_size = block_size - random_size
            random_chunk = np.random.randint(low=0, high=self.vocab_size, size=(random_size,)).tolist()

        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk] + random_chunk
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next

        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during training will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can parallelize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.

        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """

        if self.all_same:
            dix = [1] * len(dix)

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--num-epochs', type=int, default=2, help="number of epochs to train for")
    parser.add_argument('-b', '--batch-size', type=int, default=64, help="batch size to train with")
    parser.add_argument('-l', '--block-size', type=int, default=128,
                        help="block size for the model (length of window of context)")
    parser.add_argument('-g', '--num-gpus', type=int, default=1, help="number of gpus to train on")
    parser.add_argument('-n', '--num-workers', type=int, default=0, help="number of workers for dataloading")
    parser.add_argument('-p', '--pin-memory', type=int, default=0, help="pin memory on dataloaders?")
    parser.add_argument('-r', '--precision', type=int, default=32, help="fp precision to use, e.g. 32/16")
    parser.add_argument('-o', '--default_root_dir', type=str, default=SAVE_DIR,
                        help="best model checkpoint will be written at this location")
    args = parser.parse_args()
    log.info(vars(args))

    torch.backends.cudnn.benchmark = True  # autotune kernels

    log.info("preparing the data loaders")
    # NOTE: REDUCED DATA SIZE FOR DEBUGGING, TODO CLEAN BEFORE MERGE IF EVER
    # train_dataset = Text8Dataset('text8', args.block_size, crop=(0,         int(90e6)))
    # val_dataset   = Text8Dataset('text8', args.block_size, crop=(int(90e6), int(5e6)), override_vocab=train_dataset.vocab)
    # test_dataset  = Text8Dataset('text8', args.block_size, crop=(int(95e6), int(5e6)), override_vocab=train_dataset.vocab)

    train_dataset = CharDataset(open('train_shakespeare.txt', 'r').read(), args.block_size)  # one line of poem is roughly 50 characters
    val_dataset = CharDataset(open('val_shakespeare.txt', 'r').read(), args.block_size)
    test_dataset = CharDataset(open('test_shakespeare.txt', 'r').read(), args.block_size)

    common = {'batch_size': args.batch_size, 'pin_memory': bool(args.pin_memory), 'num_workers': args.num_workers}
    train_dataloader = DataLoader(train_dataset, shuffle=True, **common)
    val_dataloader = DataLoader(val_dataset, shuffle=False, **common)

    log.info("creating the model")
    model = GPT(train_dataset.vocab_size, args.block_size, n_layer=8, n_head=8, n_embd=256)

    log.info("preparing the learning rate schedule")
    iter_tokens = args.batch_size * args.block_size  # number of tokens backpropped in one iteration
    epoch_tokens = math.ceil(len(train_dataset) / args.batch_size) * iter_tokens
    lr_decay = WarmupCosineLearningRateDecay(learning_rate=6e-4,
                                             warmup_tokens=512 * 20,  # epoch_tokens // 2,
                                             final_tokens=args.num_epochs * epoch_tokens)

    t0 = time.time()
    log.info("training...")
    trainer = pl.Trainer(gpus=args.num_gpus, max_epochs=args.num_epochs, gradient_clip_val=1.0, callbacks=[lr_decay],
                         precision=args.precision, default_root_dir=args.default_root_dir)
    trainer.fit(model, train_dataloader, val_dataloader)
    t1 = time.time()
    log.info("%d epochs took %fs, or %fs/epoch" % (args.num_epochs, t1 - t0, (t1 - t0) / args.num_epochs))

    log.info("testing...")
    test_dataloader = DataLoader(test_dataset, shuffle=False, **common)
    trainer.test(test_dataloaders=test_dataloader)

    log.info("sampling:")
    # context = "anarchism originated as a term of"
    context = "O God, O God!"
    x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None, ...]
    if next(model.parameters()).is_cuda:
        x = x.cuda()
    y = sample(model, x, 200, temperature=1.0, sample=True, top_k=None)[0]
    completion = ''.join([train_dataset.itos[int(i)] for i in y])
    log.info(completion)


if __name__ == '__main__':
    main()
