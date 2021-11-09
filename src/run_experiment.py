import argparse
import os
import pickle
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Tuple

import h5py as h5
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from tqdm import tqdm

from data import ComicsDataset, ComicPanelBatch
from models.lstm import TextOnlyHeirarchicalLSTM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_fn(batches_and_labels: List[List[Tuple[ComicPanelBatch, torch.Tensor]]]):
    """
    Dummy collate function to just return the single element from the list (which will
    already be a list of batches), since our dataset does the batching logic.
    """
    assert len(batches_and_labels) == 1
    return batches_and_labels[0]


def train_one_epoch(
    model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer
):
    model.train()

    total_loss = 0
    n_examples = 0

    # NOTE: It's hard to get tqdm working for the training page/batch count, mainly
    # because we don't know how many examples total there are, and how many pages
    # and panels will be valid.
    for batches in dataloader:
        for batch, labels in batches:
            batch.to(device)
            labels = labels.to(device)

            logits = model(batch)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item()
            n_examples += batch.batch_size

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    avg_loss = total_loss / n_examples
    return avg_loss


def eval_one_epoch(model: nn.Module, dataloader: DataLoader):
    model.eval()

    with torch.no_grad():
        total_loss = 0
        n_examples = 0
        all_preds = []
        all_labels = []
        for batches in dataloader:
            for batch, labels in batches:
                batch.to(device)
                labels = labels.to(device)
                
                logits = model(batch)
                loss = F.cross_entropy(logits, labels)

                preds = torch.argmax(logits, dim=-1)
                all_preds.append(preds)
                all_labels.append(labels)

                total_loss += loss.item()
                n_examples += batch.batch_size

        avg_loss = total_loss / n_examples

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc = (torch.sum(all_preds == all_labels) / n_examples).item()

        return avg_loss, acc, all_preds, all_labels


def make_dataloader(
    comics_data_path: str,
    vgg_feats_path: str,
    vocab_path: str,
    folds_dir: str,
    fold: str,
    difficulty: str,
    megabatch_size: int,
    batch_size: int,
    num_workers: int,
):
    dataset = ComicsDataset(
        comics_data_path=comics_data_path,
        vgg_feats_path=vgg_feats_path,
        vocab_path=vocab_path,
        folds_dir=folds_dir,
        difficulty=difficulty,
        fold=fold,
        batch_size=batch_size,
    )

    # We use SequentialSampler because the original code did not shuffle example order,
    # and we use BatchSampler to pass multiple indices to our dataset.
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        sampler=BatchSampler(
            SequentialSampler(dataset), batch_size=megabatch_size, drop_last=False
        ),
        num_workers=num_workers,
    )

    return dataloader, dataset


def main(
    comics_data_path: str = '../comics/data/comics.h5',
    vgg_feats_path: str = '../comics/data/vgg_features.h5',
    vocab_path: str = '../comics/data/comics_vocab.p',
    folds_dir: str = '../comics/folds',
    difficulty: str = 'hard',
    n_epochs: int = 10,
    megabatch_size: int = 512,
    batch_size: int = 64,
    num_workers: int = 7,
    lr: float = 0.001,
    show_tqdm: bool = False,
):
    comics_data = h5.File(comics_data_path, 'r')
    vgg_feats = h5.File(vgg_feats_path, 'r')
    # NOTE: Need to pass latin1 as the encoding scheme here, there seems to be some
    # incompability between python 2/3 pickle. For more info see:
    # https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
    word_to_idx, idx_to_word = pickle.load(open(vocab_path, 'rb'), encoding='latin1')

    data_kwargs = {
        'comics_data_path': comics_data_path,
        'vgg_feats_path': vgg_feats_path,
        'vocab_path': vocab_path,
        'folds_dir': folds_dir,
        'difficulty': difficulty,
        'megabatch_size': megabatch_size,
        'batch_size': batch_size,
        'num_workers': num_workers,
    }
    train_dataloader, train_dataset = make_dataloader(**data_kwargs, fold='train')
    valid_dataloader, valid_dataset = make_dataloader(**data_kwargs, fold='dev')
    test_dataloader, test_dataset = make_dataloader(**data_kwargs, fold='test')

    # Predefined parameters.
    # total_pages, max_panels, max_boxes, max_words = train_data.words.shape
    vocab_len = len(word_to_idx)

    model = TextOnlyHeirarchicalLSTM(vocab_len=vocab_len)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(n_epochs), desc='epochs', disable=not show_tqdm):
        start = time.time()

        train_loss = train_one_epoch(model, train_dataloader, optimizer)
        valid_loss, valid_acc, _, _ = eval_one_epoch(model, valid_dataloader)

        end = time.time()
        duration = str(timedelta(seconds=end - start)).split('.')[0]

        print(
            f'Epoch {epoch}: {train_loss=:.4f}, {valid_loss=:.4f}, {valid_acc=:.4f}. Took {duration}s.'
        )

    test_loss, test_acc, test_preds, test_labels = eval_one_epoch(
        model, test_dataloader
    )
    print(f'{test_loss=:.4f}, {test_acc=:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(**vars(parser.parse_args()))
