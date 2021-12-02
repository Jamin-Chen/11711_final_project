import os
import pickle
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, List, Optional, Tuple, Union

import h5py as h5
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from tqdm import tqdm

from config import ExperimentConfig
from data.text_cloze import TextClozeBatch, TextClozeDataset
from data.visual_cloze import VisualClozeBatch, VisualClozeDataset

# from models.lstm import TextOnlyHeirarchicalLSTM
from models import (
    TextOnlyTextClozeTransformerBaseline,
    TextOnlyVisualClozeTransformerBaseline,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# We need this line below because each dataloader process will attempt to read from the
# h5 file.
torch.multiprocessing.set_sharing_strategy('file_system')


Batch = Union[TextClozeBatch, VisualClozeBatch]


def collate_fn(batches_and_labels: List[List[Tuple[Batch, torch.Tensor]]]):
    """
    Dummy collate function to just return the single element from the list (which will
    already be a list of batches), since our dataset does the batching logic.
    """
    assert len(batches_and_labels) == 1
    return batches_and_labels[0]


def make_dataloader(
    model_name: str,
    comics_data_path: str,
    vgg_feats_path: str,
    vocab_path: str,
    folds_dir: str,
    fold: str,
    difficulty: str,
    megabatch_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    do_spell_check: bool,
):
    # Names are:
    # text_only_text_cloze
    # image_text_text_cloze
    # text_only_visual_cloze
    # image_text_visual_cloze
    dataset_cls = (
        TextClozeDataset if model_name.endswith('text_cloze') else VisualClozeDataset
    )
    load_image_feats = model_name.startswith('image')

    dataset = dataset_cls(
        comics_data_path=comics_data_path,
        vgg_feats_path=vgg_feats_path,
        vocab_path=vocab_path,
        folds_dir=folds_dir,
        difficulty=difficulty,
        fold=fold,
        batch_size=batch_size,
        load_image_feats=load_image_feats,
        do_spell_check=do_spell_check,
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
        pin_memory=pin_memory,
    )

    return dataloader, dataset


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    pbar_total: int,
    pbar_step: int,
    pbar_desc: str,
    iters_to_accumulate: int = 1,
):
    model.train()

    total_loss = 0
    n_batches = 0

    with tqdm(leave=False, total=pbar_total, desc=pbar_desc) as pbar:
        for batches in dataloader:
            for batch, labels in batches:
                batch.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    logits = model(batch)

                loss = F.cross_entropy(logits, labels)
                loss /= iters_to_accumulate

                total_loss += loss.item()
                n_batches += 1

                # Accumulates scaled gradients.
                scaler.scale(loss).backward()

                if n_batches % iters_to_accumulate == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # loss.backward()
                # optimizer.step()
            pbar.update(pbar_step)

    avg_loss = total_loss / n_batches
    return avg_loss


def eval_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    pbar_total: int,
    pbar_step: int,
    pbar_desc: str,
):
    model.eval()

    with torch.no_grad():
        total_loss = 0
        n_batches = 0
        all_preds = []
        all_labels = []

        with tqdm(leave=False, total=pbar_total, desc=pbar_desc) as pbar:
            for batches in dataloader:
                for batch, labels in batches:
                    batch.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with torch.cuda.amp.autocast():
                        logits = model(batch)

                    loss = F.cross_entropy(logits, labels)

                    preds = torch.argmax(logits, dim=-1)
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

                    total_loss += loss.item()
                    n_batches += 1
                pbar.update(pbar_step)

        avg_loss = total_loss / n_batches

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        n_examples = all_preds.size(0)
        acc = (torch.sum(all_preds == all_labels) / n_examples).item()

        return avg_loss, acc, all_preds, all_labels


@hydra.main(config_path='../conf', config_name='default')
def main(config: ExperimentConfig):
    print(OmegaConf.to_yaml(config))

    megabatch_size = config.data.megabatch_size
    n_epochs = config.model.n_epochs
    lr = config.model.lr
    iters_to_accumulate = config.model.iters_to_accumulate
    model_name = config.model.name

    # NOTE: Need to pass bytes as the encoding scheme here, there seems to be some
    # incompability between python 2/3 pickle. For more info see:
    # https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
    word_to_idx, idx_to_word = pickle.load(
        open(config.data.vocab_path, 'rb'), encoding='bytes'
    )

    data_kwargs = {
        'model_name': config.model.name,
        **config.data,
        'difficulty': config.task.difficulty,
        'batch_size': config.model.batch_size,
    }
    train_dataloader, train_dataset = make_dataloader(**data_kwargs, fold='train')
    valid_dataloader, valid_dataset = make_dataloader(**data_kwargs, fold='dev')
    test_dataloader, test_dataset = make_dataloader(**data_kwargs, fold='test')

    n_train_pages = len(train_dataset)
    n_valid_pages = len(valid_dataset)
    n_test_pages = len(test_dataset)

    # Predefined parameters.
    # total_pages, max_panels, max_boxes, max_words = train_data.words.shape
    vocab_len = len(word_to_idx)

    if model_name == 'text_only_text_cloze':
        model = TextOnlyTextClozeTransformerBaseline()
    elif model_name == 'text_only_visual_cloze':
        model = TextOnlyVisualClozeTransformerBaseline()
    else:
        raise ValueError(f'Unsupported model name: {model_name}.')

    print(f'Initialized model instance {model.__class__.__name__}')

    # model = TextOnlyHeirarchicalLSTM(vocab_len)
    model.to(device)

    scaler = torch.cuda.amp.GradScaler()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    effective_batch_size = config.model.batch_size * config.model.iters_to_accumulate
    print(f'{n_params} parameters.')
    print(f'Using an effective batch size of {effective_batch_size}.')

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        start = time.time()

        train_loss = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            scaler,
            pbar_total=n_train_pages,
            pbar_step=megabatch_size,
            pbar_desc='Train pages',
            iters_to_accumulate=iters_to_accumulate,
        )
        valid_loss, valid_acc, _, _ = eval_one_epoch(
            model,
            valid_dataloader,
            pbar_total=n_valid_pages,
            pbar_step=megabatch_size,
            pbar_desc='Valid. pages',
        )
        test_loss, test_acc, test_preds, test_labels = eval_one_epoch(
            model,
            test_dataloader,
            pbar_total=n_test_pages,
            pbar_step=megabatch_size,
            pbar_desc='Test Pages',
        )

        end = time.time()
        duration = str(timedelta(seconds=end - start)).split('.')[0]

        print(
            f'Epoch {epoch}: {train_loss=:.4f}, {valid_loss=:.4f}, {valid_acc=:.4f}, {test_loss=:.4f}, {test_acc=:.4f}. Took {duration}s.'
        )


if __name__ == '__main__':
    main()
