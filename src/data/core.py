import csv
from functools import partial
import os
import pickle
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


torch.multiprocessing.set_sharing_strategy('file_system')


class ComicsDataset(Dataset):
    def __init__(
        self,
        batch_gen_fn: Callable,
        comics_data_path: str,
        vgg_feats_path: str,
        vocab_path: str,
        folds_dir: str,
        difficulty: str,
        fold: str,
        batch_size: int,
        load_image_feats: bool,
    ):  
        """
        Parameters
        ----------
        batch_gen_fn : Callable
            Function to generate batches.
        """
        assert fold in ('train', 'dev', 'test')
        self.batch_gen_fn = batch_gen_fn
        self.comics_data_path = comics_data_path
        self.vgg_feats_path = vgg_feats_path
        self.vocab_path = vocab_path
        self.folds_dir = folds_dir
        self.difficulty = difficulty
        self.fold = fold
        self.batch_size = batch_size
        self.load_image_feats = load_image_feats

        # NOTE: Need to pass bytes as the encoding scheme here, there seems to be some
        # incompability between python 2/3 pickle. However this means that all strings
        # will be bytestrings, so we need to decode afterwards. For more info see:
        # https://stackoverflow.com/questions/46001958/typeerror-a-bytes-like-object-is-required-not-str-when-opening-python-2-pick/47814305#47814305
        word_to_idx, idx_to_word = pickle.load(open(vocab_path, 'rb'), encoding='bytes')
        self.word_to_idx = {k.decode('utf-8'): v for k, v in word_to_idx.items()}
        self.idx_to_word = {k: v.decode('utf-8') for k, v in idx_to_word.items()}

        with h5.File(self.comics_data_path, 'r') as comics_data:
            words = comics_data[self.fold]['words']
            self.n_pages = words.shape[0]

        self.fold_dict = None
        if fold in ('dev', 'test'):
            self.fold_dict = read_fold(
                os.path.join(folds_dir, f'text_cloze_{fold}_{difficulty}.csv'),
                vdict=self.word_to_idx,
            )

    def __getitem__(self, indices: Iterable[int]) -> List:
        # NOTE: We need to open the hdf5 file inside here in order to ensure thread
        # safety when num_workers > 0.
        with h5.File(self.comics_data_path, 'r') as comics_data:
            fold_data = comics_data[self.fold]

            batches = self.batch_gen_fn(
                fold_data,
                vdict=self.word_to_idx,
                mb_start=indices[0],
                mb_end=indices[-1] + 1,
                batch_size=self.batch_size,
                max_unk=30 if self.fold == 'train' else 2,
                difficulty=self.difficulty,
                fold_dict=self.fold_dict,
                load_image_feats=self.load_image_feats,
            )

            return batches

    def __len__(self) -> int:
        return self.n_pages


# read csv, extract answer candidates and label, and store as dict
def read_fold(csv_file, vdict, max_len=30):
    """
    Reads a CSV, extracts answer candidates and labels, and returns the result as a
    dictionary.

    This function was copied from the original author's code.
    """
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        fold_dict = {}
        for row in reader:
            key = '%s_%s_%s' % (row['book_id'], row['page_id'], row['answer_panel_id'])
            fold_dict[key] = []
            candidates = np.zeros((3, max_len)).astype('int32')
            candidate_masks = np.zeros((3, max_len)).astype('float32')
            candidate_text = np.zeros(3, dtype=object)
            label = [0, 0, 0]
            label[int(row['correct_answer'])] = 1
            for i in range(3):
                c = row['answer_candidate_%d_text' % i].split()
                candidates[i, : len(c)] = [vdict[w] for w in c]
                candidate_masks[i, : len(c)] = 1.0
                candidate_text[i] = row[f'answer_candidate_{i}_text'].encode('utf-8')

            fold_dict[key] = [candidates, candidate_masks, candidate_text, label]

    return fold_dict
