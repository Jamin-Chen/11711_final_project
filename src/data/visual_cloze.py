import csv
import os
import pickle
import random
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional

import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, ViTFeatureExtractor


@dataclass
class VisualClozeBatch:
    """
    Batch of examples from the COMICS dataset.

    For the shapes below:
     - batch_size: Batch size.
     - n_answers: Number of answer choices (3).
     - n_context: Number of context panels (3).
     - n_dim_vgg_fc7: Dimension of fc7 layer in VGG16 (4096).
     - n_boxes_max: Maximum number of boxes per panel (3).
     - n_words_max: Maximum number of words per text box (30).

    Attributes
    ----------
    context_box_text : List[str], length (batch_size * n_context)
        Each element in the list will contain the text for all of the text boxes in that
        panel, joined into a single string.

    context_images : shape (batch_size, n_context, 224, 224, 3)

    answer_panel_images : shape (batch_size, n_answers, 224, 224, 3)
    """

    batch_size: int
    n_context: int
    context_panel_bert_input: Dict
    context_panel_vit_input: Optional[torch.Tensor]
    answer_panel_vit_input: torch.Tensor

    def to(self, device: str, non_blocking: bool = False) -> None:
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(device, non_blocking=non_blocking))


class VisualClozeDataset(Dataset):
    def __init__(
        self,
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

        read_fold_fn : Callable
            Function to read the fold dictionary (precomputed dev and test labels).
        """
        assert fold in ('train', 'dev', 'test')
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
            self.fold_dict = read_fold_visual_cloze(
                os.path.join(folds_dir, f'visual_cloze_{fold}_{difficulty}.csv'),
                vdict=self.word_to_idx,
            )

    def __getitem__(self, indices: List[int]) -> List:
        # NOTE: We need to open the hdf5 file inside here in order to ensure thread
        # safety when num_workers > 0.
        with h5.File(self.comics_data_path, 'r') as comics_data:
            fold_data = comics_data[self.fold]

            batches = generate_minibatches_from_megabatch_visual_cloze(
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


def read_fold_visual_cloze(csv_file, vdict, max_len=30):
    """
    Reads a CSV, extracts answer candidates and labels, and returns the result as a
    dictionary.

    This function was copied from the original author's code.
    """
    reader = csv.DictReader(open(csv_file, 'r'))
    fold_dict = {}
    for row in reader:
        key = (row['book_id'], row['page_id'], row['answer_panel_id'])
        fold_dict[key] = []
        candidates = []
        label = [0, 0, 0]
        label[int(row['correct_answer'])] = 1
        for i in range(3):
            c = row['answer_candidate_id_%d' % i]
            candidates.append([int(x) for x in c.split('_')])

        fold_dict[key] = [candidates, label]

    return fold_dict


def generate_minibatches_from_megabatch_visual_cloze(
    fold_data,
    vdict,
    mb_start,
    mb_end,
    difficulty,
    batch_size=64,
    context_size=3,
    shortest_answer=3,
    window_size=3,
    num_candidates=3,
    max_unk=2,
    fold_dict=None,
    shuffle_candidates=True,
    load_image_feats=False,
) -> List[VisualClozeBatch]:
    """
    Takes a "megabatch" (multiple pages of comics) and generates a bunch of minibatches.

    This function was originally copied from the original authors' code, but then
    modified to suit our needs.
    """
    # Read in the fold data.
    images = fold_data['images']
    image_masks = fold_data['panel_mask']
    book_ids = fold_data['book_ids']
    page_ids = fold_data['page_ids']
    bbox_mask = fold_data['bbox_mask']
    words = fold_data['words']
    word_mask = fold_data['word_mask']
    raw_text = fold_data['raw_text']

    # binarize bounding box mask (no narrative box distinction)
    curr_bmask_raw = bbox_mask[mb_start:mb_end]
    curr_bmask = np.clip(curr_bmask_raw, 0, 1)

    curr_words = words[mb_start:mb_end]
    curr_wmask = word_mask[mb_start:mb_end]
    curr_book_ids = book_ids[mb_start:mb_end]
    curr_page_ids = page_ids[mb_start:mb_end]
    if load_image_feats:
        curr_images = images[mb_start:mb_end]
    curr_image_masks = image_masks[mb_start:mb_end]
    curr_raw_text = raw_text[mb_start:mb_end]

    # inverse mapping for visual cloze candidates
    # page_id to actual index
    page_to_idx = {}
    for idx, (book_id, page_id) in enumerate(zip(curr_book_ids, curr_page_ids)):
        page_to_idx[(book_id, page_id)] = idx

    num_panels = np.sum(curr_image_masks, axis=-1).astype('int32')

    # need to sum the number of words per box
    words_per_box = np.sum(curr_wmask, axis=-1).astype('int32')
    possible_candidates = np.where(words_per_box >= shortest_answer)
    possible_candidates = set(zip(*possible_candidates))

    # compute number of UNKs per box for filtering
    unks_in_candidates = np.sum((curr_words == vdict['UNK']), axis=-1)
    unk_candidates = np.where(unks_in_candidates < max_unk)
    unk_candidates = set(zip(*unk_candidates))
    possible_candidates = possible_candidates.intersection(unk_candidates)
    pc_tuple = tuple(possible_candidates)

    # loop through each page, create as many training examples as possible
    context_raw_text = []
    context_images = []
    candidates = []

    iter_end = num_panels.shape[0] - 1
    for i in range(0, iter_end):
        curr_np = num_panels[i]

        # not enough panels to have context and candidate
        if curr_np < context_size + 1:
            continue

        # see if there is a previous and next page
        if (
            curr_page_ids[i - 1] != curr_page_ids[i] - 1
            or curr_page_ids[i + 1] != curr_page_ids[i] + 1
        ):
            continue

        num_examples = curr_np - context_size
        for j in range(num_examples):
            # I (Jamin) added:
            context_raw_text.append(curr_raw_text[i, j : j + context_size])
            if load_image_feats:
                context_images.append(curr_images[i, j : j + context_size])

            # Answer information.
            key = (curr_book_ids[i], curr_page_ids[i], j + context_size)

            # if cached fold, just use the stored candidates
            # TODO: Make sure we get some hits here
            if fold_dict and key in fold_dict:
                # if False:
                # print('cache hit')
                candidates.append(fold_dict[key])

            # otherwise randomly sample candidates (for training)
            else:
                # print('cache miss')
                window_start = max(0, i - window_size)
                window_end = min(i + 1 + window_size, iter_end + 1)

                # candidates come from previous / next page
                if difficulty == 'hard':
                    random_page_1 = random.randint(window_start, max(0, i - 1))
                    random_page_2 = random.randint(i + 1, window_end - 1)

                # candidates come from random pages in the megabatch
                else:
                    random_page_1 = random.randint(0, iter_end - 1)
                    random_page_2 = random.randint(0, iter_end - 1)

                prev_sel = random.randint(0, num_panels[random_page_1] - 1)
                next_sel = random.randint(0, num_panels[random_page_2] - 1)

                # corr = 0, prev = 1, next = 2
                candidate_ids = []
                candidate_ids.append(key)
                candidate_ids.append(
                    (
                        curr_book_ids[random_page_1],
                        curr_page_ids[random_page_1],
                        prev_sel,
                    )
                )
                candidate_ids.append(
                    (
                        curr_book_ids[random_page_2],
                        curr_page_ids[random_page_2],
                        next_sel,
                    )
                )

                candidates.append([candidate_ids, [1, 0, 0]])

    # create numpy-ized minibatches
    batch_inds = [(x, x + batch_size) for x in range(0, len(candidates), batch_size)]

    all_batch_data = []
    for start, end in batch_inds:
        context_panel_images = None
        if load_image_feats:
            context_panel_images = np.array(context_images[start:end])

        c_txt = np.array(context_raw_text[start:end])

        answer_panel_images = []
        labels = []

        # TODO: Use same indexing scheme as author, the book and panel are merged into one dimension
        for cand in candidates[start:end]:
            curr_answer_panel_images = []
            for book_id, page_id, panel_idx in cand[0]:
                page_idx = page_to_idx[(book_id, page_id)]
                curr_answer_panel_images.append(images[page_idx, panel_idx])
            answer_panel_images.append(curr_answer_panel_images)

            labels.append(cand[1])

        answer_panel_images = np.array(answer_panel_images).astype('float32')
        labels = np.array(labels).astype('int32')

        if shuffle_candidates and not fold_dict:
            for idx in range(answer_panel_images.shape[0]):
                p = np.random.permutation(answer_panel_images.shape[1])
                answer_panel_images[idx] = answer_panel_images[idx, p]
                labels[idx] = labels[idx, p]

        # Convert panel images into a list of numpy arrays.
        answer_panel_images = answer_panel_images.reshape(-1, 224, 224, 3)
        answer_panel_images = [
            answer_panel_images[i] for i in range(answer_panel_images.shape[0])
        ]

        if load_image_feats:
            context_panel_images = context_panel_images.reshape(-1, 224, 224, 3)
            context_panel_images = [
                context_panel_images[i] for i in range(context_panel_images.shape[0])
            ]

        # Convert from bytestrings to strings.
        for indices, _ in np.ndenumerate(c_txt):
            c_txt[indices] = c_txt[indices].decode('utf-8')

        # Convert raw sentences to flat lists instead of numpy arrays.
        context_panel_text = c_txt.ravel().tolist()

        # Combine the text boxes within each panel into a single string.
        context_panel_text = [
            ' '.join(context_panel_text[i : i + 3])
            for i in range(0, len(context_panel_text), 3)
        ]

        # Assert shapes are correct before we encode with BERT tokenizer.
        # The true batch size (which can be smaller).
        true_batch_size = min(end, len(candidates)) - start
        assert len(context_panel_text) == true_batch_size * context_size
        assert len(answer_panel_images) == true_batch_size * num_candidates
        assert answer_panel_images[0].shape == (224, 224, 3)
        if context_panel_images is not None:
            assert len(context_panel_images) == true_batch_size * context_size
            assert context_panel_images[0].shape == (224, 224, 3)

        # Encode text as BERT tokens (this step is CPU bound).
        # TODO: Should probably pass this stuff in somehow instead of hardcoding it.
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        do_bert_tokenize = partial(
            bert_tokenizer,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128,  # shorter sequence length to save memory
        )
        context_panel_bert_input = do_bert_tokenize(context_panel_text)

        # Prepare data for vision transformer.
        vit_feature_extractor = ViTFeatureExtractor.from_pretrained(
            'google/vit-base-patch16-224-in21k'
        )
        do_vit_featurize = partial(vit_feature_extractor, return_tensors='pt')

        # Reshape images to be shape (C, H, W) before we pass to ViT feature extractor.
        answer_panel_images = [np.moveaxis(arr, 2, 0) for arr in answer_panel_images]
        answer_panel_vit_input = do_vit_featurize(answer_panel_images)

        context_panel_vit_input = None
        if load_image_feats:
            context_panel_images = [
                np.moveaxis(arr, 2, 0) for arr in context_panel_images
            ]
            context_panel_vit_input = do_vit_featurize(context_panel_images)

        batch = VisualClozeBatch(
            batch_size=true_batch_size,
            n_context=context_size,
            context_panel_bert_input=context_panel_bert_input,
            context_panel_vit_input=context_panel_vit_input,
            answer_panel_vit_input=answer_panel_vit_input,
        )

        labels = torch.LongTensor(np.argmax(labels, axis=-1))

        all_batch_data.append((batch, labels))

    return all_batch_data
