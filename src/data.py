import csv
from functools import partial
import os
import pickle
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


torch.multiprocessing.set_sharing_strategy('file_system')


# TODO: Probably safe to remove the bounding box features, those aren't used anyways.


@dataclass
class TextClozeBatch:
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

    answer_text : List[str], length (batch_size * n_answers)
    """

    batch_size: int
    n_context: int
    context_panel_bert_input: Dict
    context_panel_images: Optional[torch.Tensor]
    answer_panel_bert_input: Dict

    def to(self, device: str, non_blocking: bool = False) -> None:
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(device, non_blocking=non_blocking))


# TODO: Pass fold_dict through here.
class ComicsDataset(Dataset):
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
            self.fold_dict = read_fold(
                os.path.join(folds_dir, f'text_cloze_{fold}_{difficulty}.csv'),
                vdict=self.word_to_idx,
            )

    def __getitem__(self, indices: Iterable[int]) -> List[TextClozeBatch]:
        # NOTE: We need to open the hdf5 file inside here in order to ensure thread
        # safety when num_workers > 0.
        with h5.File(self.comics_data_path, 'r') as comics_data:
            fold_data = comics_data[self.fold]

            batches = generate_minibatches_from_megabatch_text_cloze(
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


# images already have the text boxes masked out
# images is same shape as the raw text, so it's a 1:1 mapping
# img_mask tells us which panels actually contain images (vs being just black)


def generate_minibatches_from_megabatch_text_cloze(
    fold_data,
    vdict,
    mb_start,
    mb_end,
    batch_size=64,
    context_size=3,
    shortest_answer=3,
    window_size=3,
    num_candidates=3,
    max_unk=2,
    difficulty='hard',
    only_singleton_panels=True,
    fold_dict=None,
    shuffle_candidates=True,
    load_image_feats=False,
) -> List[TextClozeBatch]:
    """
    Takes a "megabatch" (multiple pages of comics) and generates a bunch of minibatches.

    This function was originally copied from the original authors' code, but then
    modified to suit our needs.
    """
    # Read in the fold data.
    if load_image_feats:
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
    curr_images = images[mb_start:mb_end]
    curr_image_masks = image_masks[mb_start:mb_end]
    curr_raw_text = raw_text[mb_start:mb_end]

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
    a_txt = []

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

            # if text cloze, make sure answer panel has only one text box
            if only_singleton_panels:
                if np.sum(curr_bmask[i, j + context_size]) != 1:
                    continue

            # make sure answer text box isn't blank or super short
            if words_per_box[i, j + context_size, 0] < shortest_answer:
                continue

            # make sure context/answer text doesn't have too many UNKs
            # because that is an indicator of poor OCR
            too_many_unks = False
            for c_ind in range(context_size + 1):
                if np.sum(unks_in_candidates[i, j + c_ind] >= max_unk, axis=-1) > 0:
                    too_many_unks = True
            if too_many_unks:
                continue

            # I (Jamin) added:
            context_raw_text.append(curr_raw_text[i, j : j + context_size])
            if load_image_feats:
                context_images.append(curr_images[i, j : j + context_size])

            # Answer information.
            key = (curr_book_ids[i], curr_page_ids[i], j + context_size)

            # if cached fold, just use the stored candidates
            key = '_'.join([str(z) for z in key])
            if fold_dict and key in fold_dict:
                # if False:
                candidates.append(fold_dict[key])

            # otherwise randomly sample candidates (for training)
            else:

                # candidates come from previous / next page
                if difficulty == 'hard':
                    text_candidates = np.zeros((3, curr_words.shape[-1]))
                    mask_candidates = np.zeros((3, curr_words.shape[-1]))
                    raw_text_candidates = np.zeros(3, dtype=object)

                    # see if any panels in the surrounding pages have long enough text boxes
                    window_start = max(0, i - window_size)
                    window_end = min(i + 1 + window_size, iter_end + 1)

                    coords_1 = [
                        coord
                        for coord in pc_tuple
                        if coord[0] in set(range(window_start, i))
                    ]
                    coords_2 = [
                        coord
                        for coord in pc_tuple
                        if coord[0] in set(range(i + 1, window_end))
                    ]

                    # if no usable candidates found in neighboring pages
                    # just randomly sample from all possible candidates
                    # note: this is very rare!
                    if len(coords_1) == 0:
                        chosen_prev_candidate = random.choice(pc_tuple)
                    else:
                        chosen_prev_candidate = random.choice(coords_1)

                    if len(coords_2) == 0:
                        chosen_next_candidate = random.choice(pc_tuple)
                    else:
                        chosen_next_candidate = random.choice(coords_2)

                    # corr = 0, prev = 1, next = 2
                    text_candidates[0] = curr_words[i, j + context_size, 0]
                    text_candidates[1] = curr_words[chosen_prev_candidate]
                    text_candidates[2] = curr_words[chosen_next_candidate]
                    mask_candidates[0] = curr_wmask[i, j + context_size, 0]
                    mask_candidates[1] = curr_wmask[chosen_prev_candidate]
                    mask_candidates[2] = curr_wmask[chosen_next_candidate]

                    # I (Jamin) added:
                    raw_text_candidates[0] = curr_raw_text[i, j + context_size, 0]
                    raw_text_candidates[1] = curr_raw_text[chosen_prev_candidate]
                    raw_text_candidates[2] = curr_raw_text[chosen_next_candidate]

                    candidates.append(
                        (
                            text_candidates,
                            mask_candidates,
                            raw_text_candidates,
                            [1, 0, 0],
                        )
                    )

                # candidates come from random pages in the megabatch
                else:
                    text_candidates = np.zeros((num_candidates, curr_words.shape[-1]))
                    mask_candidates = np.zeros((num_candidates, curr_words.shape[-1]))
                    raw_text_candidates = np.zeros(num_candidates, dtype=object)

                    # corr = 0, all other indices are wrong candidates
                    text_candidates[0] = curr_words[i, j + context_size, 0]
                    mask_candidates[0] = curr_wmask[i, j + context_size, 0]
                    raw_text_candidates[0] = curr_raw_text[i, j + context_size, 0]

                    for cand_idx in range(num_candidates - 1):
                        coords = random.choice(pc_tuple)

                        text_candidates[cand_idx + 1] = curr_words[coords]
                        mask_candidates[cand_idx + 1] = curr_wmask[coords]
                        raw_text_candidates[cand_idx + 1] = curr_raw_text[coords]

                    candidates.append(
                        (
                            text_candidates,
                            mask_candidates,
                            raw_text_candidates,
                            [1, 0, 0],
                        )
                    )

    # create numpy-ized minibatches
    batch_inds = [(x, x + batch_size) for x in range(0, len(candidates), batch_size)]

    all_batch_data = []
    for start, end in batch_inds:
        context_panel_images = None
        if load_image_feats:
            context_panel_images = np.array(context_images[start:end])

        c_txt = np.array(context_raw_text[start:end])

        a_w = []
        a_wm = []
        labels = []
        a_txt = []

        for cand in candidates[start:end]:
            a_w.append(cand[0])
            a_wm.append(cand[1])
            a_txt.append(cand[2])
            labels.append(cand[3])

        a_w = np.array(a_w).astype('int32')
        a_wm = np.array(a_wm).astype('float32')
        labels = np.array(labels).astype('int32')
        a_txt = np.array(a_txt)

        if shuffle_candidates and not fold_dict:
            for idx in range(a_w.shape[0]):
                p = np.random.permutation(a_w.shape[1])
                a_w[idx] = a_w[idx, p]
                a_wm[idx] = a_wm[idx, p]
                labels[idx] = labels[idx, p]
                a_txt[idx] = a_txt[idx, p]

        # Convert from bytestrings to strings.
        for indices, _ in np.ndenumerate(c_txt):
            c_txt[indices] = c_txt[indices].decode('utf-8')

        for indices, _ in np.ndenumerate(a_txt):
            a_txt[indices] = a_txt[indices].decode('utf-8')

        # Convert raw sentences to flat lists instead of numpy arrays.
        context_panel_text = c_txt.ravel().tolist()
        answer_panel_text = a_txt.ravel().tolist()

        # Combine the text boxes within each panel into a single string.
        context_panel_text = [
            ' '.join(context_panel_text[i : i + 3])
            for i in range(0, len(context_panel_text), 3)
        ]

        # Assert shapes are correct before we encode with BERT tokenizer.
        # The true batch size (which can be smaller).
        true_batch_size = min(end, len(candidates)) - start
        assert len(context_panel_text) == true_batch_size * context_size
        assert len(answer_panel_text) == true_batch_size * 3
        if context_panel_images is not None:
            assert context_panel_images.shape == (
                true_batch_size,
                context_size,
                224,
                224,
                3,
            )

        # Encode text as BERT tokens (this step is CPU bound).
        # TODO: Should probably pass this stuff in somehow instead of hardcoding it.
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenize = partial(
            bert_tokenizer,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128,  # shorter sequence length to save memory
        )

        context_panel_bert_input = tokenize(context_panel_text)
        answer_panel_bert_input = tokenize(answer_panel_text)

        batch = TextClozeBatch(
            batch_size=true_batch_size,
            n_context=context_size,
            context_panel_bert_input=context_panel_bert_input,
            context_panel_images=context_panel_images,
            answer_panel_bert_input=answer_panel_bert_input,
        )

        labels = torch.LongTensor(np.argmax(labels, axis=-1))

        all_batch_data.append((batch, labels))

    return all_batch_data
