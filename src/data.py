import csv
import os
import pickle
import random
from dataclasses import dataclass
from typing import Iterable, List

import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset


torch.multiprocessing.set_sharing_strategy('file_system')


@dataclass
class ComicPanelBatch:
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
    answer_ids : torch.Tensor, shape (batch_size, n_answers)

    context_vgg_feats : torch.Tensor, shape (batch_size, n_context, n_dim_vgg_fc7)

    context_bounding_boxes: torch.Tensor, shape (batch_size, n_context, n_boxes_max, 4)

    context_box_masks : torch.Tensor, shape (batch_size, n_context, n_boxes_max)
        Mask will be 1 if the ith speech bubble exists in the panel.

    context_words : torch.Tensor, shape (batch_size, n_context, n_boxes_max, n_words_max)

    context_word_masks : torch.Tensor, shape (batch_size, n_context, n_boxes_max, n_words_max)
        Mask will be 1 if the ith word exists in the speech.

    answer_vgg_feats : torch.Tensor, shape (batch_size, n_dim_vgg_fc7)

    answer_bounding_boxes : torch.Tensor, shape (batch_size, 4)

    answer_words : torch.Tensor, shape (batch_size, n_answers, n_words_max)

    answer_words_masks : torch.Tensor, shape (batch_size, n_answers, n_words_max)

    context_box_text : list of strings, length (batch_size * n_context * n_boxes_max)

    answer_text : list of strings, length (batch_size * n_answers)
    """

    answer_ids: torch.Tensor
    context_vgg_feats: torch.Tensor
    context_bounding_boxes: torch.Tensor
    context_box_masks: torch.Tensor
    context_words: torch.Tensor
    context_word_masks: torch.Tensor
    answer_vgg_feats: torch.Tensor
    answer_bounding_boxes: torch.Tensor
    answer_words: torch.Tensor
    answer_word_masks: torch.Tensor
    context_box_text: np.ndarray
    answer_text: np.ndarray

    def to(self, device: str, non_blocking: bool = False) -> None:
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(device, non_blocking=non_blocking))

    @property
    def batch_size(self) -> int:
        return self.answer_ids.size(0)


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
    ):
        assert fold in ('train', 'dev', 'test')
        self.comics_data_path = comics_data_path
        self.vgg_feats_path = vgg_feats_path
        self.vocab_path = vocab_path
        self.folds_dir = folds_dir
        self.difficulty = difficulty
        self.fold = fold
        self.batch_size = batch_size

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

    def __getitem__(self, indices: Iterable[int]) -> List[ComicPanelBatch]:
        # NOTE: We need to open the hdf5 file inside here in order to ensure thread
        # safety when num_workers > 0.
        with h5.File(self.comics_data_path, 'r') as comics_data, h5.File(
            self.vgg_feats_path, 'r'
        ) as vgg_feats:
            fold_data = comics_data[self.fold]
            fold_vgg_feats = vgg_feats[self.fold]

            batches = generate_minibatches_from_megabatch_text_cloze(
                img_mask=fold_data['panel_mask'],
                book_ids=fold_data['book_ids'],
                page_ids=fold_data['page_ids'],
                bboxes=fold_data['bbox'],
                bbox_mask=fold_data['bbox_mask'],
                words=fold_data['words'],
                word_mask=fold_data['word_mask'],
                comics_fc7=fold_vgg_feats['vgg_features'],
                raw_text=fold_data['raw_text'],
                vdict=self.word_to_idx,
                mb_start=indices[0],
                mb_end=indices[-1] + 1,
                batch_size=self.batch_size,
                max_unk=30 if self.fold == 'train' else 2,
                difficulty=self.difficulty,
                fold_dict=self.fold_dict,
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


def generate_minibatches_from_megabatch_text_cloze(
    img_mask,
    book_ids,
    page_ids,
    bboxes,
    bbox_mask,
    words,
    word_mask,
    comics_fc7,
    raw_text,
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
) -> List[ComicPanelBatch]:
    """
    Takes a "megabatch" (multiple pages of comics) and generates a bunch of minibatches.

    This function was originally copied from the original authors' code, but then
    modified to suit our needs.
    """
    curr_fc7 = comics_fc7[mb_start:mb_end]

    # binarize bounding box mask (no narrative box distinction)
    curr_bmask_raw = bbox_mask[mb_start:mb_end]
    curr_bmask = np.clip(curr_bmask_raw, 0, 1)

    curr_bboxes = bboxes[mb_start:mb_end] / 224.0
    curr_words = words[mb_start:mb_end]
    curr_wmask = word_mask[mb_start:mb_end]
    curr_book_ids = book_ids[mb_start:mb_end]
    curr_page_ids = page_ids[mb_start:mb_end]
    curr_imasks = img_mask[mb_start:mb_end]
    curr_raw_text = raw_text[mb_start:mb_end]

    num_panels = np.sum(curr_imasks, axis=-1).astype('int32')

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
    context_imgs = []
    context_words = []
    context_wmask = []
    context_bboxes = []
    context_bmask = []
    context_fc7 = []
    answer_ids = []
    answer_imgs = []
    answer_fc7 = []
    answer_bboxes = []
    answer_bmask = []
    candidates = []
    context_raw_text = []
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

        # subtract 1 because random.randint is inclusive
        prev_np = num_panels[i - 1] - 1
        next_np = num_panels[i + 1] - 1

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

            # Context information.
            context_fc7.append(curr_fc7[i, j : j + context_size])
            context_bboxes.append(curr_bboxes[i, j : j + context_size])
            context_bmask.append(curr_bmask[i, j : j + context_size])
            context_words.append(curr_words[i, j : j + context_size])
            context_wmask.append(curr_wmask[i, j : j + context_size])

            # I (Jamin) added:
            context_raw_text.append(curr_raw_text[i, j : j + context_size])

            # Answer information.
            key = (curr_book_ids[i], curr_page_ids[i], j + context_size)
            answer_ids.append(key)
            answer_fc7.append(curr_fc7[i, j + context_size])
            answer_bboxes.append(curr_bboxes[i, j + context_size][0])

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

        a_id = answer_ids[start:end]
        c_fc7 = np.array(context_fc7[start:end])
        c_bb = np.array(context_bboxes[start:end]).astype('float32')
        c_bbm = np.array(context_bmask[start:end]).astype('float32')
        c_w = np.array(context_words[start:end]).astype('int32')
        c_wm = np.array(context_wmask[start:end]).astype('float32')
        a_fc7 = np.array(answer_fc7[start:end])
        a_bb = np.array(answer_bboxes[start:end]).astype('float32')
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
        c_txt = c_txt.ravel().tolist()
        a_txt = a_txt.ravel().tolist()

        labels = np.argmax(labels, axis=-1)

        batch_data = [
            a_id,
            c_fc7,
            c_bb,
            c_bbm,
            c_w,
            c_wm,
            a_fc7,
            a_bb,
            a_w,
            a_wm,
            labels,
        ]
        all_batch_data.append(
            (
                ComicPanelBatch(
                    answer_ids=torch.IntTensor(a_id),
                    context_vgg_feats=torch.FloatTensor(c_fc7),
                    context_bounding_boxes=torch.FloatTensor(c_bb),
                    context_box_masks=torch.BoolTensor(c_bbm),
                    context_words=torch.IntTensor(c_w),
                    context_word_masks=torch.BoolTensor(c_wm),
                    answer_vgg_feats=torch.FloatTensor(a_fc7),
                    answer_bounding_boxes=torch.FloatTensor(a_bb),
                    answer_words=torch.IntTensor(a_w),
                    answer_word_masks=torch.BoolTensor(a_wm),
                    context_box_text=c_txt,
                    answer_text=a_txt,
                ),
                torch.LongTensor(labels),
            )
        )

    return all_batch_data
