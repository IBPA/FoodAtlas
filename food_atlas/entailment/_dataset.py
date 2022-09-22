# -*- coding: utf-8 -*-
"""Dataset methods for natural language inference.

Tokenization -> lower casing -> stop words removal -> lemmatization

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * TODOs

"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase
import pandas as pd


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length.
    This is a simple heuristic which will always truncate the longer sequence
    one token at a time. This makes more sense than truncating an equal percent
    of tokens from each, since if one sequence is very short then each token
    that's truncated likely contains more information than a longer sequence.

    Reference: https://github.com/huggingface/transformers/blob/main/examples/
    legacy/run_swag.py

    Args:
        tokens_a: A list of tokens.
        tokens_b: A list of tokens.
        max_length: Maximum length of the output sequence.

    Returns:
        A truncated list of tokens.

    """

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class FoodAtlasNLIDataset(Dataset):
    """NLI dataset class.
    Reference: https://www.kaggle.com/code/tks0123456789/nli-by-bert-pytorch.

    Args:
        premises: list of premises
        hypotheses: list of hypotheses
        labels: list of labels
        tokenizer: tokenizer
        max_seq_len: maximum sequence length

    """

    def __init__(
            self,
            premises: list[str],
            hypotheses: list[str],
            tokenizer: PreTrainedTokenizerBase,
            labels: list[int] = None,
            label_mapper: dict = {
                'Entails': 1, 'Does not entail': 0
            },
            max_seq_len: int = 512):
        if labels is not None:
            self.labels = torch.LongTensor(
                [label_mapper[label] for label in labels]
            )
        else:
            self.labels = None

        self.max_tokens = 0
        self.inputs = []
        for p, h in zip(premises, hypotheses):
            p_ids = tokenizer.encode(p, add_special_tokens=False)
            h_ids = tokenizer.encode(h, add_special_tokens=False)
            _truncate_seq_pair(p_ids, h_ids, max_seq_len - 3)

            input_ids = [tokenizer.cls_token_id] \
                + p_ids \
                + [tokenizer.sep_token_id] \
                + h_ids \
                + [tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * (len(p_ids) + 2) + [1] * (len(h_ids) + 1)

            self.inputs.append([
                torch.LongTensor(input_ids),
                torch.IntTensor(attention_mask),
                torch.IntTensor(token_type_ids)
            ])
            self.max_tokens = max(self.max_tokens, len(input_ids))

        print("Longest Sequence Length:", self.max_tokens)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.inputs[idx], self.labels[idx]
        else:
            return self.inputs[idx], None


def collate_fn_padding(batch):
    """Collate function for padding.

    Args:
        batch: A list of samples.

    Returns:
        A tuple of (inputs, labels). Inputs are tuples of the following:
            input_ids: A tensor of shape (batch_size, seq_len)
            attention_mask: A tensor of shape (batch_size, seq_len)
            token_type_ids: A tensor of shape (batch_size, seq_len)

    """
    inputs, labels = list(zip(*batch))

    input_ids_batch, attention_mask_batch, token_type_ids_batch = zip(*inputs)
    input_ids_batch = pad_sequence(
        input_ids_batch, batch_first=True, padding_value=0)
    attention_mask_batch = pad_sequence(
        attention_mask_batch, batch_first=True, padding_value=0)
    token_type_ids_batch = pad_sequence(
        token_type_ids_batch, batch_first=True, padding_value=1)

    if labels[0] is None:
        return (input_ids_batch, attention_mask_batch, token_type_ids_batch), \
            None
    else:
        return (input_ids_batch, attention_mask_batch, token_type_ids_batch), \
            torch.stack(labels, dim=0)


# def get_food_atlas_data_loaders(
#         path_data_train: str,
#         tokenizer: PreTrainedTokenizerBase,
#         path_data_test: str = None,
#         max_seq_len: int = 512,
#         batch_size: int = 1,
#         shuffle: bool = True,
#         num_workers: int = 0,
#         collate_fn: callable = collate_fn_padding,
#         verbose: bool = True):
#     """Get data loader for food atlas dataset.

#     Args:
#         path_data_train: path to the training data
#         tokenizer: tokenizer
#         path_data_test: path to the testing data
#         max_seq_len: maximum sequence length
#         batch_size: batch size
#         shuffle: whether to shuffle the data
#         num_workers: number of workers
#         collate_fn: collate function
#         verbose: whether to print out the information

#     Returns:
#         data loaders for training and testing

#     """
#     data_loaders = []
#     for path, name in zip(
#             [path_data_train, path_data_test], ['train', 'test']):
#         if path is not None:
#             data = pd.read_csv(path, sep='\t')
#             data = data[['premise', 'hypothesis_string', 'answer']]
#             data = data.rename(
#                 {'hypothesis_string': 'hypothesis'}, axis=1
#             )
#             data = data[~(data['answer'] == 'Skip')]

#             if verbose:
#                 print(f"==={name} set info start===")
#                 print(data['answer'].value_counts())
#                 print(f"===={name} set info end====")

#             dataset = FoodAtlasNLIDataset(
#                 premises=data['premise'].tolist(),
#                 hypotheses=data['hypothesis'].tolist(),
#                 labels=data['answer'].tolist(),
#                 tokenizer=tokenizer,
#                 max_seq_len=max_seq_len
#             )
#             data_loader = DataLoader(
#                 dataset=dataset,
#                 batch_size=batch_size,
#                 shuffle=shuffle,
#                 num_workers=num_workers,
#                 collate_fn=collate_fn
#             )
#         else:
#             data_loader = None
#         data_loaders += [data_loader]

#     data_loader_train, data_loader_test = data_loaders

#     return data_loader_train, data_loader_test


def get_food_atlas_data_loader(
        path_data: str,
        tokenizer: PreTrainedTokenizerBase,
        train: bool = True,
        max_seq_len: int = 512,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        collate_fn: callable = collate_fn_padding,
        verbose: bool = True):
    """Get data loader for food atlas dataset.

    Args:
        path_data: path to the training data
        tokenizer: tokenizer
        train: whether the dataset is used to training. if false, the dataset
            will not contain labels
        max_seq_len: maximum sequence length
        batch_size: batch size
        shuffle: whether to shuffle the data
        num_workers: number of workers
        collate_fn: collate function
        verbose: whether to print out the information

    Returns:
        data loaders for training and testing

    """
    data = pd.read_csv(path_data, sep='\t')
    if train:
        data = data[['premise', 'hypothesis_string', 'answer']]
        data = data[~(data['answer'] == 'Skip')]
    else:
        data = data[['premise', 'hypothesis_string']]

    if verbose:
        print()
        print(f'Number of samples: {data.shape[0]}')
        print()
        if train:
            print(data['answer'].value_counts())
            print()

    dataset = FoodAtlasNLIDataset(
        premises=data['premise'].tolist(),
        hypotheses=data['hypothesis_string'].tolist(),
        labels=data['answer'].tolist() if train else None,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return data_loader
