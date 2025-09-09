import copy
from typing import List, Dict, Union, Iterable

from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class RLDataset(Dataset):

    def __init__(self, tokenizer, data=None, chat_template=True):
        """
        Dataset for loading RL examples
        :param tokenizer: hf tokenizer
        :param data: df - must include "messages" column
        :param chat_template: str
        """
        if data is None:
            logger.info('Using dummy data as data path not specified')
            dummy_data = [{"id": f"dummy_{x}",
                           "messages": [
                               {"content": "What is the capital of France and tell me about it?", "role": "user"}]} for
                          x in range(64)]
            self.data = pd.DataFrame(dummy_data)
        else:
            self.data = data
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    def __getitem__(self, idx):
        messages = [*self.data.iloc[idx]['messages']]
        # Include other columns e.g ground truth answer
        other_columns = {col: self.data.iloc[idx][col] for col in self.data.columns if col != 'messages'}
        all_inputs = {}
        # Apply chat template if provided
        if self.chat_template:
            prompt_inputs = self.tokenizer.apply_chat_template(
                [message for message in messages if message['role'] != 'assistant'], return_tensors='pt',
                return_dict=True)
        else:
            prompt_inputs = self.tokenizer(
                ['\n'.join([message['content'] for message in messages if message['role'] != 'assistant'])],
                return_tensors='pt')
        all_inputs['input_ids'] = prompt_inputs['input_ids'].flatten()
        all_inputs['attention_mask'] = prompt_inputs['attention_mask'].flatten()
        prompt_tok_len = prompt_inputs['input_ids'].shape[-1]
        all_inputs['prompt_tok_len'] = prompt_tok_len
        all_inputs['labels'] = copy.deepcopy(prompt_inputs['input_ids'].flatten())
        all_inputs['labels'][:] = -100
        all_inputs = {**all_inputs, **other_columns}
        return all_inputs

    def __len__(self):
        return len(self.data)

""" BELOW MODIFIED FROM ORIGINAL TORCHTUNE
https://github.com/pytorch/torchtune
"""
def pad_sequence(
        sequences: Union[Tensor, List[Tensor]],
        batch_first: bool = False,
        padding_value: float = 0.0,
        padding_side: str = "left",
) -> Tensor:
    r"""Pad a list of variable length Tensors with :attr:`padding_value`.

    ``pad_sequence`` stacks a list of Tensors along a new dimension, and pads them
    to equal length. :attr:`sequences` can be list of sequences with size ``L x *``,
    where `L` is length of the sequence and ``*`` is any number of dimensions
    (including 0). If :attr:`batch_first` is ``False``, the output is of size
    ``T x B x *``, and ``B x T x *`` otherwise, where ``B`` is the batch size
    (the number of elements in :attr:`sequences`), ``T`` is the length of the longest
    sequence.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): if ``True``, the output will be in ``B x T x *``
            format, ``T x B x *`` otherwise.
        padding_value (float, optional): value for padded elements. Default: 0.
        padding_side (str, optional): the side to pad the sequences on.
            Default: "right".

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """
    if not (torch.jit.is_tracing() or torch.jit.is_scripting()):
        # JIT doesn't support `Iterable`
        if not isinstance(sequences, Iterable):
            msg = (
                "pad_sequence: Expected iterable for input sequences, but got arg of type: "
                f"{type(sequences)}"
            )
            raise RuntimeError(msg)

        # In JIT context this leads to,
        # RuntimeError: cannot statically infer the expected size of a list in this context
        sequences = tuple(sequences)  # type: ignore[assignment]
    else:
        # For JIT, we only support Union[Tensor, Tuple[Tensor]]
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.unbind(0)  # type: ignore[assignment]

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    return torch._C._nn.pad_sequence(
        sequences, batch_first, padding_value, padding_side  # type: ignore[arg-type]
    )


def padded_collate_rl(
        batch,
        padding_idx: int = 0,
        ignore_idx: int = -100,
) -> Dict[str, torch.Tensor]:
    """Pad a batch of sequences to the longest sequence length in the batch, and
    convert integer lists to tensors.

    Args:
        batch (List[Dict[str, List[int]]]): A list of dictionaries containing input, label pairs.
        padding_idx (int): Padding index for input ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.

    Returns:
        Dict[str, torch.Tensor]: Collated input and label tensors.

    Example:
        >>> token_pairs = [
        >>>    {"tokens": [1, 2, 3], "labels": [4, 5, 6]},
        >>>    {"tokens": [7,], "labels": [10,]},
        >>> ]
        >>> collated = padded_collate(
        >>>    batch=token_pairs,
        >>>    padding_idx=padding_idx,
        >>>    ignore_idx=ignore_idx,
        >>> )
        >>> collated["tokens"]
        >>> tensor([[1, 2, 3], [7, 0, 0]])
        >>> collated["labels"]
        >>> tensor([[4, 5, 6], [10, -100, -100]])
    """
    input_ids = pad_sequence(
        [x["input_ids"].detach().clone() for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    labels = pad_sequence(
        [x["labels"].detach().clone() for x in batch],
        batch_first=True,
        padding_value=ignore_idx,
    )
    attention_mask = pad_sequence(
        [x["attention_mask"].detach().clone() for x in batch],
        batch_first=True,
        padding_value=0,
    )

    input_ids_seq_len = input_ids.shape[-1]
    labels_seq_len = labels.shape[-1]
    # Hack to pad correctly and not use max_seq_len, which is costly
    if input_ids_seq_len > labels_seq_len:
        labels = F.pad(
            labels, (0, input_ids_seq_len - labels_seq_len), value=ignore_idx
        )
    elif labels_seq_len > input_ids_seq_len:
        input_ids = F.pad(
            input_ids,
            (0, labels_seq_len - input_ids_seq_len),
            value=padding_idx,
        )
    collated_dict = {"input_ids": input_ids.long(),
            "labels": labels.long(),
            'attention_mask': attention_mask}
    other_info = {'oi_' + key: torch.tensor([x[key] for x in batch]) if pd.api.types.is_numeric_dtype(batch[0][key]) else [x[key] for x in batch] for key in batch[0].keys() if
                  key not in collated_dict.keys()}
    collated_dict = {**collated_dict, **other_info}
    return collated_dict
