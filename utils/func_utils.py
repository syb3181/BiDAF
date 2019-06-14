import numpy as np
import random
import torch
import torch.nn.functional as F


def load_embedding_dict(file_path, dim=300):
    """
    :param file_path: the file path of embedding dict
    :return: dict[key=word, value=embedding_matrix]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        ret = {}
        for line in f.readlines():
            row = line.split()
            ret[''.join(row[:-dim])] = [float(x) for x in row[-dim:]]
    return ret


def build_embedding_matrix(em_dict: dict, tks: list):
    """
    :param em_dict: embedding_dict
    :param tks: list of vocabulary from text
    :return: FIELD, embedding matrix
    """
    vocab_size = len(tks)
    em_size = len(em_dict[next(iter(em_dict))])
    w = np.zeros((vocab_size, em_size), dtype='float32')
    for i, word in enumerate(tks):
        if word in em_dict:
            w[i] = em_dict[word]
    return w


def length_to_mask(length, max_len=None, dtype=None):
    """
    length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.
    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.
    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)
    return probs


def random_shuffle(data, random_state=None):
    random.seed(random.getstate() if random_state is None else random_state)
    random.shuffle(data)
