import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model.layers import BiDAFAttention
from model.layers import MaskedLSTMEncoder
from model.bidaf_embedding_layer import BiDAFEmbeddingLayer
from utils.func_utils import length_to_mask
from utils.func_utils import masked_softmax


class Model(nn.Module):

    def __init__(self, params):
        """
        :param params:
            word_vocab:          list
            word_vocab_size:     int
            word_embedding_path  str
        """
        super().__init__()
        # embedding layer
        self.embedding_layer = BiDAFEmbeddingLayer(params)
        # attention layer
        E = params.bidaf_embedding_dim
        self.attention_layer = BiDAFAttention(hidden_size=E * 2)
        # model layer
        self.lstm = MaskedLSTMEncoder(
            input_size=8 * E,
            hidden_size=E,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=params.lstm_dropout
        )
        # output layer
        self.p1_w_g = nn.Linear(E * 8, 1, False)
        self.p1_w_m = nn.Linear(E * 2, 1, False)
        self.p2_w_g = nn.Linear(E * 8, 1, False)
        self.p2_w_m = nn.Linear(E * 2, 1, False)
        self.output_lstm = MaskedLSTMEncoder(
            input_size=2 * E,
            hidden_size=E,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
            dropout=params.lstm_dropout
        )

    def forward(self, batch):
        """
        B: batch_size
        T: length of context
        J: length of query
        E: embedding dim
        context:    B x T
        query:    B x J
        :param batch       data batch
        :return:
                output:    B x T x 2E
                contextual information about the word with respect to the entire context
                paragraph and the query
        """

        (context_word, c_lens), context_char = batch['c_word'], batch['c_char']
        (query_word, q_lens), query_char = batch['q_word'], batch['q_char']

        # B x T x 2E
        c = self.embedding_layer(context_word, context_char, c_lens)
        # B x J x 2E
        q = self.embedding_layer(query_word, query_char, q_lens)
        # B x T
        c_mask = length_to_mask(c_lens).cuda()
        q_mask = length_to_mask(q_lens).cuda()
        # B x T x 8E
        g = self.attention_layer(c, q, c_mask, q_mask)
        # B x T x 2E
        m = self.lstm(g, c_lens)
        # B x T
        p1_pred = (self.p1_w_g(g) + self.p1_w_m(m)).squeeze()
        p1_pred = masked_softmax(p1_pred, c_mask, log_softmax=True)
        # B x T x 2E
        m2 = self.output_lstm(m, c_lens)
        # B x T
        p2_pred = (self.p2_w_g(g) + self.p2_w_m(m2)).squeeze()
        p2_pred = masked_softmax(p2_pred, c_mask, log_softmax=True)
        return p1_pred, p2_pred

    def loss_fn(self, outputs, labels):
        p1_pred, p2_pred = outputs
        p1_stan, p2_stan = torch.cuda.LongTensor(labels[0]), torch.cuda.LongTensor(labels[1])
        return F.nll_loss(p1_pred, p1_stan) + F.nll_loss(p2_pred, p2_stan)

    def outputs_to_pred(self, outputs, enable_soft_prediction=True):
        """
        :param outputs: B x L
        :return: list of (s, t)
        """
        p1_pred, p2_pred = outputs
        if enable_soft_prediction:
            pred1, pred2 = discretize(p1_pred, p2_pred)
        else:
            pred1 = np.argmax(p1_pred, axis=1)
            pred2 = np.argmax(p2_pred, axis=1)
        pred = [(x, y) for x, y in zip(pred1, pred2)]
        return pred

    def accuracy(self, outputs, labels):
        pred = self.outputs_to_pred(outputs)
        acc = sum([x == y for x, y in zip(pred, zip(labels[0], labels[1]))])/float(len(labels[0]))
        return acc


def discretize(p_start, p_end, max_len=15, no_answer=False):
    """Discretize soft predictions to get start and end indices.
    Choose the pair `(i, j)` of indices that maximizes `p1[i] * p2[j]`
    subject to `i <= j` and `j - i + 1 <= max_len`.
    Args:
        p_start (torch.Tensor): Soft predictions for start index.
            Shape (batch_size, context_len).
        p_end (torch.Tensor): Soft predictions for end index.
            Shape (batch_size, context_len).
        max_len (int): Maximum length of the discretized prediction.
            I.e., enforce that `preds[i, 1] - preds[i, 0] + 1 <= max_len`.
        no_answer (bool): Treat 0-index as the no-answer prediction. Consider
            a prediction no-answer if `preds[0, 0] * preds[0, 1]` is greater
            than the probability assigned to the max-probability span.
    Returns:
        start_idxs (torch.Tensor): Hard predictions for start index.
            Shape (batch_size,)
        end_idxs (torch.Tensor): Hard predictions for end index.
            Shape (batch_size,)
    """
    p_start, p_end = torch.from_numpy(p_start).exp(), torch.from_numpy(p_end).exp()
    if p_start.min() < 0 or p_start.max() > 1 \
            or p_end.min() < 0 or p_end.max() > 1:
        raise ValueError('Expected p_start and p_end to have values in [0, 1] {}, {}, {}, {}'.format(p_start.min(),
                p_start.max(), p_end.min(), p_end.max()))

    # Compute pairwise probabilities
    p_start = p_start.unsqueeze(dim=2)
    p_end = p_end.unsqueeze(dim=1)
    p_joint = torch.matmul(p_start, p_end)  # (batch_size, c_len, c_len)

    # Restrict to pairs (i, j) such that i <= j <= i + max_len - 1
    c_len, device = p_start.size(1), p_start.device
    is_legal_pair = torch.triu(torch.ones((c_len, c_len), device=device))
    is_legal_pair -= torch.triu(torch.ones((c_len, c_len), device=device),
                                diagonal=max_len)
    if no_answer:
        # Index 0 is no-answer
        p_no_answer = p_joint[:, 0, 0].clone()
        is_legal_pair[0, :] = 0
        is_legal_pair[:, 0] = 0
    else:
        p_no_answer = None
    p_joint *= is_legal_pair

    # Take pair (i, j) that maximizes p_joint
    max_in_row, _ = torch.max(p_joint, dim=2)
    max_in_col, _ = torch.max(p_joint, dim=1)
    start_idxs = torch.argmax(max_in_row, dim=-1)
    end_idxs = torch.argmax(max_in_col, dim=-1)

    if no_answer:
        # Predict no-answer whenever p_no_answer > max_prob
        max_prob, _ = torch.max(max_in_col, dim=-1)
        start_idxs[p_no_answer > max_prob] = 0
        end_idxs[p_no_answer > max_prob] = 0

    return start_idxs, end_idxs


if __name__ == '__main__':
    from utils.model_utils import Params
    params = Params('../model_dir/configs.json')
    params.update('../data/dataset_configs.json')
    from model.data_loader import DataLoader
    data_loader = DataLoader(params)
    _model = Model(params)
    print(_model)
