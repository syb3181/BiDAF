import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.layers import BiDAFAttention
from model.layers import MaskedLSTMEncoder
from model.bidaf_embedding_layer import BiDAFEmbeddingLayer
from utils.func_utils import length_to_mask
from utils.func_utils import masked_softmax

from model.eval_func import outputs_to_preds
from model.eval_func import preds_to_answers
from model.eval_func import metric_max_over_ground_truths
from model.eval_func import f1_score
from model.eval_func import exact_match_score


class Model(nn.Module):

    @staticmethod
    def exact_match_score(outputs, data_batch):
        preds = outputs_to_preds(outputs)
        preds = preds_to_answers(preds, data_batch['tkd_c'])
        scores = [metric_max_over_ground_truths(exact_match_score, pred, data_batch['gts'][i]) for i, pred in enumerate(preds)]
        return np.mean(scores)

    @staticmethod
    def f1_score(outputs, data_batch):
        preds = outputs_to_preds(outputs)
        preds = preds_to_answers(preds, data_batch['tkd_c'])
        scores = [metric_max_over_ground_truths(f1_score, pred, data_batch['gts'][i]) for i, pred in enumerate(preds)]
        return np.mean(scores)

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


if __name__ == '__main__':
    from utils.model_utils import Params
    params = Params('../model_dir/configs.json')
    params.update('../data/dataset_configs.json')
    from model.data_loader import DataLoader
    data_loader = DataLoader(params)
    _model = Model(params)
    print(_model)
