import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import numpy as np

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
        self.att_weight_c = nn.Linear(E * 2, 1)
        self.att_weight_q = nn.Linear(E * 2, 1)
        self.att_weight_cq = nn.Linear(E * 2, 1)
        # model layer
        self.lstm = nn.LSTM(
            input_size=8*E,
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
        self.output_lstm = nn.LSTM(
            input_size=2*E,
            hidden_size=E,
            batch_first=True,
            bidirectional=True
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
        def vanilla_att_flow_layer(c, q):
            """
            :param c: B x T x 2E
            :param q: B x J x 2E
            :return: B x T x ?E
            """
            # (B, T, 2E) bmm (B, 2E, J) -> (B, T, J)
            S = torch.bmm(c, q.transpose(1, 2))
            # (B, T, J)
            A = F.softmax(S, dim=2)
            # (B, T, J) bmm (B, J, 2E) -> (B, T, 2E)
            U_tilde = torch.bmm(A, q)
            ret = torch.cat((U_tilde, c), dim=2)
            return ret

        def att_flow_layer(c, q, c_mask, q_mask):
            """
            :param c: B x T x 2E
            :param q: B x J x 2E
            :return:  B x T x 8E
            """
            c_len = c.size(1)
            q_len = q.size(1)
            cq = []
            for i in range(q_len):
                # (batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                # (batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)

            # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq
            s = s.masked_fill(1 - c_mask.unsqueeze(2), -1e9)
            s = s.masked_fill(1 - q_mask.unsqueeze(1), -1e9)
            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze()
            # (batch, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        (context_word, c_lens), context_char = batch.c, batch.c_char.cuda()
        (query_word, q_lens), query_char = batch.q, batch.q_char.cuda()

        # B x T x 2E
        c = self.embedding_layer(context_word.cuda(), context_char.cuda(), c_lens)
        # B x J x 2E
        q = self.embedding_layer(query_word.cuda(), query_char.cuda(), q_lens)
        # B x T
        c_mask = length_to_mask(c_lens).cuda()
        q_mask = length_to_mask(q_lens).cuda()
        # B x T x 8E
        g = att_flow_layer(c, q, c_mask, q_mask)
        # g = vanilla_att_flow_layer(c, q)
        # B x T x 2E
        m, _ = self.lstm(pack_padded_sequence(g, c_lens, enforce_sorted=False, batch_first=True))
        m, _ = pad_packed_sequence(m, batch_first=True)
        # B x T
        p1_pred = (self.p1_w_g(g) + self.p1_w_m(m)).squeeze()
        p1_pred = masked_softmax(p1_pred, c_mask)
        # B x T x 2E
        m2, _ = self.output_lstm(pack_padded_sequence(m, c_lens, enforce_sorted=False, batch_first=True))
        m2, _ = pad_packed_sequence(m2, batch_first=True)
        # B x T
        p2_pred = (self.p2_w_g(g) + self.p2_w_m(m2)).squeeze()
        p2_pred = masked_softmax(p2_pred, c_mask)
        return p1_pred, p2_pred

    def loss_fn(self, outputs, labels):
        p1_pred, p2_pred = outputs
        # 2B
        reshaped_labels = torch.LongTensor([x for x, y in labels] + [y for x, y in labels]).cuda()
        # 2B x T
        p_pred = torch.cat((p1_pred, p2_pred), dim=0)
        ce_loss = F.cross_entropy
        return ce_loss(p_pred, reshaped_labels)

    def outputs_to_pred(self, outputs):
        """
        :param outputs: B x L
        :return: list of (s, t)
        """
        p1_pred, p2_pred = outputs
        pred1 = np.argmax(p1_pred, axis=1)
        pred2 = np.argmax(p2_pred, axis=1)
        pred = [(x, y) for x, y in zip(pred1, pred2)]
        return pred

    def accuracy(self, outputs, labels):
        pred = self.outputs_to_pred(outputs)
        acc = sum([x == y for x, y in zip(pred, labels)])/float(len(labels))
        return acc


if __name__ == '__main__':
    from utils.model_utils import Params
    params = Params('../model_dir/configs.json')
    params.update('../data/dataset_configs.json')
    from model.data_loader import DataLoader
    import os
    data_loader = DataLoader(os.path.join('../data', 'train', 'small.json'), params)
    _model = Model(params)
    print(_model)
