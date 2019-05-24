import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model.bidaf_embedding_layer import BiDAFEmbeddingLayer


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
        # model_layer
        self.lstm = nn.LSTM(
            input_size=8*E,
            hidden_size=E,
            batch_first=True,
            bidirectional=True,
            # num_layers=2,
            # dropout=params.lstm_dropout,
        )
        # output_layer
        self.linear0 = nn.Linear(10 * E, 1)
        self.output_lstm = nn.LSTM(
            input_size=2*E,
            hidden_size=E,
            batch_first=True,
            bidirectional=True
        )
        self.linear1 = nn.Linear(10 * E, 1)

    def forward(self, context_word, context_char, query_word, query_char):
        """
        B: batch_size
        T: length of context
        J: length of query
        E: embedding dim
        :param     context:    B x T
        :param       query:    B x J
        :return:
                output:    B x T x 2E
                contextual information about the word with respect to the entire context
                paragraph and the query
        """
        def att_flow_layer(c, q):
            """
            borrowed from
            https://github.com/galsang/BiDAF-pytorch/blob/master/model/model.py
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
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

        # B x T x 2E
        c = self.embedding_layer(context_word, context_char)
        # B x J x 2E
        q = self.embedding_layer(query_word, query_char)
        # B x T x 8E
        g = att_flow_layer(c, q)
        # B x T x 2E
        m, _ = self.lstm(g)
        # B x T x 10E
        g_m = torch.cat((g, m), dim=2)
        # B x T
        p1_pred = self.linear0(g_m).squeeze()
        p1_pred = F.softmax(p1_pred, dim=1)
        # B x T x 2E
        m2, _ = self.output_lstm(m)
        # B x T x 10E
        g_m2 = torch.cat((g, m2), dim=2)
        # B x T
        p2_pred = self.linear1(g_m2).squeeze()
        p2_pred = F.softmax(p2_pred, dim=1)
        return p1_pred, p2_pred

    def loss_fn(self, outputs, labels):
        p1_pred, p2_pred = outputs
        # 2B
        reshaped_labels = torch.LongTensor([x for x, y in labels] + [y for x, y in labels]).cuda()
        # 2B x T
        p_pred = torch.cat((p1_pred, p2_pred), dim=0)
        return F.cross_entropy(p_pred, reshaped_labels)

    def accuracy(self, outputs, labels):
        p1_pred, p2_pred = outputs
        pred1 = np.argmax(p1_pred, axis=1)
        pred2 = np.argmax(p2_pred, axis=1)
        pred = [(x, y) for x, y in zip(pred1, pred2)]
        acc = sum([x == y for x, y in zip(pred, labels)])/float(len(labels))
        return acc
