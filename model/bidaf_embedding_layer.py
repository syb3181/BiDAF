import torch
import torch.nn as nn
from utils.func_utils import load_embedding_dict
from utils.func_utils import build_embedding_matrix
from model.layers import HighwayEncoder
from model.layers import MaskedLSTMEncoder


class BiDAFEmbeddingLayer(nn.Module):

    def __init__(self, params):
        super().__init__()
        hidden_size = params.hidden_size
        self.word_embedding = nn.Embedding(params.word_vocab_size, params.word_embedding_dim, padding_idx=1)
        self.__load_embedding(params)
        self.word_embedding_dropout = nn.Dropout(params.dropout)
        self.embedding_dim = params.word_embedding_dim
        self.proj = nn.Linear(self.embedding_dim, hidden_size, bias=False)
        self.highway_network = HighwayEncoder(2, hidden_size)
        self.encoder = MaskedLSTMEncoder(hidden_size, hidden_size, True, True, 1, params.dropout)

    def __load_embedding(self, params):
        embedding_dict = load_embedding_dict(params.word_embedding_path)
        w = build_embedding_matrix(embedding_dict, params.word_vocab)
        self.word_embedding.from_pretrained(torch.from_numpy(w).cuda(), freeze=False)

    def forward(self, x_word, x_lens):
        """
        :param x_word:      B x L
        :param x_lens        B x L
        :return:  B x L x 2E
        """
        # B x L x E
        x = self.word_embedding(x_word)
        x = self.word_embedding_dropout(x)
        x = self.proj(x)
        x = self.highway_network(x)
        x = self.encoder(x, x_lens)
        return x
