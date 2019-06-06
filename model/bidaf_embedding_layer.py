import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.func_utils import load_embedding_dict
from utils.func_utils import build_embedding_matrix
from model.layers import HighwayEncoder
from model.layers import MaskedLSTMEncoder


class BiDAFEmbeddingLayer(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.params = params
        # word level
        self.word_embedding = nn.Embedding(
            padding_idx=1,
            embedding_dim=params.word_embedding_dim,
            num_embeddings=params.word_vocab_size
        )
        embedding_dict = load_embedding_dict(params.word_embedding_path)
        w = build_embedding_matrix(embedding_dict, params.word_vocab)
        self.word_embedding.from_pretrained(torch.from_numpy(w).cuda(), freeze=False)
        # char level
        self.char_embedding = nn.Embedding(
            padding_idx=1,
            embedding_dim=params.char_embedding_dim,
            num_embeddings=params.char_vocab_size
        )
        self.dropout = nn.Dropout(params.char_cnn_dropout)
        self.char_conv = nn.Conv2d(
            in_channels=1,
            out_channels=params.char_cnn_output_channels,
            kernel_size=(params.char_embedding_dim, params.char_cnn_channel_width)
        )
        self.highway_network = HighwayEncoder(2, params.embedding_lstm_input_size - 10)
        self.lstm = MaskedLSTMEncoder(
            input_size=params.embedding_lstm_input_size - 10,
            hidden_size=params.bidaf_embedding_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
            dropout=params.embedding_lstm_dropout
        )

    def forward(self, x_word, x_char, x_lens):
        """
        WL: params.max_word_len
        CE: params.char_embedding_dim
        CO: params.char_cnn_output_channels
        E:  params.bidaf_embedding_size
        :param x_word:      B x L
        :param x_char:      B x (L x WL)
        :param x_lens        B x L
        :return:  B x L x 2E
        """
        def char_embedding_layer(x):
            """
            :param x: B x (L x WL)
            :return: B x L x CO
            """
            B = x.size(0)
            # B x L x WL
            x = x.view(B, -1, WL)
            # B x L x WL x CE
            x = self.dropout(self.char_embedding(x))
            # B x L x CE x WL
            x = x.transpose(-1, -2)
            # (B x L) x CI(1) x CE x WL
            x = x.view(-1, CE, WL).unsqueeze(1)
            # (B x L) x CO x w_out(1) x h_out
            x = self.char_conv(x).squeeze()
            # (B x L) x CO
            x = F.max_pool1d(x, x.size(2)).squeeze()
            # B x L x CO
            x = x.view(B, -1, CO)
            return x

        WL = self.params.max_word_len
        CE = self.params.char_embedding_dim
        CO = self.params.char_cnn_output_channels
        # B x L x E
        word_e = self.word_embedding(x_word)
        # B x L x CO
        char_e = char_embedding_layer(x_char)
        # B x L x (E + CO)
        lstm_input = word_e
        # lstm_input = torch.cat((word_e, char_e), dim=2)
        lstm_input = self.highway_network(lstm_input)
        lstm_output = self.lstm(lstm_input, x_lens)
        return lstm_output
