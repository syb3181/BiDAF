import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.func_utils import load_embedding_dict
from utils.func_utils import build_embedding_matrix


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
        self.word_embedding.from_pretrained(torch.from_numpy(w).cuda(), freeze=True)
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
        self.lstm = nn.LSTM(
            input_size=params.embedding_lstm_input_size,
            hidden_size=params.bidaf_embedding_dim,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x_word, x_char):
        """
        WL: params.max_word_len
        CE: params.char_embedding_dim
        CO: params.char_cnn_output_channels
        E:  params.bidaf_embedding_size
        :param x_word:      B x L
        :param x_char:      B x (L x WL)
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
            # B x L x CE x WL
            x = self.dropout(self.char_embedding(x))
            # (B x L) x 1 x CE x WL
            x = x.view(-1, CE, WL).unsqueeze(1)
            # (B x L) x CO x conv_len
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
        lstm_input = torch.cat((word_e, char_e), dim=2)
        lstm_output, _ = self.lstm(lstm_input)
        return lstm_output
