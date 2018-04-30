#-*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import settings

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, output_dim, dropout=0.75):
        super(BiLSTM, self).__init__()

        self.word_embed = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=embed_size,
                                       padding_idx=settings.pad_idx)

        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)

        self.linear = nn.Linear(2*hidden_size, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, batch_seq_id, input_length=None):
        """

        :param batch_seq_id: Tensor Variable: [batch, max_seq_len], tensor for sequence id
        :param input_length: list[int]: list of sequences lengths of each sequence
        :return:
        """
        """Embedding Layer"""
        embedding = self.word_embed(batch_seq_id)           #[batch, max_seq_len, embed_size]
        if input_length:
            input_lstm = nn.utils.rnn.pack_padded_sequence(embedding, input_length, batch_first=True)
        else:
            input_lstm = embedding

        """LSTM Layer"""
        lstm_outputs, _ = self.encoder(input_lstm)          #[batch, max_seq_len, 2*hidden_size]

        if input_length:
            lstm_outputs, _ = nn.utils.rnn.pad_packed_sequence(lstm_outputs, batch_first=True)

        """Prediction Layer"""
        predicted_log_probs = self.log_softmax(self.linear(lstm_outputs))
        return predicted_log_probs                          #[batch, max_seq_len, trg_vocab_size]