#-*- encoding: utf-8 -*-
import codecs
import settings
from util import Util

class BiVocab(object):
    def __init__(self, src_vocab, trg_vocab):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_vocab_size = src_vocab.vocab_size
        self.trg_vocab_size = trg_vocab.vocab_size
        self.pad_id = self.trg_vocab.get_idx(settings.PAD)
        self.unk_id = self.trg_vocab.get_idx(settings.UNK)

    def get_src_word(self, idx):
        return self.src_vocab.get_word(idx)

    def get_trg_word(self, idx):
        return self.trg_vocab.get_word(idx)

    def get_src_idx(self, w):
        return self.src_vocab.get_idx(w)

    def get_trg_idx(self, w):
        return self.trg_vocab.get_idx(w)


class Vocab(object):
    def __init__(self, vocab_size, vocab_fn):
        self.word2idx = dict()
        self.idx2word = dict()
        self.vocab_size = vocab_size
        self.build_vocab(vocab_fn)

    def build_vocab(self, vocab_fn):
        with codecs.open(vocab_fn, encoding='utf-8', mode='r', buffering=settings.read_buffer_size) as fp:
            for line in fp:
                word, idx, _ = line.strip().split()
                Util.add_vocab(self.word2idx, word, int(idx))
                Util.add_vocab(self.idx2word, int(idx), word)

    def get_idx(self, word):
        if not word in self.word2idx:
            return self.word2idx[settings.UNK]
        if self.word2idx[word] > self.vocab_size - 1:
            return self.word2idx[settings.UNK]
        return self.word2idx[word]

    def get_word(self, idx):
        if idx > self.vocab_size - 1:
            return settings.UNK
        return self.idx2word[idx]