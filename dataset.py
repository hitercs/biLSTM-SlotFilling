#-*- encoding: utf-8-*-
from torch.utils.data import Dataset
import settings
import codecs
import torch
from torch.autograd import Variable

class Pair(object):
    def __init__(self, word_seq, label_seq):
        self.word_seq = word_seq
        self.label_seq = label_seq
        self.length = len(self.word_seq)
        assert(len(self.word_seq) == len(self.label_seq))
        self.word_seq_id = []
        self.label_seq_id = []

class NERDataset(Dataset):
    def __init__(self, fn, vocab):
        self.vocab = vocab
        self.all_dataset = []
        self.data_count = 0
        self.load_dataset(fn)

    def parse_line(self, line):
        if line != "":
            words = line.strip().split()
            tok, label = words[0], words[1]
            return tok, label
        return None, None

    def load_dataset(self, fn):
        with codecs.open(fn, encoding='utf-8', mode='r', buffering=settings.read_buffer_size) as fp:
            all_lines = fp.readlines()
            length = len(all_lines)
            i = 0
            while i < length:
                tmp_word_seq = []
                tmp_label_seq = []
                while i < length and all_lines[i].strip():
                    tmp_word_seq.append(self.parse_line(all_lines[i])[0])
                    tmp_label_seq.append(self.parse_line(all_lines[i])[1])
                    i += 1
                p = Pair(tmp_word_seq, tmp_label_seq)
                p.word_seq_id = [self.vocab.get_src_idx(w) for w in p.word_seq]
                p.label_seq_id = [self.vocab.get_trg_idx(w) for w in p.label_seq]
                self.all_dataset.append(p)
                self.data_count += 1
                i += 1

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        return self.all_dataset[index]

class Batch(object):
    def __init__(self):
        self.word_seq_id = None
        self.label_seq_id = None
        self.length = []

    @staticmethod
    def get_batch(data, pad_id, sorted_with_batch=True, is_volatile=False):
        rank_idx = None
        batch_size = len(data)
        if sorted_with_batch:
            # sorted data
            rank_idx = sorted(range(len(data)), key=lambda x : data[x].length, reverse=True)
            data = sorted(data, key=lambda x : x.length, reverse=True)
        batch = Batch()
        max_len = max(data, key=lambda x:x.length).length
        word_seq_id = torch.LongTensor(batch_size, max_len)
        label_seq_id = torch.LongTensor(batch_size, max_len)

        for batch_id in range(batch_size):
            batch.length.append(data[batch_id].length)
            tmp_len = data[batch_id].length
            for seq_id in range(max_len):
                if seq_id < tmp_len:
                    word_seq_id[batch_id, seq_id] = data[batch_id].word_seq_id[seq_id]
                    label_seq_id[batch_id, seq_id] = data[batch_id].label_seq_id[seq_id]
                else:
                    word_seq_id[batch_id, seq_id] = pad_id
                    label_seq_id[batch_id, seq_id] = pad_id

        batch.word_seq_id = Variable(word_seq_id, volatile=is_volatile)
        batch.label_seq_id = Variable(label_seq_id, volatile=is_volatile)
        if torch.cuda.is_available():
            batch.word_seq_id = batch.word_seq_id.cuda()
            batch.label_seq_id = batch.label_seq_id.cuda()
        return batch, rank_idx