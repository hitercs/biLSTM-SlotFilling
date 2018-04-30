# encoding: utf-8
import argparse
import os
import codecs
import settings

class Util:
    @staticmethod
    def add_vocab(vocab, key):
        if not key in vocab:
            vocab[key] = 1
        else:
            vocab[key] += 1

class Vocab(object):
    def __init__(self):
        self.src_vocab = dict()
        self.trg_vocab = dict()


    def word_freq(self, in_dir):
        with codecs.open(os.path.join(in_dir, "atis.train.txt"), encoding='utf-8', mode='r', buffering=settings.read_buffer_size) as src_fp:
            all_lines = src_fp.readlines()
            length = len(all_lines)
            i = 0
            while i < length:
                if all_lines[i].strip():
                    words = all_lines[i].split()
                    Util.add_vocab(self.src_vocab, words[0])
                    Util.add_vocab(self.trg_vocab, words[1])
                i += 1

    def print_vocab(self, out_dir):
        sorted_src_vocab = sorted(self.src_vocab.items(), key=lambda x: x[1], reverse=True)
        idx = 2
        with codecs.open(os.path.join(out_dir, "src_vocab.txt"), encoding='utf-8', mode='w', buffering=settings.write_buffer_size) as re_src_fp:
            re_src_fp.write(u"{}\t{}\t{}\n".format(settings.PAD, settings.PAD_ID, 1))
            re_src_fp.write(u"{}\t{}\t{}\n".format(settings.UNK, settings.UNK_ID, 1))

            for x in sorted_src_vocab:
                if x[1] >= 0 and x[0] and x[0] != "<UNK>":
                    re_src_fp.write(u"{}\t{}\t{}\n".format(x[0], idx, x[1]))
                    idx += 1

        sorted_trg_vocab = sorted(self.trg_vocab.items(), key=lambda x: x[1], reverse=True)
        idx = 2
        with codecs.open(os.path.join(out_dir, "trg_vocab.txt"), encoding='utf-8', mode='w', buffering=settings.write_buffer_size) as re_trg_fp:
            re_trg_fp.write(u"{}\t{}\t{}\n".format(settings.PAD, settings.PAD_ID, 1))
            re_trg_fp.write(u"{}\t{}\t{}\n".format(settings.UNK, settings.UNK_ID, 1))

            for x in sorted_trg_vocab:
                if x[1] >= 0 and x[0]:
                    re_trg_fp.write(u"{}\t{}\t{}\n".format(x[0], idx, x[1]))
                    idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-in-dir", "--in-dir", type=str, default=r"/home/shuang/Workspace/Courses/ComputationalSemantic/project/project/data")
    parser.add_argument("-out-dir", "--out-dir", type=str, default=r"/home/shuang/Workspace/Courses/ComputationalSemantic/project/project/data")
    args = parser.parse_args()

    v = Vocab()
    v.word_freq(args.in_dir)
    v.print_vocab(args.out_dir)