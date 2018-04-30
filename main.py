#-*- encoding: utf-8 -*-
import model
import dataset
import torch
import torch.nn as nn
import settings
import random
import vocab
from torch.utils.data import DataLoader
from dataset import Batch
import argparse
import os
import codecs
import subprocess

def train(model_, batch, criteria, optimizer):
    model_.train(True)
    predicted_log_probs = model_(batch.word_seq_id, batch.length)
    seq_len = predicted_log_probs.size(1)
    loss = 0
    for i in range(seq_len):
        loss +=criteria(predicted_log_probs[:,i,:], batch.label_seq_id[:,i])
    model_.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.data[0]

def evaluate(eval_script, output_path):
    script_args = ['perl', eval_script]
    with codecs.open(output_path, encoding='utf-8', mode='r', buffering=settings.read_buffer_size) as fp:
        p = subprocess.Popen(script_args, stdout=subprocess.PIPE, stdin=fp)
        std_results = p.stdout.readlines()
        std_results = str(std_results[1]).split()
        precision = float(std_results[3].replace('%;', ''))
        recall = float(std_results[5].replace('%;', ''))
        f1 = float(std_results[7].replace('%;', '').replace("\\n'", ''))
        os.remove(output_path)
    return precision, recall, f1

def test(model_, batch, sorted_idx, vocab):
    model_.train(False)
    predicted_probs = model_(batch.word_seq_id, batch.length)
    labels = torch.topk(predicted_probs, 1, dim=-1)
    gen_labels = []
    batch_size = labels[1].size(0)
    for i in range(batch_size):
        gen_labels.append(" ".join([vocab.get_trg_word(idx) for idx in labels[1][i].squeeze().data]))
    recover_order_gen_results = [None for _ in range(batch_size)]
    for i in range(batch_size):
        recover_order_gen_results[sorted_idx[i]] = gen_labels[i]
    return recover_order_gen_results

def test_results(model_, eval_script, dev_data_loader, output_path, vocab, dev_ner_data):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    gen_results = []
    for _data in dev_data_loader:
        batch, sorted_idx = Batch.get_batch(_data, settings.pad_idx, sorted_with_batch=True, is_volatile=True)
        recover_results = test(model_, batch, sorted_idx, vocab)
        gen_results = gen_results + recover_results
    
    j = 0
    with codecs.open(os.path.join(output_path, "output.txt"), encoding='utf-8', mode='w', buffering=settings.write_buffer_size) as fp:
        for item in dev_ner_data.all_dataset:
            for i in range(len(item.word_seq)):
                fp.write("{} - - {} {}\n".format(item.word_seq[i], item.label_seq[i], gen_results[j].split()[i]))
            fp.write("\n")
            j += 1
    p, r, f1 = evaluate(eval_script, os.path.join(output_path, "output.txt"))
    return p, r, f1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-path", type=str, default=r"./data")
    parser.add_argument("--eval-script", "--eval-script", type=str, default=r"./eval/conlleval.pl")
    args = parser.parse_args()

    lstm_model = model.BiLSTM(settings.src_vocab_size, settings.word_embed_size,
                         settings.hidden_size, settings.num_layers, settings.trg_vocab_size)
    weights = torch.ones(settings.trg_vocab_size)
    weights[settings.pad_idx] = 0
    criteria = nn.NLLLoss(weight=weights, size_average=True)
    optimizer = torch.optim.Adam(lstm_model.parameters())

    torch.manual_seed(settings.seed)
    random.seed(settings.seed)

    if torch.cuda.is_available():
        lstm_model.cuda()
        criteria.cuda()
        torch.cuda.manual_seed(settings.seed)

    for param in lstm_model.parameters():
        param.data.uniform_(-0.08, 0.08)

    # Load Voccab
    src_vocab = vocab.Vocab(settings.src_vocab_size, os.path.join(args.path, "src_vocab.txt"))
    trg_vocab = vocab.Vocab(settings.trg_vocab_size, os.path.join(args.path, "trg_vocab.txt"))
    biVocab = vocab.BiVocab(src_vocab, trg_vocab)

    # Load Dataset
    train_ner_data = dataset.NERDataset(os.path.join(args.path, "atis.train.txt"), biVocab)
    train_ner_data_loader = DataLoader(train_ner_data, batch_size=settings.batch_size, shuffle=True, collate_fn=lambda x:x)
    dev_ner_data = dataset.NERDataset(os.path.join(args.path, "atis.test.txt"), biVocab)
    dev_ner_data_loader = DataLoader(dev_ner_data, batch_size=settings.batch_size, shuffle=False, collate_fn=lambda x:x)

    epoch = 100
    steps_per_epoch = int(len(train_ner_data) / settings.batch_size)
    best_test_f1 = 0.0

    for i in range(epoch):
        sum_loss_per_epoch = 0
        step = 0
        for _data in train_ner_data_loader:
            step += 1
            batch, _ = Batch.get_batch(_data, settings.pad_idx)
            loss = train(lstm_model, batch, criteria, optimizer)
            sum_loss_per_epoch += loss
        
        p, r, f1 = test_results(lstm_model, args.eval_script, dev_ner_data_loader, os.path.join(args.path, "Output"), biVocab, dev_ner_data)
        if f1 > best_test_f1:
            best_test_f1 = f1
        print("loss = {:.4f}, f1 = {:.4f} %, best f1 = {:.4f} %, epoch = {}".format(sum_loss_per_epoch/step, f1, best_test_f1, i))