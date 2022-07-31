import argparse
import datetime
import os
import torch
import torch.nn as nn
import numpy as np
from torch import optim
import pandas as pd
import torch.utils.data as Data
from random import *
from preprocess import DataSet
from utils import next_batch, get_evalution, make_exchange_matrix, Loss_Function

import config as gl

# python bert_run_update.py
# --device 1 --bs 256 --epoch 100 --loss loss --datalen 5
# --train_dataset train_traj_first_10_day --test_dataset test_traj_5
parser = argparse.ArgumentParser()
parser.add_argument('--device', default=1, type=int, help='train device')
parser.add_argument('--bs', default=256, type=int, help='batch size')
parser.add_argument('--epoch', default=1, type=int, help='epoch size')
parser.add_argument('--loss', default='loss', type=str, help='loss or spatial_loss')
parser.add_argument('--datalen', default=5, type=int, help='datalen')
parser.add_argument('--train_dataset', default="train_traj_5", type=str, help='test dataset')
parser.add_argument('--test_dataset', default="test_traj_5", type=str, help='test dataset')
parser.add_argument('--d_model', default=1024, type=int, help='embed size')
parser.add_argument('--head', default=2, type=int, help='multi head num')
parser.add_argument('--layer', default=2, type=int, help='layer')

args = parser.parse_args()
device = 'cuda:%s' % args.device
batch_size = args.bs
epoch_size = args.epoch
loss_fun = args.loss  # loss or spatial_loss
datalen = args.datalen
train_dataset = args.train_dataset + ".h5"
test_dataset = args.test_dataset + ".h5"

d_model = args.d_model
head = args.head
layer = args.layer
gl._init()
gl.set_value('d_model', d_model)
gl.set_value('head', head)
gl.set_value('layer', layer)
gl.set_value('device', device)
pth_dic = 'pth_test_2'

max_pred = args.datalen  # max tokens of prediction
train_df = pd.read_hdf(os.path.join('data/Dataset Filtered 2 h5', train_dataset), key='data')
test_df = pd.read_hdf(os.path.join('data/Dataset Filtered 2 h5', test_dataset), key='data')
dataset = DataSet(train_df, test_df)
train_data = dataset.gen_train_data()  # [seq, user_index, day]
test_data = dataset.gen_test_data()  # [seq, masked_pos, masked_tokens, user_index, day]

train_word_list = list(
    set(str(train_data[i][0][j]) for i in range(len(train_data)) for j in range(len(train_data[i][0]))))
test_word_list = list(
    set(str(test_data[i][0][j]) for i in range(len(test_data)) for j in range(len(test_data[i][0]))))
test_masked_list = list(
    set(str(test_data[i][2][j]) for i in range(len(test_data)) for j in range(len(test_data[i][2]))))

word_list = []
word_list.extend(train_word_list)
word_list.extend(test_word_list)
word_list.extend(test_masked_list)
word_list = list(set(word_list))
word_list.sort()

word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
for i, w in enumerate(word_list):
    if w != '[PAD]' and w != '[MASK]':
        word2idx[w] = i + 4

idx2word = {i: w for i, w in enumerate(word2idx)}
vocab_size = len(word2idx)

train_token_list = list()
train_user_list = list()
train_day_list = list()
for sentence in train_data:
    seq, user_index, day = sentence
    arr = [word2idx[s] for s in seq]
    train_token_list.append(arr)
    train_user_list.append(user_index)
    train_day_list.append(day)

exchange_map = make_exchange_matrix(token_list=train_token_list, token_size=vocab_size)
exchange_map = torch.Tensor(exchange_map).to(device)


def make_train_data(token_list):
    total_data = []
    for i in range(len(token_list)):
        tokens_a_index = i  # sample every index in sentences
        tokens_a = token_list[tokens_a_index]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']]

        # MASK LM
        n_pred = max_pred
        # n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))  # 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word2idx['[CLS]'] and token != word2idx['[SEP]'] and token != word2idx[
                              '[PAD]']]  # candidate masked position
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = word2idx['[MASK]']  # make mask
            elif random() > 0.9:  # 10%
                index = randint(0, vocab_size - 1)  # random index in vocabulary
                while index < 4:  # can't involve 'CLS', 'SEP', 'PAD'
                    index = randint(0, vocab_size - 1)
                input_ids[pos] = index  # replace
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
        total_data.append([input_ids, masked_tokens, masked_pos])
    return total_data


def make_test_data(test_data):
    # [seq, masked_pos, masked_tokens, user_index, day]
    total_test_data = []
    for sentence in test_data:
        arr = [word2idx[s] for s in sentence[0]]
        user = sentence[3]
        arr = [word2idx['[CLS]']] + arr + [word2idx['[SEP]']]
        masked_pos = [pos + 1 for pos in sentence[1]]  # masked pos向后偏移一位
        masked_tokens = [word2idx[str(s)] for s in sentence[2]]
        day = sentence[4]
        total_test_data.append([arr, masked_tokens, masked_pos, user, day])
    return total_test_data


total_data = make_train_data(train_token_list)  # [input_ids, masked_tokens, masked_pos]
print("total length of train data is", str(len(total_data)))  #
input_ids, masked_tokens, masked_pos = zip(*total_data)
user_ids, day_ids = torch.LongTensor(train_user_list).to(device), torch.LongTensor(train_day_list).to(device)
input_ids, masked_tokens, masked_pos, = torch.LongTensor(input_ids).to(device), \
                                        torch.LongTensor(masked_tokens).to(device), \
                                        torch.LongTensor(masked_pos).to(device)


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, masked_tokens, masked_pos, user_ids, day_ids):
        self.input_ids = input_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
        self.user_ids = user_ids
        self.day_ids = day_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.user_ids[idx], self.day_ids[idx]


loader = Data.DataLoader(MyDataSet(input_ids, masked_tokens, masked_pos, user_ids, day_ids), batch_size, True)
from downstream.bert import BERT

model = BERT(vocab_size=vocab_size).to(device)
if loss_fun == "spatial_loss":
    criterion = Loss_Function()
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.Adadelta(model.parameters(), lr=0.001)
train_predict = []
train_truth = []

for epoch in range(epoch_size):
    train_predict, train_truth = [], []
    for i, (input_ids, masked_tokens, masked_pos, user_ids, day_ids) in enumerate(loader):
        logits_lm = model(input_ids, masked_pos, user_ids, day_ids)
        train_truth.extend(masked_tokens.flatten().cpu().data.numpy())
        train_predict.extend(logits_lm.data.max(2)[1].flatten().cpu().data.numpy())
        if loss_fun == "spatial_loss":
            loss_lm = criterion.Spatial_Loss(exchange_map, logits_lm.view(-1, vocab_size),
                                             masked_tokens.view(-1))  # for masked LM
        else:
            loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1))  # for masked LM

        loss_lm = (loss_lm.float()).mean()
        loss = loss_lm
        print('Epoch:', '%06d' % (epoch + 1), 'Iter:', '%06d' % (i + 1), 'loss =', '{:.6f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


torch.save({'model': model.state_dict()},
           pth_dic + '/data-%s-%s-batch%s-epoch%s-%s-%s.pth' % (
               args.train_dataset, args.test_dataset, batch_size, epoch_size, loss_fun,
               datetime.datetime.now().strftime("%Y%m%d%H")))

# state_dict = torch.load('pth/dataset-batch256-epoch50-loss-20220707_tem_without_pos.pth.pth')
# model.load_state_dict(state_dict['model'])

test_total_data = make_test_data(test_data)
print("total length of test data is", str(len(test_total_data)))
test_input_ids, test_masked_tokens, test_masked_pos, test_user_ids, test_day_ids = zip(*test_total_data)


def test(test_token_list, test_masked_tokens, test_masked_pos, test_user_ids, test_day_ids):
    masked_tokens = np.array(test_masked_tokens).reshape(-1)
    print(masked_tokens.shape)
    a = list(zip(test_token_list, test_masked_pos, test_user_ids, test_day_ids))
    predict_prob = torch.Tensor([]).to(device)

    for batch in next_batch(a, batch_size=64):
        # Value filled with num_loc stands for masked tokens that shouldn't be considered.
        batch_token_list, batch_masked_pos, batch_user_ids, batch_day_ids = zip(*batch)
        logits_lm = model(torch.LongTensor(batch_token_list).to(device),
                          torch.LongTensor(batch_masked_pos).to(device),
                          torch.LongTensor(batch_user_ids).to(device),
                          torch.LongTensor(batch_day_ids).to(device), )
        logits_lm = torch.topk(logits_lm, 100, dim=2)[1]
        predict_prob = torch.cat([predict_prob, logits_lm], dim=0)

    ground_truth_origin = [str(idx2word[s]) for s in masked_tokens]
    predict_loc = predict_prob[:, :, 0].flatten().cpu().data.numpy()
    predict_loc_origin = [str(idx2word[s]) for s in predict_loc]
    wirte_csv(ground_truth_origin, 'ground_truth_origin_%s_%s_%s' % (args.train_dataset, args.test_dataset, loss_fun))
    wirte_csv(predict_loc_origin, 'predict_loc_origin_%s_%s_%s' % (args.train_dataset, args.test_dataset, loss_fun))

    accuracy_score, fuzzy_score, top3_score, top5_score, top10_score, top30_score, top50_score, top100_score, map_score = get_evalution(
        ground_truth=masked_tokens, logits_lm=predict_prob, exchange_matrix=exchange_map)

    print('fuzzy score =', '{:.6f}'.format(fuzzy_score))
    print('test top1 score =', '{:.6f}'.format(accuracy_score))
    print('test top3 score =', '{:.6f}'.format(top3_score))
    print('test top5 score =', '{:.6f}'.format(top5_score))
    print('test top10 score =', '{:.6f}'.format(top10_score))
    print('test top30 score =', '{:.6f}'.format(top30_score))
    print('test top50 score =', '{:.6f}'.format(top50_score))
    print('test top100 score =', '{:.6f}'.format(top100_score))
    print('test map score =', '{:.6f}'.format(map_score))


def wirte_csv(token_list, csv_name):
    import csv
    token_list = [[token_list[i]] for i in range(len(token_list))]
    with open("csv_file/" + csv_name + ".csv", "w") as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerows(token_list)


test(test_input_ids, test_masked_tokens, test_masked_pos, test_user_ids, test_day_ids)