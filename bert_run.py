import datetime
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from torch import optim
import pandas as pd
import torch.utils.data as Data
from random import *

from preprocess import DataSet
from utils import next_batch, get_evalution, make_exchange_matrix, Loss_Function
from downstream.bert import BERT

device = 'cuda:1'
batch_size = 256
epoch_size = 50
max_pred = 5  # max tokens of prediction
loss_fun = "spatial_loss"  # loss and spatial_loss
# n_segments = 2

train_df = pd.read_hdf(os.path.join('data/Dataset Filtered h5', "train_traj" + ".h5"), key='data')
test_df = pd.read_hdf(os.path.join('data/Dataset Filtered h5', "test_traj" + ".h5"), key='data')
dataset = DataSet(train_df, test_df)
# all_data = dataset.gen_all_data()
train_data = dataset.gen_train_data()
test_data = dataset.gen_test_data()

train_word_list = list(
    set(str(train_data[i][j]) for i in range(len(train_data)) for j in range(len(train_data[i]))))
word_list = list(set(train_word_list))

word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
for i, w in enumerate(word_list):
    if w != '[PAD]' and w != '[MASK]':
        word2idx[w] = i + 4

idx2word = {i: w for i, w in enumerate(word2idx)}
vocab_size = len(word2idx) + 2

train_token_list = list()
for sentence in train_data:
    arr = [word2idx[s] for s in sentence]
    train_token_list.append(arr)

exchange_map = make_exchange_matrix(token_list=train_token_list, token_size=vocab_size)
exchange_map = torch.Tensor(exchange_map).to(device)


# test_token_list, test_masked_tokens, test_masked_pos = list(), list(), list()
# for sentence in test_data:
#     arr = [word2idx[s] for s in sentence[0]]
#     arr = [word2idx['[CLS]']] + arr + [word2idx['[SEP]']]
#     test_token_list.append(arr)
#     masked = [word2idx[str(s)] for s in sentence[2]]
#     test_masked_tokens.append(masked)
#     test_masked_pos.append([pos + 1 for pos in sentence[1]])


def make_train_data(token_list):
    total_data = []
    for i in range(len(token_list)):
        tokens_a_index = i  # sample random index in sentences
        tokens_a = token_list[tokens_a_index]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']]

        # MASK LM
        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))  # 15 % of tokens in one sentence
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
    # seq, masked_pos, masked_tokens, user_index
    total_test_data = []
    for sentence in test_data:
        arr = [word2idx[s] for s in sentence[0]]
        arr = [word2idx['[CLS]']] + arr + [word2idx['[SEP]']]
        masked_pos = [pos + 1 for pos in sentence[1]]
        masked_tokens = [word2idx[str(s)] for s in sentence[2]]
        total_test_data.append([arr, masked_tokens, masked_pos])
    return total_test_data


total_data = make_train_data(train_token_list)  # [input_ids, masked_tokens, masked_pos]
print("total length of train data is", str(len(total_data)))  #
input_ids, masked_tokens, masked_pos = zip(*total_data)
input_ids, masked_tokens, masked_pos, = torch.LongTensor(input_ids).to(device), \
                                        torch.LongTensor(masked_tokens).to(device), \
                                        torch.LongTensor(masked_pos).to(device)


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, masked_tokens, masked_pos):
        self.input_ids = input_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.masked_tokens[idx], self.masked_pos[idx]


loader = Data.DataLoader(MyDataSet(input_ids, masked_tokens, masked_pos), batch_size, True)

model = BERT(vocab_size=vocab_size).to(device)
if loss_fun == "spatial_loss":
    criterion = Loss_Function()
else:
    criterion = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)
train_predict = []
train_truth = []

for epoch in range(epoch_size):
    train_predict, train_truth = [], []
    for i, (input_ids, masked_tokens, masked_pos) in enumerate(loader):
        logits_lm = model(input_ids, masked_pos)
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
           'pth/dataset-batch%s-epoch%s-%s-%s.pth' % (
               batch_size, epoch_size, loss_fun, datetime.datetime.now().strftime("%Y%m%d")))

# state_dict = torch.load('model/model_name.pth')
# model.load_state_dict(state_dict['model'])

test_total_data = make_test_data(test_data)
print("total length of test data is", str(len(test_total_data)))
test_input_ids, test_masked_tokens, test_masked_pos = zip(*test_total_data)


def test(test_token_list, test_masked_tokens, test_masked_pos):
    masked_tokens = np.array(test_masked_tokens).reshape(-1)
    a = list(zip(test_token_list, test_masked_pos))
    predict_prob = torch.Tensor([]).to(device)

    for batch in next_batch(a, batch_size=64):
        # Value filled with num_loc stands for masked tokens that shouldn't be considered.
        batch_token_list, batch_masked_pos = zip(*batch)
        logits_lm = model(torch.LongTensor(batch_token_list).to(device), torch.LongTensor(batch_masked_pos).to(device))
        logits_lm = torch.topk(logits_lm, 100, dim=2)[1]
        predict_prob = torch.cat([predict_prob, logits_lm], dim=0)

    accuracy_score, fuzzzy_score, top3_score, top5_score, top10_score, top30_score, top50_score, top100_score = get_evalution(
        ground_truth=masked_tokens, logits_lm=predict_prob, exchange_matrix=exchange_map)
    print('test accuracy score =', '{:.6f}'.format(accuracy_score))
    print('fuzzzy score =', '{:.6f}'.format(fuzzzy_score))
    print('test top3 score =', '{:.6f}'.format(top3_score))
    print('test top5 score =', '{:.6f}'.format(top5_score))
    print('test top10 score =', '{:.6f}'.format(top10_score))
    print('test top30 score =', '{:.6f}'.format(top30_score))
    print('test top50 score =', '{:.6f}'.format(top50_score))
    print('test top100 score =', '{:.6f}'.format(top100_score))


test(test_input_ids, test_masked_tokens, test_masked_pos)
