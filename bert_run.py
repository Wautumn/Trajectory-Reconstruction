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

device = 'cuda:3'

test_index = 4

batch_size = 512
epoch_size = 2000
max_pred = 5  # max tokens of prediction
loss_fun = "spatial_loss"  # loss
# n_segments = 2

train_df = pd.read_hdf(os.path.join('data/h5_data_all', "train_traj" + ".h5"), key='data')
test_df_1 = pd.read_hdf(os.path.join('data/h5_data_all', "test_traj_1" + ".h5"), key='data')
test_df_2 = pd.read_hdf(os.path.join('data/h5_data_all', "test_traj_2" + ".h5"), key='data')
test_df_3 = pd.read_hdf(os.path.join('data/h5_data_all', "test_traj_3" + ".h5"), key='data')
test_df_4 = pd.read_hdf(os.path.join('data/h5_data_all', "test_traj_4" + ".h5"), key='data')

dataset = DataSet(train_df, test_df_1, test_df_2, test_df_3, test_df_4)
train_data = dataset.gen_train_data()
test_data = dataset.gen_test_data(index=1)

train_word_list = list(
    set(str(train_data[i][j]) for i in range(len(train_data)) for j in range(len(train_data[i]))))
test_word_list = list(
    set(str(test_data[i][0][j]) for i in range(len(test_data)) for j in range(len(test_data[i][0]))))
test_masked_list = list(
    set(str(test_data[i][2][j]) for i in range(len(test_data)) for j in range(len(test_data[i][2]))))
word_list = []
word_list.extend(train_word_list)
word_list.extend(test_word_list)
word_list.extend(test_masked_list)
word_list = list(set(word_list))


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

test_token_list, test_masked_tokens, test_masked_pos = list(), list(), list()
for sentence in test_data:
    arr = [word2idx[s] for s in sentence[0]]
    arr = [word2idx['[CLS]']] + arr + [word2idx['[SEP]']]
    test_token_list.append(arr)
    masked = [word2idx[str(s)] for s in sentence[2]]
    test_masked_tokens.append(masked)
    test_masked_pos.append(sentence[1])


def make_train_data(token_list):
    batch = []
    while len(batch) < batch_size:
        tokens_a_index = randrange(len(token_list))  # sample random index in sentences
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

        # # Zero Paddings
        # n_pad = max_len - len(input_ids)
        # print(n_pad)
        # input_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
        batch.append([input_ids, masked_tokens, masked_pos])
    return batch


batch = make_train_data(train_token_list)
input_ids, masked_tokens, masked_pos = zip(*batch)
input_ids, masked_tokens, masked_pos = torch.LongTensor(input_ids).to(device), torch.LongTensor(
    masked_tokens).to(device), torch.LongTensor(
    masked_pos).to(device)


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
    for input_ids, masked_tokens, masked_pos in loader:
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
        if (epoch + 1) % 10 == 0:
            accuracy = accuracy_score(train_truth, train_predict)
            print('Epoch:', '%06d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss), 'train accracy =',
                  '{:.6f}'.format(accuracy))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save({'model': model.state_dict()},
           'pth/dataset%s-batch%s-epoch%s-%s.pth' % (test_index, batch_size, epoch_size, loss_fun))

# state_dict = torch.load('model/model_name.pth')
# model.load_state_dict(state_dict['model'])

# test_token_list, test_masked_tokens, test_masked_pos
for i in range(len(test_masked_pos)):
    for j in range(len(test_masked_pos[i])):
        test_masked_pos[i][j] += 1


def test(test_token_list, test_masked_tokens, test_masked_pos):
    # token_list = np.array(test_token_list)
    masked_tokens = np.array(test_masked_tokens).reshape(-1)
    # masked_pos = np.array(test_masked_pos)
    a = list(zip(test_token_list, test_masked_pos))
    truth = []
    # predict = []
    predict_prob = torch.Tensor([]).to(device)

    for batch in next_batch(a, batch_size=64):
        # Value filled with num_loc stands for masked tokens that shouldn't be considered.
        batch_token_list, batch_masked_pos = zip(*batch)
        logits_lm = model(torch.LongTensor(batch_token_list).to(device), torch.LongTensor(batch_masked_pos).to(device))
        logits_lm = torch.topk(logits_lm, 10, dim=2)[1]
        predict_prob = torch.cat([predict_prob, logits_lm], dim=0)
        # logits_lm = logits_lm.data.max(2)[1]
        # logits_lm = logits_lm.flatten().cpu().data.numpy()
        # predict.extend(list(logits_lm))

    accuracy_score, recall3_score, recall5_score = get_evalution(ground_truth=masked_tokens, logits_lm=predict_prob)
    print('test accuracy score =', '{:.6f}'.format(accuracy_score))
    print('test recall3 score =', '{:.6f}'.format(recall3_score))
    print('test recall5 score =', '{:.6f}'.format(recall5_score))


test(test_token_list, test_masked_tokens, test_masked_pos)
