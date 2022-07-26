import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

import argparse
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

import config as gl

# --device 1 --bs 256 --epoch 50 --loss loss --dataset 5
parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, type=int, help='train device')
parser.add_argument('--bs', default=256, type=int, help='batch size')
parser.add_argument('--epoch', default=0, type=int, help='epoch size')
parser.add_argument('--loss', default='loss', type=str, help='loss fun')
parser.add_argument('--dataset', default='5', type=str, help='dataset')
parser.add_argument('--embed', default='11', type=str, help='cell id embed')

parser.add_argument('--d_model', default=768, type=int, help='embed size')
parser.add_argument('--head', default=2, type=int, help='multi head num')
parser.add_argument('--layer', default=2, type=int, help='layer')

args = parser.parse_args()
# args = parser.parse_args("""--device 1 --bs 256 --epoch 1 --loss los --dataset 5 --embed 9 --d_model 768 --head 2 --layer 2""".split(' '))

# device = 'cuda:1'
# batch_size = 256
# epoch_size = 50
# max_pred = 5  # max tokens of prediction
# loss_fun = "loss"  # loss and spatial_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = args.bs
epoch_size = args.epoch
loss_fun = args.loss  # loss and spatial_loss
train_dataset = "train_traj_%s.h5" % args.dataset
test_dataset = "test_traj_%s.h5" % args.dataset
embed_index = int(args.embed)

d_model = args.d_model
head = args.head
layer = args.layer
gl._init()
gl.set_value('d_model', d_model)
gl.set_value('head', head)
gl.set_value('layer', layer)
gl.set_value('device', device)

embed_path = '/home/traj/jinchengData/'
if embed_index == 1:
    embed_file = 'embedding_dynamic_poi_glove.npy'
elif embed_index == 2:
    embed_file = 'embedding_graph_dynamic_poi_glove.npy'
elif embed_index == 3:
    embed_file = 'embedding_graph_poi_glove.npy'
elif embed_index == 4:
    embed_file = 'embedding_poi_glove.npy'
elif embed_index == 4:
    embed_file = 'embedding_poi_glove.npy'

elif embed_index == 5:
    embed_file = 'embedding_graph_dynamic_poi_64.npy'
elif embed_index == 6:
    embed_file = 'embedding_graph_dynamic_poi_128.npy'
elif embed_index == 7:
    embed_file = 'embedding_graph_dynamic_poi_256.npy'
elif embed_index == 8:
    embed_file = 'embedding_graph_dynamic_poi_512.npy'
elif embed_index == 9:
    embed_file = 'embedding_graph_dynamic_poi_1024.npy'
elif embed_index == 10:
    embed_file = 'embedding_graph_dynamic_poi_glove_1024.npy'
elif embed_index == 11:
    embed_file = 'embedding_dynamic_poi.npy'
else:
    embed_file = "error"
embed_npy = np.load(embed_path + embed_file)
embed_size = embed_npy.shape[1]
gl.set_value('d_model', embed_size)
gl.set_value('pre_em_size',embed_size)
embed_size = embed_npy.shape[1]
print("embed_file:%s, embed_size: %d" %(embed_file,embed_size))



max_pred = 5  # max tokens of prediction
train_df = pd.read_hdf(os.path.join('/home/yj/traj/AttnMove/data/dataset2_o/', train_dataset))
test_df = pd.read_hdf(os.path.join('/home/yj/traj/AttnMove/data/dataset2_o/', test_dataset))
dataset = DataSet(train_df, test_df)
# all_data = dataset.gen_all_data()
train_data = dataset.gen_train_data()  # [seq, user_index, day]
test_data = dataset.gen_test_data()  # [seq, masked_pos, masked_tokens, user_index, day]

train_word_list = list(
    set(str(train_data[i][0][j]) for i in range(len(train_data)) for j in range(len(train_data[i][0]))))
train_word_list.remove('[PAD]')
train_word_list_int = [int(train_word_list[i]) for i in range(len(train_word_list))]
train_word_list_int.sort()
word2embed = dict()
for i, loc in enumerate(train_word_list_int):
    word2embed[str(loc)] = embed_npy[i]
word2embed['[PAD]'] = np.ones(shape=(embed_size,))
word2embed['[MASK]'] = np.ones(shape=(embed_size,))
word2embed['[CLS]'] = np.ones(shape=(embed_size,))
word2embed['[SEP]'] = np.ones(shape=(embed_size,))

word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
for i, w in enumerate(train_word_list_int):
    if w == '[PAD]' or w == '[MASK]':
        print("error")
    word2idx[str(w)] = i + 4

idx2word = {i: w for i, w in enumerate(word2idx)}
idx2embed = {i: word2embed[w] for i, w in enumerate(word2idx)}
idx2embed = torch.from_numpy(np.array(list(idx2embed.values())).astype(float))
vocab_size = len(word2idx) + 2

train_token_list = list()
train_user_list = list()
train_day_list = list()
max_value = 0
for sentence in train_data:
    seq, user_index, day = sentence
    for s in seq:
        max_value = max(max_value, word2idx[s])
    arr = [word2idx[s] for s in seq]
    train_token_list.append(arr)
    train_user_list.append(user_index)
    train_day_list.append(day)

exchange_map = make_exchange_matrix(token_list=train_token_list, token_size=vocab_size)
exchange_map = torch.Tensor(exchange_map).to(device)


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
            input_ids[pos] = word2idx['[MASK]']  # make mask
            # if random() < 0.8:  # 80%
            #     input_ids[pos] = word2idx['[MASK]']  # make mask
            # elif random() > 0.9:  # 10%
            #     index = randint(0, vocab_size - 1)  # random index in vocabulary
            #     while index < 4:  # can't involve 'CLS', 'SEP', 'PAD'
            #         index = randint(0, vocab_size - 1)
            #     input_ids[pos] = index  # replace
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
        masked_pos = [pos + 1 for pos in sentence[1]]
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

test_total_data = make_test_data(test_data)
print("total length of test data is", str(len(test_total_data)))
test_input_ids, test_masked_tokens, test_masked_pos, test_user_ids, test_day_ids = zip(*test_total_data)


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


train_loader = Data.DataLoader(MyDataSet(input_ids, masked_tokens, masked_pos, user_ids, day_ids), batch_size, True)
from downstream.model import Model
if __name__ == '__main__':

    model = Model(vocab_size=vocab_size,id2embed=idx2embed)
    print(model)

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(accelerator="gpu", devices=8) (if you have GPUs)
    trainer = pl.Trainer(   accelerator="gpu", devices=[0], max_epochs=500)
    trainer.fit(model=model, train_dataloaders=train_loader)
