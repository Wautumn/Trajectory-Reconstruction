#%%
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
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

# --device 1 --bs 256 --epoch 50 --loss loss --dataset 5
parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, type=int, help='train device')
parser.add_argument('--bs', default=256, type=int, help='batch size')
parser.add_argument('--epoch', default=501, type=int, help='epoch size')
parser.add_argument('--loss', default='spatial_loss', type=str, help='loss fun')
parser.add_argument('--dataset', default='5', type=str, help='dataset')
parser.add_argument('--embed', default='11', type=str, help='cell id embed')

parser.add_argument('--d_model', default=1024, type=int, help='embed size')
parser.add_argument('--head', default=2, type=int, help='multi head num')
parser.add_argument('--layer', default=2, type=int, help='layer')

args = parser.parse_args()
# args = parser.parse_args("""--device 1 --bs 256 --epoch 1 --loss spatial_loss --dataset 5 --embed 9 --d_model 768 --head 2 --layer 2""".split(' '))

# device = 'cuda:1'
# batch_size = 256
# epoch_size = 50
# max_pred = 5  # max tokens of prediction
# loss_fun = "loss"  # loss and spatial_loss

device = 'cuda:%s' % args.device
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
gl.set_value('pre_em_size', embed_size)
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

train_df = pd.read_hdf(os.path.join('/home/yj/traj/AttnMove/data/dataset2_o/', train_dataset))
train_df=train_df[train_df['day']<=5]
dataset = DataSet(train_df, test_df)
train_data = dataset.gen_train_data()  # [seq, user_index, day]
test_data = dataset.gen_test_data()  # [seq, masked_pos, masked_tokens, user_index, day]

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


loader = Data.DataLoader(MyDataSet(input_ids, masked_tokens, masked_pos, user_ids, day_ids), batch_size, True)
from downstream.bert_embed2 import BERT

#%%
model = BERT(vocab_size=vocab_size, id2embed=idx2embed).to(device)
if loss_fun == "spatial_loss":
    criterion = Loss_Function()
else:
    criterion = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)
train_predict = []
train_truth = []


def map_score(ground_truth_list, logits_lm):
    MAP = 0

    # pred_topk = torch.flatten(logits_lm,start_dim=0,end_dim=1).cpu().data.numpy()
    logits_lm = logits_lm.cpu().data.numpy()
    MAP=0
    for j in range(len(ground_truth_list)):    
        ground_truth = ground_truth_list[j]
        pred_topk = logits_lm[j]

        for i in range(len(ground_truth)):
            if ground_truth[i] in pred_topk[i]:
                rank = np.argwhere(ground_truth[i]==pred_topk[i]) +1
                MAP += 1.0/rank[0][0]
    return MAP

def test(test_token_list, test_masked_tokens, test_masked_pos, test_user_ids, test_day_ids):
    masked_tokens = np.array(test_masked_tokens).reshape(-1)
    a = list(zip(test_masked_tokens, test_token_list, test_masked_pos, test_user_ids, test_day_ids))
    predict_prob = torch.Tensor([]).to(device)
    MAP=0
    for batch in next_batch(a, batch_size=64):
        # Value filled with num_loc stands for masked tokens that shouldn't be considered.
        batch_masked_tokens, batch_token_list, batch_masked_pos, batch_user_ids, batch_day_ids = zip(*batch)
        logits_lm = model(torch.LongTensor(batch_token_list).to(device),
                          torch.LongTensor(batch_masked_pos).to(device),
                          torch.LongTensor(batch_user_ids).to(device),
                          torch.LongTensor(batch_day_ids).to(device), )
        logits_lm = torch.topk(logits_lm, 100, dim=2)[1]
        predict_prob = torch.cat([predict_prob, logits_lm], dim=0)
        MAP+= map_score(batch_masked_tokens, logits_lm=logits_lm)
    MAP=MAP/len(test_total_data)/5
    

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
    print('test top100 score =', '{:.6f}'.format(MAP))

    return 'test accuracy score =' + '{:.6f}'.format(accuracy_score) + '\n' + \
        'fuzzzy score =' +  '{:.6f}'.format(fuzzzy_score) + '\n'\
            + 'test top3 score ='+ '{:.6f}'.format(top3_score) + '\n'\
            + 'test top5 score ='+ '{:.6f}'.format(top5_score) + '\n'\
            + 'test top10 score ='+ '{:.6f}'.format(top10_score) + '\n'\
            + 'test top30 score ='+ '{:.6f}'.format(top30_score) + '\n'\
            + 'test top50 score ='+ '{:.6f}'.format(top50_score) + '\n'\
            + 'test top100 score ='+ '{:.6f}'.format(top100_score) + '\n' \
            + 'test MAP score ='+ '{:.6f}'.format(MAP) + '\n'

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
        # if i % 10 == 0:
        #     break
    if epoch%5 ==0:
        torch.save({'model': model.state_dict()},
           'pth_embed/dataset-batch%s-epoch%s-%s-dmodel%s-head%s_layer%s_embed%s_%s_%s_epoch%d.pth' % (
               batch_size, epoch_size, loss_fun, d_model, head, layer, embed_index, embed_file.replace(".npy", ""),
               datetime.datetime.now().strftime("%Y%m%d"),epoch))
        f = open('result.txt','a+')
        f.write("epoch: %d \n" %epoch)
        f.write("loss: %.6f" % loss)
        f.write('pth_embed/dataset-batch%s-epoch%s-%s-dmodel%s-head%s_layer%s_embed%s_%s_%s_epoch%d.pth\n' % (
               batch_size, epoch_size, loss_fun, d_model, head, layer, embed_index, embed_file.replace(".npy", ""),
               datetime.datetime.now().strftime("%Y%m%d"),epoch))
        model.eval()
        result = test(test_input_ids, test_masked_tokens, test_masked_pos, test_user_ids, test_day_ids)
        model.train()        f.write(result)
        f.close()

        
torch.save({'model': model.state_dict()},
           'pth_embed/dataset-batch%s-epoch%s-%s-dmodel%s-head%s_layer%s_embed%s_%s_%s.pth' % (
               batch_size, epoch_size, loss_fun, d_model, head, layer, embed_index, embed_file.replace(".npy", ""),
               datetime.datetime.now().strftime("%Y%m%d")))

# state_dict = torch.load('pth/dataset-batch256-epoch50-loss-20220707_tem_without_pos.pth.pth')
# model.load_state_dict(state_dict['model'])



            
model.eval()
test(test_input_ids, test_masked_tokens, test_masked_pos, test_user_ids, test_day_ids)
