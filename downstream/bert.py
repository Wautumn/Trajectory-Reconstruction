import torch
import torch.nn as nn
import numpy as np
import math

from sklearn.metrics import accuracy_score
from torch import optim
import pandas as pd
import os
import torch.utils.data as Data
from random import *

from preprocess import DataSet
from utils import traj_to_slot, next_batch, get_evalution

device = 'cuda:4'

max_len = 50  # 轨迹的最大长度
batch_size = 768
max_pred = 5  # max tokens of prediction
n_layers = 6
n_heads = 12
d_model = 768
d_ff = 768 * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
# n_segments = 2

train_df = pd.read_hdf(os.path.join('../data/h5_data', "train_trajectory" + ".h5"), key='data')
test_df = pd.read_hdf(os.path.join('../data/h5_data', "test_trajectory" + ".h5"), key='data')
dataset = DataSet(train_df=train_df, test_df=test_df)
train_data = dataset.gen_train_data()
test_data = dataset.gen_test_data()

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
    elif w == '0':
        print("error")
    else:
        print()

idx2word = {i: w for i, w in enumerate(word2idx)}
vocab_size = len(word2idx) + 2

train_token_list = list()
for sentence in train_data:
    arr = [word2idx[s] for s in sentence]
    train_token_list.append(arr)

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


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]


def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(max_len, d_model)  # position embedding
        # self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x).to(device) # [seq_len] -> [batch_size, seq_len]
        self.tok_embed(x)
        self.pos_embed(pos)
        embedding = self.tok_embed(x) + self.pos_embed(pos)
        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_heads * d_v)  # context: [batch_size, seq_len, n_heads * d_v]
        output = self.fc(context)
        return nn.LayerNorm(d_model).to(device)(output + residual)  # output: [batch_size, seq_len, d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                         enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(d_model, 2)
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, masked_pos):
        output = self.embedding(input_ids)  # [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)  # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output = layer(output, enc_self_attn_mask)
        # it will be decided by first token(CLS)
        # h_pooled = self.fc(output[:, 0])  # [batch_size, d_model]
        # logits_clsf = self.classifier(h_pooled)  # [batch_size, 2] predict isNext

        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model)  # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size, max_pred, d_model]
        h_masked = self.activ2(self.linear(h_masked))  # [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked)  # [batch_size, max_pred, vocab_size]
        return logits_lm


model = BERT().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)

train_predict = []
train_truth = []
for epoch in range(1000):
    train_predict, train_truth = [], []
    for input_ids, masked_tokens, masked_pos in loader:
        logits_lm = model(input_ids, masked_pos)
        train_truth.extend(masked_tokens.flatten().cpu().data.numpy())
        train_predict.extend(logits_lm.data.max(2)[1].flatten().cpu().data.numpy())
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
    predict = []
    predict_prob = torch.Tensor([]).to(device)

    for batch in next_batch(a, batch_size=32):
        # Value filled with num_loc stands for masked tokens that shouldn't be considered.
        batch_token_list, batch_masked_pos = zip(*batch)
        logits_lm = model(torch.LongTensor(batch_token_list).to(device), torch.LongTensor(batch_masked_pos).to(device))
        predict_prob = torch.cat([predict_prob, logits_lm], dim=0)
        logits_lm = logits_lm.data.max(2)[1]
        logits_lm = logits_lm.flatten().cpu().data.numpy()
        predict.extend(list(logits_lm))
    # print(len(predict), masked_tokens.shape)
    # test_accuracy = accuracy_score(np.array(predict), masked_tokens)
    # print('test accracy =', '{:.6f}'.format(test_accuracy))
    accuracy_score, recall3_score, recall5_score = get_evalution(ground_truth=masked_tokens, logits_lm=predict_prob)
    print('test accuracy score =', '{:.6f}'.format(accuracy_score))
    print('test recall3 score =', '{:.6f}'.format(recall3_score))
    print('test recall5 score =', '{:.6f}'.format(recall5_score))


test(test_token_list, test_masked_tokens, test_masked_pos)
