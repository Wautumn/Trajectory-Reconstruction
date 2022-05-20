import os
from collections import defaultdict
from itertools import zip_longest
import math
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from torch import optim
from sklearn.utils import shuffle
import torch.utils.data as Data

from dataset import DataSetRaw
from utils import next_batch, get_slot_by_unix, ts_to_slot, traj_to_slot

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V


# n_layers = 6  # number of Encoder of Decoder Layer
# n_heads = 8  # number of heads in Multi-Head Attention
# src_vocab_size = 100000
# tgt_vocab_size = 100000
# device = 'cuda:0'

def split_time(embedding_inputs, timestamps, timeslot=30):
    '''
    :param timeslot: 30 minute
    :param embedding_inputs: (batch_size, src_len, embedding_size)
    :param timestamps: (batch_size, src_len)
    :return: (batch_size, normalize_len, embedding_size)
    '''
    count = 24 * (60 // timeslot)
    batch_size, src_len, embedding_size = embedding_inputs.size()
    slot_emb = torch.zeros(batch_size, count, embedding_size)
    for i in range(batch_size):
        temp = defaultdict(list)
        for j in range(src_len):
            if timestamps[i][j] == 0:
                break
            ts_slot = get_slot_by_unix(timestamps[i][j])
            ts_src = embedding_inputs[i, j, :]
            temp[ts_slot].append(ts_src.cpu().data.numpy())

        for slot in temp:
            if len(temp[slot]) == 1:
                slot_emb[i, slot, :] = torch.Tensor(temp[slot]).cuda()
            elif len(temp[slot]) > 1:
                average = torch.mean(torch.tensor(temp[slot]), dim=0)
                slot_emb[i, slot, :] = average
            else:
                raise Exception("List length exception")
    slot_emb = torch.Tensor(slot_emb).cuda()
    # slot_emb: (batch_size, count, embed_size)
    return slot_emb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        temp = self.pe[:x.size(0), :]
        x = x + temp
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = embed_size
        self.W_Q = nn.Linear(self.d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, self.d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, d_k).transpose(1,
                                                                                2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, d_k).transpose(1,
                                                                                2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, d_v).transpose(1,
                                                                                2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, embed_size):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = embed_size
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, self.d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, n_heads=8):
        super(EncoderLayer, self).__init__()
        self.d_model = embed_size
        self.enc_self_attn = MultiHeadAttention(embed_size)
        self.pos_ffn = PoswiseFeedForwardNet(embed_size)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, embedding_layer, embed_size, n_layers=6, n_heads=8):
        super(Encoder, self).__init__()
        self.embedding_layer = embedding_layer
        self.d_model = embed_size
        # self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(embed_size) for _ in range(n_layers)])

    def forward(self, enc_inputs, input_ts, input_ts_pad):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        # enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        # enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_outputs = self.embedding_layer(enc_inputs)
        enc_outputs = split_time(enc_outputs, input_ts)  # enc_inputs: [batch_size, slot_len, d_model]
        # enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attn_mask = get_attn_pad_mask(input_ts_pad, input_ts_pad)  # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self, embed_size):
        super(DecoderLayer, self).__init__()
        self.d_model = embed_size
        self.dec_self_attn = MultiHeadAttention(embed_size)
        self.dec_enc_attn = MultiHeadAttention(embed_size)
        self.pos_ffn = PoswiseFeedForwardNet(embed_size)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, embedding_layer, embed_size, n_layers=6):
        super(Decoder, self).__init__()
        self.embedding_layer = embedding_layer
        self.d_model = embed_size
        # self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        # self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(embed_size) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs, ts_outputs, input_ts_pad, output_ts_pad):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]
        '''
        # dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        # dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).cuda()  # [batch_size, tgt_len, d_model]
        dec_outputs = self.embedding_layer(dec_inputs)
        dec_outputs = split_time(dec_outputs, ts_outputs)
        # dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_pad_mask = get_attn_pad_mask(output_ts_pad,
                                                   output_ts_pad).cuda()  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(
            output_ts_pad).cuda()  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0).cuda()  # [batch_size, tgt_len, tgt_len]

        # dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]
        dec_enc_attn_mask = get_attn_pad_mask(output_ts_pad, input_ts_pad)  # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, embedding_layer, embed_size, num_loc, n_layers=6, n_heads=8):
        super(Transformer, self).__init__()
        self.embedding_layer = embedding_layer
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = embed_size
        self.num_loc = num_loc
        self.src_vocab_size = num_loc
        self.tgt_vocab_size = num_loc

        self.encoder = Encoder(embedding_layer, embed_size=embed_size, n_layers=6)
        self.decoder = Decoder(embedding_layer, embed_size=embed_size, n_layers=6)
        self.projection = nn.Linear(self.d_model, self.tgt_vocab_size + 2, bias=False).cuda()

    def forward(self, enc_inputs, dec_inputs, input_ts, output_ts, input_ts_pad, output_ts_pad):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, input_ts, input_ts_pad)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs, output_ts,
                                                                  input_ts_pad, output_ts_pad)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


def greedy_decoder(model, enc_input, input_ts, input_ts_pad, output_ts, output_ts_pad, seq_len=48):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    batch_size, _ = enc_input.size()
    # for i in range(batch_size):
    #     one_enc_input, one_input_ts, one_input_ts_pad, one_output_ts, one_output_ts_pad = enc_input[i, :].unsqueeze(
    #         0), input_ts[i, :].unsqueeze(0), input_ts_pad[i, :].unsqueeze(0), output_ts[i, :].unsqueeze(
    #         0), output_ts_pad[i, :].unsqueeze(0)
    #     one_enc_outputs, one_enc_self_attns = model.encoder(one_enc_input, one_input_ts, one_input_ts_pad)
    #     one_dec_input = torch.zeros(1, 0).type_as(one_enc_outputs.data)
    #     terminal = False
    #     next_symbol = 0
    #     while not terminal:
    #         one_dec_input = torch.cat(
    #             [one_dec_input.detach(), torch.tensor([[next_symbol]], dtype=one_enc_outputs.dtype).cuda()],
    #             -1)
    #         dec_outputs, _, _ = model.decoder(one_dec_input, one_enc_input, one_enc_outputs, one_output_ts,
    #                                           one_input_ts_pad, one_output_ts_pad)
    #         projected = model.projection(dec_outputs)
    #         prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
    #         next_word = prob.data[-1]
    #         next_symbol = next_word
    #         if dec_input.size(1) == 48:
    #             terminal = True
    #         print(next_word)

    enc_outputs, enc_self_attns = model.encoder(enc_input, input_ts, input_ts_pad)
    dec_input = torch.zeros(batch_size, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = 0
    next_temp = torch.tensor([[next_symbol] for _ in range(batch_size)], dtype=enc_input.dtype)
    while not terminal:
        dec_input = torch.cat([dec_input.detach(), next_temp.cuda()], -1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs, output_ts, input_ts_pad, output_ts_pad)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[:, -1].unsqueeze(1)
        next_temp = next_word
        if dec_input.size(1) == 48:
            terminal = True
    return dec_input


def test_transformer(transformer, enc_inputs, dec_inputs, dec_outputs, input_ts, output_ts, device):
    test_enc_inputs = np.transpose(np.array(list(zip_longest(*enc_inputs, fillvalue=0))))
    test_enc_inputs = torch.LongTensor(test_enc_inputs).to(device)

    test_dec_inputs = np.transpose(np.array(list(zip_longest(*dec_inputs, fillvalue=0))))
    test_dec_inputs = torch.LongTensor(test_dec_inputs).to(device)

    test_dec_outputs = [traj_to_slot(dec_outputs[i], output_ts[i]) for i in range(len(dec_outputs))]
    test_dec_outputs = torch.LongTensor(test_dec_outputs).to(device)

    test_input_ts = np.transpose(np.array(list(zip_longest(*input_ts, fillvalue=0))))
    test_output_ts = np.transpose(np.array(list(zip_longest(*output_ts, fillvalue=0))))
    test_input_ts = torch.LongTensor(test_input_ts).to(device)
    test_output_ts = torch.LongTensor(test_output_ts).to(device)

    test_input_ts_pad = [ts_to_slot(input_ts[i]) for i in range(len(input_ts))]
    test_input_ts_pad = torch.Tensor(np.array(test_input_ts_pad)).to(device)

    test_output_ts_pad = [ts_to_slot(output_ts[i]) for i in range(len(output_ts))]
    test_output_ts_pad = torch.Tensor(np.array(test_output_ts_pad)).to(device)

    greedy_dec_input = greedy_decoder(transformer, test_enc_inputs, test_input_ts, test_input_ts_pad,
                                      test_output_ts, test_output_ts_pad, )
    predict, _, _, _ = transformer(test_enc_inputs, greedy_dec_input, test_input_ts, test_output_ts,
                                   test_input_ts_pad, test_output_ts_pad)
    predict = predict.data.max(1, keepdim=True)[1]
    predict = predict.squeeze()
    predict = predict.view(-1, 48)

    test_dec_outputs = test_dec_outputs.data.cpu().numpy().reshape(-1)
    predict = predict.data.cpu().numpy().reshape(-1)

    accuracy = accuracy_score(test_dec_outputs, predict)
    # precision = precision_score(test_dec_outputs, predict)
    # f1 = f1_score(test_dec_outputs, predict)
    # recall = recall_score(test_dec_outputs, predict)

    return accuracy

    # test_dec_outputs = [traj_to_slot(dec_outputs[i], output_ts[i]) for i in range(len(dec_outputs))]
    # test_inputs = []
    # pre_outpus = []
    # for j in range(len(traj_enc_inputs)):
    #     test_enc_input = torch.LongTensor(np.array(traj_enc_inputs[j])).to(device)
    #     test_output = np.array(traj_dec_outputs[j])
    #     greedy_dec_input = greedy_decoder(transformer, test_enc_input.view(1, -1),
    #                                       start_symbol=dataset.start,
    #                                       end_symbol=dataset.end, max_seq_len=max_seq_len)
    #     predict, _, _, _ = transformer(test_enc_input.view(1, -1), greedy_dec_input)
    #     predict = predict.data.max(1, keepdim=True)[1]
    #     predict = predict.squeeze()
    #     pre_outpus.append(predict)
    #     test_inputs.append(test_output)


def train_transformer(dataset, max_seq_len, transformer, num_epoch, batch_size, device):
    traj_enc_inputs, traj_dec_inputs, traj_dec_outputs, traj_input_ts, traj_output_ts = zip(
        *dataset.gen_enc2dec(max_seq_len=max_seq_len))
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(transformer.parameters(), lr=1e-3, momentum=0.99)

    test_traj_enc_inputs, test_traj_dec_inputs, test_traj_dec_outputs, test_traj_input_ts, test_traj_output_ts = zip(
        *dataset.gen_enc2dec(max_seq_len=max_seq_len, min_len=36, choice_prop=0.9))

    for epoch in range(num_epoch):
        all_loss = 0
        for i, batch in enumerate(
                next_batch(shuffle(
                    list(zip(traj_enc_inputs, traj_dec_inputs, traj_dec_outputs, traj_input_ts, traj_output_ts))),
                    batch_size=batch_size)):
            # Value filled with num_loc stands for masked tokens that shouldn't be considered.
            enc_inputs, dec_inputs, dec_outputs, input_ts, output_ts = zip(*batch)

            dec_outputs = [traj_to_slot(dec_outputs[i], output_ts[i]) for i in range(len(dec_outputs))]

            enc_inputs = np.transpose(np.array(list(zip_longest(*enc_inputs, fillvalue=0))))
            dec_inputs = np.transpose(np.array(list(zip_longest(*dec_inputs, fillvalue=0))))
            dec_outputs = np.transpose(np.array(list(zip_longest(*dec_outputs, fillvalue=0))))

            input_ts_pad = [ts_to_slot(input_ts[i]) for i in range(len(input_ts))]
            input_ts_pad = torch.Tensor(np.array(input_ts_pad)).to(device)
            output_ts_pad = [ts_to_slot(output_ts[i]) for i in range(len(output_ts))]
            output_ts_pad = torch.Tensor(np.array(output_ts_pad)).to(device)

            input_ts = np.transpose(np.array(list(zip_longest(*input_ts, fillvalue=0))))
            output_ts = np.transpose(np.array(list(zip_longest(*output_ts, fillvalue=0))))
            enc_inputs, dec_inputs, dec_outputs, input_ts, output_ts = \
                torch.LongTensor(enc_inputs).to(device), torch.LongTensor(dec_inputs).to(device), torch.LongTensor(
                    dec_outputs).to(device), torch.LongTensor(input_ts).to(device), torch.LongTensor(output_ts).to(
                    device)

            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = transformer(enc_inputs, dec_inputs, input_ts,
                                                                                 output_ts, input_ts_pad, output_ts_pad)

            loss = criterion(outputs, dec_outputs.contiguous().view(-1))
            all_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if i % 40 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        print('-------Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(all_loss))
        print("-------Test result")
        accuracy = test_transformer(transformer=transformer, enc_inputs=test_traj_enc_inputs,
                                    dec_inputs=test_traj_dec_inputs,
                                    dec_outputs=test_traj_dec_outputs,
                                    input_ts=test_traj_input_ts, output_ts=test_traj_output_ts,
                                    device=device)
        print(accuracy)


if __name__ == '__main__':
    print()
    # file_name = "test_data"
    # raw_df = pd.read_hdf(os.path.join('../data/h5_data', file_name + ".h5"), key='data')
    # dataset = DataSetRaw(raw_df)
    # embed_layer = ""
    # transformer = Transformer(embedding_layer=embed_layer, n_layers=6).to(device)
    # train_transformer(dataset, transformer=transformer, num_epoch=1, batch_size=32, device=device)
