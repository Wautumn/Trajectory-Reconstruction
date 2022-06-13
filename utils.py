import collections
import math
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda:3'


def next_batch(data, batch_size):
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        yield data[start_index:end_index]


def get_slot_by_datetime(datetime):
    format = '%Y-%m-%d %H:%M:%S'
    format_time = time.strptime(datetime, format)
    day, hour, minute = format_time.tm_mday, format_time.tm_hour, format_time.tm_min
    slot = (day - 1) * 48 + hour * 2 + (0 if minute <= 30 else 1)
    return slot


def get_slot_by_unix(ts):
    dt = time.localtime(ts)
    day, hour, minute = dt.tm_mday, dt.tm_hour, dt.tm_min
    slot = hour * 2 + (0 if minute <= 30 else 1)
    return slot


def list_to_array(x):
    dff = pd.concat([pd.DataFrame({'{}'.format(index): labels}) for index, labels in enumerate(x)], axis=1)
    return dff.fillna(0).values.T.astype(int)


def ts_to_slot(ts):
    ans = [0] * 48
    for t in ts:
        slot = get_slot_by_unix(t)
        ans[slot] = 1
    return ans


def traj_to_slot(trajectory, ts, pad=0):
    ans = [pad] * 48
    for i in range(len(ts)):
        slot_id = get_slot_by_unix(ts[i])
        ans[slot_id] = trajectory[i]
    return ans


def topk(ground_truth, logits_lm, k):
    pred_topk = logits_lm[:, :, 0:k]
    pred_topk = torch.flatten(pred_topk, start_dim=0, end_dim=1).cpu().data.numpy()
    topk_token = 0
    for i in range(len(ground_truth)):
        if ground_truth[i] in pred_topk[i]:
            topk_token += 1
    topk_score = topk_token / len(ground_truth)
    return topk_token, topk_score


def get_evalution(ground_truth, logits_lm, exchange_matrix):
    pred_acc = logits_lm[:, :, 0]
    pred_acc = pred_acc.flatten().cpu().data.numpy()
    accuracy_token = 0
    for i in range(len(ground_truth)):
        if pred_acc[i] == ground_truth[i]:
            accuracy_token += 1
    print("accuracy:", accuracy_token, accuracy_token / len(ground_truth))
    accuracy_score = accuracy_token / len(ground_truth)

    pred_acc = logits_lm[:, :, 0]
    pred_acc = pred_acc.flatten().cpu().data.numpy()

    funzzy_accuracy_token = 0
    for i in range(len(pred_acc)):
        a = int(pred_acc[i])
        b = ground_truth[i]
        if exchange_matrix[b][a] > 0:
            funzzy_accuracy_token += 1
    print("fuzzzy:", funzzy_accuracy_token, funzzy_accuracy_token / len(ground_truth))
    fuzzzy_score = funzzy_accuracy_token / len(ground_truth)

    top3_token, top3_score = topk(ground_truth, logits_lm, 3)
    print("top3:", top3_token, top3_score)

    top5_token, top5_score = topk(ground_truth, logits_lm, 5)
    print("top5:", top5_token, top5_score)

    top10_token, top10_score = topk(ground_truth, logits_lm, 10)
    print("top10:", top10_token, top10_score)

    top30_token, top30_score = topk(ground_truth, logits_lm, 30)
    print("top30:", top30_token, top30_score)

    top50_token, top50_score = topk(ground_truth, logits_lm, 50)
    print("top50:", top50_token, top50_score)

    top100_token, top100_score = topk(ground_truth, logits_lm, 100)
    print("top100:", top100_token, top100_score)

    return accuracy_score, fuzzzy_score, top3_score, top5_score, top10_score, top30_score, top50_score, top100_score


class Loss_Function(nn.Module):
    def __init__(self):
        super(Loss_Function, self).__init__()

    def Spatial_Loss(self, weight, logit_lm, ground_truth):
        _, num_classes = logit_lm.size()
        p_i = torch.softmax(logit_lm, dim=1)
        spatial_matrix = torch.index_select(weight, 0, ground_truth)

        loss = spatial_matrix * torch.log(p_i + 0.0000001)
        loss = torch.sum(loss, dim=1)
        loss = -torch.mean(loss, dim=0)
        return loss

    def Cross_Entropy_Loss(self, logit_lm, ground_truth):
        _, num_classes = logit_lm.size()
        p_i = torch.softmax(logit_lm, dim=1)
        y = F.one_hot(ground_truth, num_classes=num_classes)
        loss = y * torch.log(p_i + 0.0000001)
        loss = torch.sum(loss, dim=1)
        loss = -torch.mean(loss, dim=0)
        return loss


def make_exchange_matrix(token_list, token_size, alpha=98, theta=1000):
    token_list = [list(filter(lambda x: x > 3, token)) for token in token_list]
    exchange_matrix = np.zeros(shape=(token_size, token_size))
    for token in token_list:
        for i in range(1, len(token)):
            if token[i] == token[i - 1]:
                continue
            exchange_matrix[token[i - 1]][token[i]] += 1
    print(np.min(exchange_matrix), np.max(exchange_matrix))
    exchange_matrix = np.where(exchange_matrix >= alpha, exchange_matrix, 0)
    exchange_matrix = exchange_matrix / theta
    exchange_matrix = np.where(exchange_matrix > 0, np.exp(exchange_matrix), 0)
    print(np.min(exchange_matrix), np.max(exchange_matrix))
    for i in range(token_size):
        row_sum = sum(exchange_matrix[i]) + np.exp(1)
        for j in range(token_size):
            if exchange_matrix[i][j] != 0:
                exchange_matrix[i][j] = exchange_matrix[i][j] / row_sum
    print(np.min(exchange_matrix), np.max(exchange_matrix))
    for i in range(token_size):
        exchange_matrix[i][i] = 1
    return exchange_matrix


if __name__ == '__main__':
    # x = np.random.random((2, 4, 3))
    # y = np.random.randint(low=0, high=3, size=[2, 4])
    # weight = np.array([[1, 1, 2], [2, 4, 8], [3, 2, 0]])
    #
    # x = torch.Tensor(x).cuda()
    # y = torch.LongTensor(y).cuda()
    # weight = torch.Tensor(weight).cuda()
    #
    # _, _, vocab_size = x.size()
    # loss_f = Loss_Function()
    # loss = nn.CrossEntropyLoss()
    # a, b = x.view(-1, vocab_size), y.view(-1)
    # loss_cross = loss(a, b)
    # print(loss_cross)
    # print(loss_f.Cross_Entropy_Loss(a, b))
    #
    # spatial_loss = loss_f.Spatial_Loss(weight, a, b)
    # print(spatial_loss)
    # token_list = [[1, 0, 5, 3, 6, 0, 9],
    #               [5, 0, 0, 0, 6, 5, 0, 1],
    #               [1, 5, 7, 0, 5, 3, 6]]
    # make_exchange_matrix(token_list, 10)
    pass
