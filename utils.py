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


def get_evalution(ground_truth, logits_lm):
    '''
    :param ground_truth: true token
    :param logits_lm: predict logits [length,masked_len,vocab_size]
    :return: accuracy，recall_3、recall_5、map
    '''
    # accuracy
    # pred_acc = logits_lm.data.max(2)[1]
    # pred_acc = torch.topk(logits_lm, 1, dim=2)[1]
    pred_acc = logits_lm[:, :, 0]
    pred_acc = pred_acc.flatten().cpu().data.numpy()
    accuracy_token = 0
    for i in range(len(ground_truth)):
        if pred_acc[i] == ground_truth[i]:
            accuracy_token += 1
    print(accuracy_token, accuracy_token / len(ground_truth))
    accuracy_score = accuracy_token / len(ground_truth)

    # recall3
    # pred_recall3 = torch.topk(logits_lm, 3, dim=2)[1]
    pred_recall3 = logits_lm[:, :, 0:3]
    pred_recall3 = torch.flatten(pred_recall3, start_dim=0, end_dim=1).cpu().data.numpy()
    recall3_token = 0
    for i in range(len(ground_truth)):
        if ground_truth[i] in pred_recall3[i]:
            recall3_token += 1
    print(recall3_token, recall3_token / len(ground_truth))
    recall3_score = recall3_token / len(ground_truth)

    # recall5
    # pred_recall5 = torch.topk(logits_lm, 5, dim=2)[1]
    pred_recall5 = logits_lm[:, :, 0:5]
    pred_recall5 = torch.flatten(pred_recall5, start_dim=0, end_dim=1).cpu().data.numpy()
    recall5_token = 0
    for i in range(len(ground_truth)):
        if ground_truth[i] in pred_recall5[i]:
            recall5_token += 1
    print(recall5_token, recall5_token / len(ground_truth))
    recall5_score = recall5_token / len(ground_truth)

    return accuracy_score, recall3_score, recall5_score


class Loss_Function(nn.Module):
    def __init__(self):
        super(Loss_Function, self).__init__()

    def Spatial_Loss(self, weight, logit_lm, ground_truth):
        _, num_classes = logit_lm.size()
        p_i = torch.softmax(logit_lm, dim=1)
        # spatial_softmax = torch.softmax(weight, dim=1)

        # spatial_matrix = torch.Tensor([weight[item].data.cpu().numpy() for item in ground_truth]).to(device)
        # spatial_matrix =weight[item] for item in ground_truth
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


def make_exchange_matrix(token_list, token_size):
    token_list = [list(filter(lambda x: x > 3, token)) for token in token_list]
    exchange_matrix = [[0] * token_size for _ in range((token_size))]
    for token in token_list:
        for i in range(1, len(token)):
            if token[i] == token[i - 1]:
                continue
            exchange_matrix[token[i - 1]][token[i]] += 1
    # for i in range(token_size):
    #     for j in range(token_size):
    #         if exchange_matrix[i][j] != 0:
    #             exchange_matrix[i][j] = np.exp(exchange_matrix[i][j])
    for i in range(token_size):
        for j in range(token_size):
            if exchange_matrix[i][j] != 0:
                exchange_matrix[i][j] = exchange_matrix[i][j] / sum(exchange_matrix[i])
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
