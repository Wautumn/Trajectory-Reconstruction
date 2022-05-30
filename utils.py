import math
import time

import pandas as pd
import torch


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
    pred_acc = torch.topk(logits_lm, 1, dim=2)[1]
    pred_acc = pred_acc.flatten().cpu().data.numpy()
    accuracy_token = 0
    for i in range(len(ground_truth)):
        if pred_acc[i] == ground_truth[i]:
            accuracy_token += 1
    print(accuracy_token, accuracy_token / len(ground_truth))
    accuracy_score = accuracy_token / len(ground_truth)

    # recall3
    pred_recall3 = torch.topk(logits_lm, 3, dim=2)[1]
    pred_recall3 = torch.flatten(pred_recall3, start_dim=0, end_dim=1).cpu().data.numpy()
    recall3_token = 0
    for i in range(len(ground_truth)):
        if ground_truth[i] in pred_recall3[i]:
            recall3_token += 1
    print(recall3_token, recall3_token / len(ground_truth))
    recall3_score = recall3_token / len(ground_truth)

    # recall5
    pred_recall5 = torch.topk(logits_lm, 5, dim=2)[1]
    pred_recall5 = torch.flatten(pred_recall5, start_dim=0, end_dim=1).cpu().data.numpy()
    recall5_token = 0
    for i in range(len(ground_truth)):
        if ground_truth[i] in pred_recall5[i]:
            recall5_token += 1
    print(recall5_token, recall5_token / len(ground_truth))
    recall5_score = recall5_token / len(ground_truth)

    # mAP
    combine = []
    pred_prob = torch.topk(logits_lm, 1, dim=2)

    return accuracy_score, recall3_score, recall5_score
