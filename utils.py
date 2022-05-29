import math
import time

import pandas as pd


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
