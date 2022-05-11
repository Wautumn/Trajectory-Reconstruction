import collections

import pandas as pd
import os
import numpy as np
import time

format = '%Y-%m-%d %H:%M:%S'


def getSlot(datetime):
    format_time = time.strptime(datetime, format)
    day, hour, minute = format_time.tm_mday, format_time.tm_hour, format_time.tm_min
    slot = (day - 1) * 48 + hour * 2 + (0 if minute <= 30 else 1)
    return slot


# def slot_time(file_path, file_name, time_slot=30):
#     indexes = ['user_index', 'loc_index', 'datetime', 'lat', 'lng']
#     abs_file = os.path.join(file_path + file_name)
#     row = pd.read_csv(abs_file, header=None)
#     row["datetime"] = row[2].apply(lambda x: time.strftime(format, time.localtime(x + 1633017600)))
#     row['day'] = pd.to_datetime(row["datetime"], errors='coerce').dt.day
#     records = []
#     for (user_index, day), group in row.groupby([0, 'day']):
#         print(day)
#         one_set = [user_index, group[1].tolist(), group['datetime'].tolist(), group.shape[0]]
#         print(one_set)


# class DataSetSlot:
#     def __init__(self, raw_df, time_slot=30):
#         self.df = raw_df
#         self.user_id = raw_df['user_index']
#         self.num_user = len(set(self.user_id))
#         self.loc_id = raw_df["loc_index"]
#         self.num_loc = len(set(self.loc_id))
#         self.time_slot = time_slot
#
#     def gen_sequence(self):
#         data = pd.DataFrame(self.df, copy=True)
#         data['day'] = pd.to_datetime(data['datetime'], errors='coerce').dt.day
#         data['timestamp'] = pd.to_datetime(data['datetime'], errors='coerce').apply(lambda x: x.timestamp())
#         seqs = []
#         for (user_index, day), group in data.groupby(["user_index", 'day']):
#             locs = group['loc_index'].tolist()
#             times = group['datetime'].tolist()
#             assert len(locs) == len(times)
#             day_traj_map = collections.defaultdict(list)
#             day_traj = []
#             for i in range(len(locs)):
#                 slot = getSlot(times[i])
#                 day_traj_map[slot].append(locs[i])
#             for i in range(48):
#                 if len(day_traj_map[i]) == 0:
#                     day_traj.append(-1)
#                 else:
#                     day_traj.append(collections.Counter(day_traj_map[i]).most_common(1)[0][0])
#
#             if day_traj.count(-1) >= 24:
#                 continue
#             one_seq = [user_index, day_traj, day]
#             seqs.append(one_seq)
#
#         print(len(seqs))
#         # indexes = ['user_index', 'loc_seq', 'day']
#         # records = pd.DataFrame(seqs, columns=indexes, dtype=str)
#         # h5 = pd.HDFStore('data/h5_data/%s.h5' % "trajectory_data", 'w')
#         # h5['data'] = records
#         # h5.close()
#         return seqs


class DataSetRaw:
    def __init__(self, raw_df):
        self.df = raw_df
        self.user_id = raw_df['user_index']
        self.num_user = len(set(self.user_id))
        self.loc_id = raw_df["loc_index"]
        self.num_loc = len(set(self.loc_id))

    def gen_sequence(self, min_len = 5):
        data = pd.DataFrame(self.df, copy=True)
        data['day'] = pd.to_datetime(data['datetime'], errors='coerce').dt.day
        data['timestamp'] = pd.to_datetime(data['datetime'], errors='coerce').apply(lambda x: x.timestamp())
        seqs = []
        for (user_index, day), group in data.groupby(["user_index", 'day']):
            if group.shape[0] < min_len:
                continue
            one_seq = [user_index, group['loc_index'].tolist(), group['timestamp'].astype(int).tolist(), group.shape[0]]
            seqs.append(one_seq)
        print(len(seqs))
        return seqs


def read_row_data(path):
    '''
    row_data:['user_index', 'loc_index', 'datetime', 'lat', 'lng']
    :param path:
    :return:
    '''
    records = []
    indexes = ['user_index', 'loc_index', 'datetime', 'lat', 'lng']
    names = []
    for i in range(51):
        name = "0000" + str(i).zfill(2) + "_0"
        names.append(name)
    # for i in range(25):
    for i in range(len(names)):
        abs_file = os.path.join(path + names[i])
        data = pd.read_csv(abs_file, header=None)
        if len(records) == 0:
            records = data
        else:
            records = np.concatenate([records, data], axis=0)
        print(abs_file + "," + str(len(records)))

    records = pd.DataFrame(records, columns=indexes, dtype=str)
    h5 = pd.HDFStore('data/h5_data/%s.h5' % "complete_row_data", 'w')
    h5['data'] = records
    h5.close()


# row_path = "data/row_data/"
# read_row_data(row_path)

if __name__ == '__main__':
    df1 = pd.read_hdf(os.path.join('data/h5_data', "complete_row_data_1" + ".h5"), key='data')
    df2 = pd.read_hdf(os.path.join('data/h5_data', "complete_row_data_2" + ".h5"), key='data')
    raw_df = pd.concat([df1, df2], axis=0)
    dataset = DataSetRaw(raw_df)
    seq_set = dataset.gen_sequence()
    pass
