import random
from random import shuffle
import pandas as pd
import os
import time

from utils import traj_to_slot


# def gen_data(path, train_prop=0.9):
#     names = ["0000" + str(i).zfill(2) + "_0" for i in range(2)]
#     format = '%Y-%m-%d %H:%M:%S'
#     seqs = []
#     for i in range(len(names)):
#         print(names[i])
#         record = []
#         abs_file = os.path.join(path, names[i])
#         data = pd.read_csv(abs_file, header=None)
#         data['timestamp'] = data[2].apply(lambda x: x + 1633017600)
#         data['datetime'] = data[2].apply(lambda x: time.strftime(format, time.localtime(x + 1633017600)))
#         data['day'] = pd.to_datetime(data['datetime'], errors='coerce').dt.day
#         data['user_index'] = data[0]
#         data['loc_index'] = data[1]
#         for (user_index, day), group in data.groupby(["user_index", 'day']):
#             if group.shape[0] < 12:
#                 continue
#             group.sort_values(by="timestamp")
#             ts = group['timestamp'].astype(int).tolist()
#             seq = group['loc_index'].tolist()
#             seq = traj_to_slot(seq, ts, pad='[PAD]')
#             if seq.count('[PAD]') > 24:
#                 continue
#             seq = [str(x) for x in seq]
#             record.append([seq, user_index, day])
#         seqs.extend(record)
#         print(abs_file + "," + str(len(record)))
#     all_length = len(seqs)
#     random.shuffle(seqs)
#     train_size = int(all_length * train_prop)
#     train_list = seqs[0:train_size]
#     test_list = seqs[train_size:]
#     train_token_set = set(str(train_list[i][0][j]) for i in range(len(train_list)) for j in range(len(train_list[i][0])))
#
#     print("All train length is " + str(len(seqs)))
#     seqs = pd.DataFrame(seqs)
#     indexes = ['trajectory', 'user_index', 'day']
#     seqs.columns = indexes
#     # h5 = pd.HDFStore('data/h5_data/%s.h5' % "train_trajectory", 'w')
#     # h5['data'] = seqs
#     # h5.close()


def gen_train_data(path):
    '''
    row_data:['user_index', 'loc_index', 'datetime', 'lat', 'lng']
    '''
    names = ["0000" + str(i).zfill(2) + "_0" for i in range(46)]
    format = '%Y-%m-%d %H:%M:%S'
    seqs = []
    for i in range(len(names)):
        record = []
        abs_file = os.path.join(path, names[i])
        data = pd.read_csv(abs_file, header=None)
        data['timestamp'] = data[2].apply(lambda x: x + 1633017600)
        data['datetime'] = data[2].apply(lambda x: time.strftime(format, time.localtime(x + 1633017600)))
        data['day'] = pd.to_datetime(data['datetime'], errors='coerce').dt.day
        data['user_index'] = data[0]
        data['loc_index'] = data[1]
        for (user_index, day), group in data.groupby(["user_index", 'day']):
            group.sort_values(by="timestamp")
            ts = group['timestamp'].astype(int).tolist()
            seq = group['loc_index'].tolist()
            seq = traj_to_slot(seq, ts, pad='[PAD]')
            if seq.count('[PAD]') > 24:
                continue
            seq = [str(x) for x in seq]
            record.append([" ".join(seq), user_index, day])
        seqs.extend(record)
        print(abs_file + "," + str(len(record)))
    print("All train length is " + str(len(seqs)))
    seqs = pd.DataFrame(seqs)
    indexes = ['trajectory', 'user_index', 'day']
    seqs.columns = indexes
    h5 = pd.HDFStore('data/h5_data/%s.h5' % "train_trajectory", 'w')
    h5['data'] = seqs
    h5.close()


def gen_test_data(path, n_pred):
    '''
    row_data:['user_index', 'loc_index', 'datetime', 'lat', 'lng']
    '''
    names = ["0000" + str(i).zfill(2) + "_0" for i in range(46, 51)]
    format = '%Y-%m-%d %H:%M:%S'
    seqs = []
    for i in range(len(names)):
        record = []
        abs_file = os.path.join(path, names[i])
        data = pd.read_csv(abs_file, header=None)
        data['timestamp'] = data[2].apply(lambda x: x + 1633017600)
        data['datetime'] = data[2].apply(lambda x: time.strftime(format, time.localtime(x + 1633017600)))
        data['day'] = pd.to_datetime(data['datetime'], errors='coerce').dt.day
        data['user_index'] = data[0]
        data['loc_index'] = data[1]
        for (user_index, day), group in data.groupby(["user_index", 'day']):
            group.sort_values(by="timestamp")
            ts = group['timestamp'].astype(int).tolist()
            seq = group['loc_index'].tolist()
            seq = traj_to_slot(seq, ts)
            if seq.count(0) > 24:
                continue
            cand_maked_pos = [i for i, token in enumerate(seq) if seq[i] != 0]  # candidate masked position
            shuffle(cand_maked_pos)
            masked_tokens, masked_pos = [], []
            for pos in cand_maked_pos[:n_pred]:
                masked_pos.append(pos)
                masked_tokens.append(seq[pos])
                seq[pos] = '[MASK]'  # make mask
            seq = [item if item != 0 else '[PAD]' for item in seq]
            seq, masked_pos, masked_tokens = [str(x) for x in seq], \
                                             [str(x) for x in masked_pos], [str(x) for x in masked_tokens]
            record.append([" ".join(seq), " ".join(masked_pos), " ".join(masked_tokens), user_index, day])
        seqs.extend(record)
        print(abs_file + "," + str(len(record)))
    print("All test length is " + str(len(seqs)))
    seqs = pd.DataFrame(seqs)
    indexes = ['trajectory', 'masked_pos', 'masked_tokens', 'user_index', 'day']
    seqs.columns = indexes
    h5 = pd.HDFStore('data/h5_data/%s.h5' % "test_trajectory", 'w')
    h5['data'] = seqs
    h5.close()


class DataSet:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df

    def gen_train_data(self):
        # ['trajectory', 'user_index', 'day']
        records = []
        for index, row in self.train_df.iterrows():
            seq, user_index, day = row['trajectory'], row['user_index'], row['day']
            seq = list(seq.split())
            records.append(seq)
        print("All train length is " + str(len(records)))
        return records

    def gen_test_data(self):
        # ['trajectory', 'masked_pos', 'masked_tokens', 'user_index', 'day']
        records = []
        for index, row in self.test_df.iterrows():
            seq, masked_pos, masked_tokens, user_index, day = row['trajectory'], row['masked_pos'], \
                                                              row['masked_tokens'], row['user_index'], row['day']
            seq, masked_pos, masked_tokens = list(seq.split()), list(map(int, masked_pos.split())), \
                                             list(map(int, masked_tokens.split()))
            records.append([seq, masked_pos, masked_tokens])
        print("All test length is " + str(len(records)))
        return records


if __name__ == '__main__':
    # gen_train_data("data/row_data")
    # gen_test_data("data/row_data", n_pred=5)
    # train_df = pd.read_hdf(os.path.join('data/h5_data', "train_trajectory" + ".h5"), key='data')
    # test_df = pd.read_hdf(os.path.join('data/h5_data', "test_trajectory" + ".h5"), key='data')
    # dataset = DataSet(train_df=train_df, test_df=test_df)
    # train_data = dataset.gen_train_data()
    # test_data = dataset.gen_test_data()
    pass
