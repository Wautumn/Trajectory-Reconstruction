import random
from random import shuffle
import pandas as pd
import os
import time
from utils import traj_to_slot


def gen_train_data(file_path, file_name):
    seqs = []
    abs_file = os.path.join(file_path, file_name)
    data = pd.read_csv(abs_file)
    format = '%Y-%m-%d %H:%M:%S'

    data['timestamp'] = data["datetime"]
    data['date'] = data["datetime"].apply(lambda x: time.strftime(format, time.localtime(x)))
    data['day'] = pd.to_datetime(data['date'], errors='coerce').dt.day
    for (user_index, day), group in data.groupby(["user_index", 'day']):
        if group.shape[0] < 24:
            continue
        group.sort_values(by="timestamp")
        ts = group['timestamp'].astype(int).tolist()
        seq = group['loc_index'].tolist()
        seq = traj_to_slot(seq, ts, pad='[PAD]')
        if seq.count('[PAD]') > 24:
            continue
        seq = [str(x) for x in seq]
        seqs.append([" ".join(seq), user_index, day])
    print("All data length is " + str(len(seqs)))
    breakpoint()
    seqs = pd.DataFrame(seqs)
    indexes = ['trajectory', 'user_index', 'day']
    seqs.columns = indexes
    h5 = pd.HDFStore('data/h5_data_all/%s.h5' % "all_traj", 'w')
    h5['data'] = seqs
    h5.close()


class ConstructDataSet:
    def __init__(self, data_df, n_pred):
        # ['trajectory', 'user_index', 'day']
        self.data = data_df
        self.train_seq = []
        self.test_seq = []
        self.train_token_set = set()
        self.train_user_set = set()
        self.n_pred = n_pred
        for index, row in self.data.iterrows():
            seq, user_index, day = row['trajectory'], row['user_index'], row['day']
            seq = list(seq.split())
            if day > 25:
                self.test_seq.append([seq, user_index, day])
            else:
                self.train_seq.append([seq, user_index, day])
                self.train_user_set.add(user_index)
                for loc in seq:
                    self.train_token_set.add(loc)

    def masked_test_seqs(self, test_seq):
        test_records = []
        for record in test_seq:
            seq = record[0]
            user_index = record[1]
            day = record[2]
            cand_maked_pos = [i for i, token in enumerate(seq) if seq[i] != '[PAD]']  # candidate masked position
            shuffle(cand_maked_pos)
            masked_tokens, masked_pos = [], []
            for pos in cand_maked_pos[:self.n_pred]:
                masked_pos.append(pos)
                masked_tokens.append(seq[pos])
                seq[pos] = '[MASK]'  # make mask
            seq = [item if item != 0 else '[PAD]' for item in seq]
            seq, masked_pos, masked_tokens = [str(x) for x in seq], \
                                             [str(x) for x in masked_pos], [str(x) for x in masked_tokens]
            test_records.append([" ".join(seq), " ".join(masked_pos), " ".join(masked_tokens), user_index, day])
        return test_records

    def store_train_data(self):
        train_data = pd.DataFrame(self.train_seq)
        indexes = ['trajectory', 'user_index', 'day']
        train_data.columns = indexes
        h5 = pd.HDFStore('data/h5_data_all/%s.h5' % "train_traj", 'w')
        h5['data'] = train_data
        h5.close()

    def store_test_data(self, masked_test_data, name):
        masked_test_data = pd.DataFrame(masked_test_data)
        indexes = ['trajectory', 'masked_pos', 'masked_tokens', 'user_index', 'day']
        masked_test_data.columns = indexes
        h5 = pd.HDFStore('data/h5_data_all/%s.h5' % name, 'w')
        h5['data'] = masked_test_data
        h5.close()

    def gen_train_test_data_1(self):
        # 数据集1：测试集中的基站和用户在训练中都出现过
        test_record = []
        for record in self.test_seq:
            sign = 0
            seq, user_index, day = record
            if user_index not in self.train_user_set:
                continue
            for loc in seq:
                if loc not in self.train_token_set:
                    sign = 1
                    break
            if not sign:
                test_record.append(record)
        masked_test_record = self.masked_test_seqs(test_record)
        print("All train length is " + str(len(self.train_seq)))
        print("All test length is " + str(len(masked_test_record)))
        self.store_test_data(masked_test_data=masked_test_record, name="test_traj_1")
        return self.train_seq, masked_test_record

    def gen_train_test_data_2(self):
        # 数据集2：测试集中的部分基站在训练集中未出现过，但是用户在训练集中出现
        test_record = []
        for record in self.test_seq:
            seq, user_index, day = record
            if user_index not in self.train_user_set:
                continue
            test_record.append(record)
        masked_test_record = self.masked_test_seqs(test_record)
        print("All train length is " + str(len(self.train_seq)))
        print("All test length is " + str(len(masked_test_record)))
        self.store_test_data(masked_test_data=masked_test_record, name="test_traj_2")
        return self.train_seq, masked_test_record

    def gen_train_test_data_3(self):
        # 原始数据集划分
        # 数据集3：训练集中的部分基站在训练集中未出现过，且用户在训练集中也未出现过
        masked_test_record = self.masked_test_seqs(self.test_seq)
        print("All train length is " + str(len(self.train_seq)))
        print("All test length is " + str(len(masked_test_record)))
        self.store_test_data(masked_test_data=masked_test_record, name="test_traj_3")
        return self.train_seq, self.test_seq

    def gen_train_test_data_4(self):
        # 数据集4：mask位置在7:00-19:00
        test_record = []
        for record in self.test_seq:
            sign = 0
            seq, user_index, day = record
            if user_index not in self.train_user_set:
                continue
            for loc in seq:
                if loc not in self.train_token_set:
                    sign = 1
                    break
            if not sign:
                test_record.append(record)
        test_masked_record = []
        for record in test_record:
            seq = record[0]
            user_index = record[1]
            day = record[2]
            cand_maked_pos = [i for i, token in enumerate(seq) if
                              seq[i] != '[PAD]' and 14 < i <= 38]  # candidate masked position
            shuffle(cand_maked_pos)
            masked_tokens, masked_pos = [], []
            for pos in cand_maked_pos[:self.n_pred]:
                masked_pos.append(pos)
                masked_tokens.append(seq[pos])
                seq[pos] = '[MASK]'  # make mask
            seq = [item if item != 0 else '[PAD]' for item in seq]
            seq, masked_pos, masked_tokens = [str(x) for x in seq], \
                                             [str(x) for x in masked_pos], [str(x) for x in masked_tokens]
            test_masked_record.append([" ".join(seq), " ".join(masked_pos), " ".join(masked_tokens), user_index, day])

        print("All train length is " + str(len(self.train_seq)))
        print("All test length is " + str(len(test_masked_record)))
        self.store_test_data(masked_test_data=test_masked_record, name="test_traj_4")
        return self.train_seq, test_masked_record

    # def gen_test_data(self):
    #     # ['trajectory', 'masked_pos', 'masked_tokens', 'user_index', 'day']
    #     records = []
    #     for index, row in self.test_df.iterrows():
    #         seq, masked_pos, masked_tokens, user_index, day = row['trajectory'], row['masked_pos'], \
    #                                                           row['masked_tokens'], row['user_index'], row['day']
    #         seq, masked_pos, masked_tokens = list(seq.split()), list(map(int, masked_pos.split())), \
    #                                          list(map(int, masked_tokens.split()))
    #         records.append([seq, masked_pos, masked_tokens])
    #     print("All test length is " + str(len(records)))
    #     return records


class DataSet:
    def __init__(self, train_df, test_df_1, test_df_2, test_df_3, test_df_4):
        self.train_df = train_df
        self.test_df_1 = test_df_1
        self.test_df_2 = test_df_2
        self.test_df_3 = test_df_3
        self.test_df_4 = test_df_4

    def gen_train_data(self):
        # ['trajectory', 'user_index', 'day']
        records = []
        for index, row in self.train_df.iterrows():
            seq, user_index, day = row['trajectory'], row['user_index'], row['day']
            # seq = list(seq.split())
            records.append(seq)
        print("All train length is " + str(len(records)))
        return records

    def gen_test_data(self, index):
        # ['trajectory', 'masked_pos', 'masked_tokens']
        if index == 1:
            test_df = self.test_df_1
        elif index == 2:
            test_df = self.test_df_2
        elif index == 3:
            test_df = self.test_df_3
        else:
            test_df = self.test_df_4

        records = []
        for index, row in test_df.iterrows():
            seq, masked_pos, masked_tokens = row['trajectory'], row['masked_pos'], row['masked_tokens']
            seq, masked_pos, masked_tokens = list(seq.split()), list(map(int, masked_pos.split())), \
                                             list(map(int, masked_tokens.split()))
            records.append([seq, masked_pos, masked_tokens])
        print("All test length is " + str(len(records)))
        return records


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

#
# def gen_train_data(path):
#     '''
#     row_data:['user_index', 'loc_index', 'datetime', 'lat', 'lng']
#     '''
#     names = ["0000" + str(i).zfill(2) + "_0" for i in range(46)]
#     format = '%Y-%m-%d %H:%M:%S'
#     seqs = []
#     for i in range(len(names)):
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
#             record.append([" ".join(seq), user_index, day])
#         seqs.extend(record)
#         print(abs_file + "," + str(len(record)))
#     print("All train length is " + str(len(seqs)))
#
#     seqs = pd.DataFrame(seqs)
#     indexes = ['trajectory', 'user_index', 'day']
#     seqs.columns = indexes
#     h5 = pd.HDFStore('data/h5_data/%s.h5' % "train_trajectory", 'w')
#     h5['data'] = seqs
#     h5.close()
#
#
# def gen_test_data(path, n_pred):
#     '''
#     row_data:['user_index', 'loc_index', 'datetime', 'lat', 'lng']
#     '''
#     names = ["0000" + str(i).zfill(2) + "_0" for i in range(46, 51)]
#     format = '%Y-%m-%d %H:%M:%S'
#     seqs = []
#     for i in range(len(names)):
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
#             seq = traj_to_slot(seq, ts)
#             if seq.count(0) > 24:
#                 continue
#             cand_maked_pos = [i for i, token in enumerate(seq) if seq[i] != 0]  # candidate masked position
#             shuffle(cand_maked_pos)
#             masked_tokens, masked_pos = [], []
#             for pos in cand_maked_pos[:n_pred]:
#                 masked_pos.append(pos)
#                 masked_tokens.append(seq[pos])
#                 seq[pos] = '[MASK]'  # make mask
#             seq = [item if item != 0 else '[PAD]' for item in seq]
#             seq, masked_pos, masked_tokens = [str(x) for x in seq], \
#                                              [str(x) for x in masked_pos], [str(x) for x in masked_tokens]
#             record.append([" ".join(seq), " ".join(masked_pos), " ".join(masked_tokens), user_index, day])
#         seqs.extend(record)
#         print(abs_file + "," + str(len(record)))
#     print("All test length is " + str(len(seqs)))
#     seqs = pd.DataFrame(seqs)
#     indexes = ['trajectory', 'masked_pos', 'masked_tokens', 'user_index', 'day']
#     seqs.columns = indexes
#     h5 = pd.HDFStore('data/h5_data/%s.h5' % "test_trajectory", 'w')
#     h5['data'] = seqs
#     h5.close()

#
# class DataSet:
#     def __init__(self, train_df, test_df):
#         self.train_df = train_df
#         self.test_df = test_df
#
#     def gen_train_data(self):
#         # ['trajectory', 'user_index', 'day']
#         records = []
#         for index, row in self.train_df.iterrows():
#             seq, user_index, day = row['trajectory'], row['user_index'], row['day']
#             seq = list(seq.split())
#             records.append(seq)
#         print("All train length is " + str(len(records)))
#         return records
#
#     def gen_test_data(self):
#         # ['trajectory', 'masked_pos', 'masked_tokens', 'user_index', 'day']
#         records = []
#         for index, row in self.test_df.iterrows():
#             seq, masked_pos, masked_tokens, user_index, day = row['trajectory'], row['masked_pos'], \
#                                                               row['masked_tokens'], row['user_index'], row['day']
#             seq, masked_pos, masked_tokens = list(seq.split()), list(map(int, masked_pos.split())), \
#                                              list(map(int, masked_tokens.split()))
#             records.append([seq, masked_pos, masked_tokens])
#         print("All test length is " + str(len(records)))
#         return records


if __name__ == '__main__':
    # gen_train_data("data", "jinchengTraj.csv")
    # gen_test_data("data/row_data", n_pred=5)
    # train_df = pd.read_hdf(os.path.join('data/h5_data', "train_trajectory" + ".h5"), key='data')
    # test_df = pd.read_hdf(os.path.join('data/h5_data', "test_trajectory" + ".h5"), key='data')
    # dataset = DataSet(train_df=train_df, test_df=test_df)
    # train_data = dataset.gen_train_data()
    # test_data = dataset.gen_test_data()

    df = pd.read_hdf(os.path.join('data/h5_data_all', "all_traj" + ".h5"), key='data')
    constructdataset = ConstructDataSet(data_df=df, n_pred=5)
    # constructdataset.gen_train_test_data_1()
    # constructdataset.gen_train_test_data_2()
    # constructdataset.gen_train_test_data_3()
    constructdataset.gen_train_test_data_4()

    # train_df = pd.read_hdf(os.path.join('data/h5_data_all', "train_traj" + ".h5"), key='data')
    # test_df_1 = pd.read_hdf(os.path.join('data/h5_data_all', "test_traj_1" + ".h5"), key='data')
    # test_df_2 = pd.read_hdf(os.path.join('data/h5_data_all', "test_traj_2" + ".h5"), key='data')
    # test_df_3 = pd.read_hdf(os.path.join('data/h5_data_all', "test_traj_3" + ".h5"), key='data')
    # test_df_4 = pd.read_hdf(os.path.join('data/h5_data_all', "test_traj_4" + ".h5"), key='data')
    # dataset = DataSet(train_df, test_df_1, test_df_2, test_df_3, test_df_4)

    pass
