import os
from collections import Counter
import pandas as pd
from dataset import DataSetRaw
import numpy as np
from embed.ctle import CTLE, train_ctle, CTLEEmbedding, PositionalEncoding, MaskedLM
import torch
from torch import nn
from downstream.transformer import Transformer, train_transformer

device = 'cuda:0'
embed_size = 128
embed_name = 'ctle'
task_epoch = 1
embed_epoch = 1
init_param = False
hidden_size = embed_size * 4

file_name = "test_data"
raw_df = pd.read_hdf(os.path.join('data/h5_data', file_name + ".h5"), key='data')
dataset = DataSetRaw(raw_df)

max_seq_len = Counter(dataset.df['user_index'].to_list()).most_common(1)[0][1]  # 轨迹的最大长度 862
embed_mat = np.random.uniform(low=-0.5 / embed_size, high=0.5 / embed_size, size=(dataset.num_loc, embed_size))

ctle_num_layers = 4
ctle_num_heads = 8
ctle_mask_prop = 0.2
ctle_detach = False
ctle_static = False

encoding_layer = PositionalEncoding(embed_size, max_seq_len)
obj_models = [MaskedLM(embed_size, dataset.num_loc)]
obj_models = nn.ModuleList(obj_models)

ctle_embedding = CTLEEmbedding(encoding_layer, embed_size, dataset.num_loc)
ctle_model = CTLE(ctle_embedding, hidden_size, num_layers=ctle_num_layers, num_heads=ctle_num_heads,
                  init_param=init_param, detach=ctle_detach)
embed_layer = train_ctle(dataset, ctle_model, obj_models, mask_prop=ctle_mask_prop,
                         num_epoch=embed_epoch, batch_size=64, device=device)

transformer = Transformer(embedding_layer=embed_layer, embed_size=embed_size, n_layers=6, num_loc=dataset.num_loc
                          ).to(device)
train_transformer(dataset, max_seq_len=max_seq_len, transformer=transformer, num_epoch=10, batch_size=16, device=device)
