import os
from collections import Counter
import nni
import pandas as pd
from preprocess import DataSet
import numpy as np
from model.ctle import CTLE, train_ctle, CTLEEmbedding, PositionalEncoding, MaskedLM
from torch import nn

param = nni.get_next_parameter()
device = 'cuda:5'
embed_size = int(param.get('embed_size', 128))
embed_name = param.get('embed_name', 'ctle')

task_name = param.get('task_name', 'loc_pre')
task_epoch = int(param.get('task_epoch', 1))

pre_len = int(param.get('pre_len', 3))
embed_epoch = int(param.get('embed_epoch', 5))
init_param = param.get('init_param', False)

hidden_size = embed_size * 4

file_name = "test_4"
raw_df = pd.read_hdf(os.path.join('data/h5_data', file_name + ".h5"), key='data')
dataset = DataSet(raw_df)
max_seq_len = Counter(dataset.df['user_index'].to_list()).most_common(1)[0][1]

id2coor_df = dataset.df[['loc_index', 'lat', 'lng']].drop_duplicates('loc_index').set_index('loc_index').sort_index()

embed_mat = np.random.uniform(low=-0.5 / embed_size, high=0.5 / embed_size, size=(dataset.num_loc, embed_size))

if embed_name == 'ctle':
    encoding_type = param.get('encoding_type', 'positional')
    ctle_num_layers = int(param.get('ctle_num_layers', 4))
    ctle_num_heads = int(param.get('ctle_num_heads', 8))
    ctle_mask_prop = param.get('ctle_mask_prop', 0.2)
    ctle_detach = param.get("ctle_detach", False)
    ctle_objective = param.get("ctle_objective", "mlm")
    ctle_static = param.get("ctle_static", False)

    encoding_layer = PositionalEncoding(embed_size, max_seq_len)
    obj_models = [MaskedLM(embed_size, dataset.num_loc)]
    obj_models = nn.ModuleList(obj_models)

    ctle_embedding = CTLEEmbedding(encoding_layer, embed_size, dataset.num_loc)
    ctle_model = CTLE(ctle_embedding, hidden_size, num_layers=ctle_num_layers, num_heads=ctle_num_heads,
                      init_param=init_param, detach=ctle_detach)
    embed_layer = train_ctle(dataset, ctle_model, obj_models, mask_prop=ctle_mask_prop,
                             num_epoch=embed_epoch, batch_size=64, device=device)
