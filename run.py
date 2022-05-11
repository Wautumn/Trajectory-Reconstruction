import os
from collections import Counter
import nni
import pandas as pd
from dataset import DataSetRaw
import numpy as np
from embed.ctle import CTLE, train_ctle, CTLEEmbedding, PositionalEncoding, MaskedLM
from torch import nn

device = 'cuda:5'
embed_size = 128
embed_name = 'ctle'
task_name = 'loc_pre'
task_epoch = 1
pre_len = 3
embed_epoch = 5
init_param = False
hidden_size = embed_size * 4

file_name = "test_data"
raw_df = pd.read_hdf(os.path.join('data/h5_data', file_name + ".h5"), key='data')
dataset = DataSetRaw(raw_df)

max_seq_len = Counter(dataset.df['user_index'].to_list()).most_common(1)[0][1]
embed_mat = np.random.uniform(low=-0.5 / embed_size, high=0.5 / embed_size, size=(dataset.num_loc, embed_size))

encoding_type = 'positional'
ctle_num_layers = 4
ctle_num_heads = 8
ctle_mask_prop = 0.2
ctle_detach = False
ctle_objective = "mlm"
ctle_static = False

encoding_layer = PositionalEncoding(embed_size, max_seq_len)
obj_models = [MaskedLM(embed_size, dataset.num_loc)]
obj_models = nn.ModuleList(obj_models)

ctle_embedding = CTLEEmbedding(encoding_layer, embed_size, dataset.num_loc)
ctle_model = CTLE(ctle_embedding, hidden_size, num_layers=ctle_num_layers, num_heads=ctle_num_heads,
                  init_param=init_param, detach=ctle_detach)
embed_layer = train_ctle(dataset, ctle_model, obj_models, mask_prop=ctle_mask_prop,
                         num_epoch=embed_epoch, batch_size=64, device=device)

# pre_model = ErppLocPredictor(embed_layer, input_size=embed_size, lstm_hidden_size=hidden_size,
#                              fc_hidden_size=hidden_size, output_size=dataset.num_loc, num_layers=2,
#                              seq2seq=pre_model_seq2seq)
# loc_prediction(dataset, pre_model, pre_len=pre_len, num_epoch=task_epoch,
#                batch_size=64, device=device)
