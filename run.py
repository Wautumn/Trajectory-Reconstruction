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
task_name = 'loc_pre'
task_epoch = 1
pre_len = 3
embed_epoch = 1
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

transformer = Transformer(embedding_layer=embed_layer, embed_size=embed_size, n_layers=6).to(device)
train_transformer(dataset, transformer=transformer, num_epoch=2, batch_size=64, device=device)


def greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=enc_input.dtype).cuda()], -1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tgt_vocab["."]:
            terminal = True
        print(next_word)
    return dec_input


# Test
enc_inputs, _, _ = next(iter(loader))
enc_inputs = enc_inputs.cuda()
for i in range(len(enc_inputs)):
    greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=tgt_vocab["S"])
    predict, _, _, _ = model(enc_inputs[i].view(1, -1), greedy_dec_input)
    predict = predict.data.max(1, keepdim=True)[1]
    print(enc_inputs[i], '->', [idx2word[n.item()] for n in predict.squeeze()])
