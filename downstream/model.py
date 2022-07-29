import torch
import torch.nn as nn
import numpy as np
import math
import config as gl
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import optim

from utils import next_batch, get_evalution, make_exchange_matrix, Loss_Function

# device = 'cuda:1'
# n_layers = 6
# n_heads = 12
# d_model = 768
device = gl.get_value('device')
n_layers = gl.get_value('layer')
d_model = gl.get_value('d_model')
print('d_model', d_model)
n_heads = gl.get_value('head')

d_ff = 1024 * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
max_len = 50  # 轨迹的最大长度
temp_size = 31  # day 的个数
user_size = 10000  # user的个数

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]


def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionalEncoding(pl.LightningModule):
    def __init__(self, max_len, embed_size):
        super().__init__()
        pe = torch.zeros(max_len, embed_size).float()
        pe.requires_grad = False
    
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        ans = self.pe[:, :x.size(1)]
        return self.pe[:, :x.size(1)]
        # return self.pe[:, :x.size(1)]


class Embedding(pl.LightningModule):
    def __init__(self, vocab_size, id2embed):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.d_model = d_model
        self.pos_embed = PositionalEncoding(max_len, self.d_model)  # position embedding

        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x, user=None, temporal=None):

        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=self.device)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        embedding = self.tok_embed(x)+self.pos_embed(pos)
        
        # user = user.unsqueeze(1).expand_as(x).to(device)
        # temporal = temporal.unsqueeze(1).expand_as(x).to(device)
        # a = self.user_embed(user)
        # b = self.tem_embed(temporal)
        # embedding = embedding + self.tem_embed(user)
        return self.norm(embedding.to(torch.float32))

class POI_Embedding(pl.LightningModule):
    def __init__(self, vocab_size, id2embed):
        super(POI_Embedding, self).__init__()
        self.poi_vec_size = id2embed.size(1)        
        self.tok_embed =  id2embed.clone()

        self.norm = nn.LayerNorm(self.poi_vec_size)

    def forward(self, x, user=None, temporal=None):
        
        tok_embed = self.tok_embed.clone().to(x.device)
        for i in range(x.size(0)):
            if i == 0:
                token_embed = torch.unsqueeze(torch.index_select(tok_embed, dim=0, index=x[0]), dim=0)
            else:
                token_embed = torch.cat(
                    (token_embed, torch.unsqueeze(torch.index_select(tok_embed, dim=0, index=x[i]), dim=0)), dim=0)
        embedding = token_embed
        return self.norm(embedding.to(torch.float32))


class ScaledDotProductAttention(pl.LightningModule):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(pl.LightningModule):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # v_s: [batch_size, n_heads, seq_len, d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_heads * d_v)  # context: [batch_size, seq_len, n_heads * d_v]
        output = self.fc(context)
        return self.norm(output + residual)  # output: [batch_size, seq_len, d_model]


class PoswiseFeedForwardNet(pl.LightningModule):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(pl.LightningModule):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                         enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


class BERT(pl.LightningModule):
    def __init__(self, vocab_size, id2embed):
        super(BERT, self).__init__()
        self.vocab_size = vocab_size
        self.id2embed = id2embed
        self.embedding = Embedding(self.vocab_size, self.id2embed)
        self.poi_embedding = POI_Embedding(vocab_size,id2embed)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )

        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight

        self.linear1 = nn.Linear(d_model,d_model)
        self.linear2 = nn.Linear(id2embed.shape[1], d_model)

        self.linear_prior = nn.Linear(id2embed.shape[1], d_model,False)
        self.linear_next = nn.Linear(id2embed.shape[1], d_model,False)
        self.poi_em_size = id2embed.shape[1]

    def forward(self, input_ids, masked_pos, user_ids=None, temp_ids=None):
        
        output = self.embedding(input_ids, user_ids, temp_ids)  # [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)  # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output = layer(output, enc_self_attn_mask)

        masked_pos_poi = masked_pos[:, :, None].expand(-1, -1, self.poi_em_size)  # [batch_size, max_pred, d_model]
        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model)  # [batch_size, max_pred, d_model]
        
        poi_embedding_tensor = self.poi_embedding(input_ids)
        poi_prior = torch.gather(poi_embedding_tensor, 1, masked_pos_poi-1)
        poi_next = torch.gather(poi_embedding_tensor, 1, masked_pos_poi+1)
        h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size, max_pred, d_model]
        h_masked = self.activ2(self.linear(h_masked)+self.linear_prior(poi_prior)+ self.linear_next(poi_next))  # [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked)  # [batch_size, max_pred, vocab_size]
        return logits_lm


class Model(pl.LightningModule):
    def __init__(self, vocab_size, id2embed):
        super(Model, self).__init__()
        self.model = BERT(vocab_size=vocab_size, id2embed=id2embed)
        self.criterion = nn.CrossEntropyLoss()
        self.vocab_size = vocab_size

    def training_step(self, batch, batch_idx):
        input_ids, masked_tokens, masked_pos, user_ids, day_ids = batch

        logits_lm = self.model(input_ids, masked_pos, user_ids, day_ids)
        # train_predict = []
        # train_truth = []
        # train_truth.extend(masked_tokens.flatten().cpu().data.numpy())
        # train_predict.extend(logits_lm.data.max(2)[1].flatten().cpu().data.numpy())
        loss_lm = self.criterion(logits_lm.view(-1, self.vocab_size), masked_tokens.view(-1))  # for masked LM
        loss_lm = (loss_lm.float()).mean()
        loss = loss_lm

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, masked_tokens, masked_pos, user_ids, day_ids = batch
        logits_lm = self.model(input_ids, masked_pos, user_ids, day_ids)
        loss_lm = self.criterion(logits_lm.view(-1, self.vocab_size), masked_tokens.view(-1))  # for masked LM
        loss_lm = (loss_lm.float()).mean()
        loss = loss_lm
        self.log("val_loss", loss, prog_bar=True)
    


    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.model.parameters(), lr=0.001)
        # optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer
