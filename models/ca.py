import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class AddAndNorm(nn.Module):

    def __init__(self, size, dropout):
        super(AddAndNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        return self.norm(x + self.dropout(y))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(AddAndNorm(size, dropout), 2)
        self.size = size

    def forward(self, q, k, v, mask):
        x = self.sublayer[0](v, self.self_attn(q, k, v, mask))
        x = self.sublayer[1](x, self.feed_forward(x))
        return x


class Encoder(nn.Module):

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.layer1 = clones(layer, N)
        self.layer2 = clones(layer, N)

    def forward(self, x_im, x_text, mask):
        for layer1, layer2 in zip(self.layer1, self.layer2):
            # 在此交换Q exchange Q here
            # layer1 处理 sk - layer1 process sk
            # x_text1 = layer1(x_text, x_im, x_text, mask)
            # layer2 处理 im - layer2 process im
            x_im = layer2(x_im, x_text, x_im, mask)
            # x_sk = x_text1
        return x_im

def attention(query, key, value, dropout=None, mask=None, pos=None):
    """
    dk = dv = dmodel/h = 64,h=8
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """

        :param query: size(batch,seq,512)
        :param key:
        :param value:
        :param mask:
        :return:
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # size(batch,h,seq,dk)
        query, key, value = \
            [lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for lin, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """
    d_model = 512
    d_ff = 2048 为论文中数值
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Cross_Attention(nn.Module):
    def __init__(self, h=8, n=1, d_model=768, d_ff=1024, dropout=0.1): #(self, args, h=8, n=1, d_model=768, d_ff=1024, dropout=0.1):
        super(Cross_Attention, self).__init__()
        multi_head_attention = MultiHeadedAttention(h, d_model)
        ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        encoderLayer = EncoderLayer(d_model, multi_head_attention, ffn, dropout)
        self.encoder = Encoder(encoderLayer, n)
        self.text_projection = nn.Linear(512, d_model)
        
    def forward(self, x_patch,x_text):
        length = x_text.shape[0]
        x_text = self.text_projection(x_text)
        x_sketch= self.encoder(x_patch, x_text, None)  # 不要mask - don't mask
        return x_sketch


"""
class Cross_Attention(nn.Module):
    def __init__(self, h=8, n=1, d_model=768, d_ff=1024, dropout=0.1):
        super().__init__()
        multi_head_attention = MultiHeadedAttention(h, d_model, dropout=dropout)
        ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        encoderLayer = EncoderLayer(d_model, multi_head_attention, ffn, dropout)
        self.encoder = Encoder(encoderLayer, n)
        self.text_projection = nn.Linear(512, d_model)

    def forward(self, x_patch, x_text):
        
        # x_patch: [B, N, D_model]  (visual tokens)
        # x_text : [B, 512] or [B, T, 512] (text tokens or pooled text)
        # return : [B, N, D_model]
        
        if x_text.dim() == 2:
            x_text = x_text.unsqueeze(1)  # [B, 1, 512]

        x_text = self.text_projection(x_text)  # [B, T, D_model]
        x_out = self.encoder(x_patch, x_text, None)  # Q from x_patch inside Encoder.forward (see below)
        return x_out
"""


class CrossAttnRefiner(nn.Module):
    def __init__(self, vis_dim=768, text_dim=512, num_heads=8, only_patches=True):
        super().__init__()
        self.only_patches = only_patches

        # 1. 维度对齐：如果维度不同，需要投影或使用 multihead 的内建投影
        # 这里建议显式将 text_dim 投影到 vis_dim，方便残差连接
        self.text_proj = nn.Linear(text_dim, vis_dim)

        self.ln_q = nn.LayerNorm(vis_dim)
        self.ln_kv = nn.LayerNorm(vis_dim)

        # embed_dim 必须等于 Query 的维度
        self.attn = nn.MultiheadAttention(embed_dim=vis_dim, num_heads=num_heads, batch_first=False)
        self.proj = nn.Linear(vis_dim, vis_dim)

    def forward(self, x_layer, text_ctx):
        """
        x_layer: [197, B, 768] (Sequence first)
        text_ctx: [B, 512]     (Batch of text features)
        return:  [197, B, 768]
        """
        # 1. 分离 CLS 和 Patches
        if self.only_patches:
            cls_tok = x_layer[:1, :, :]  # [1, B, 768]
            x = x_layer[1:, :, :]  # [196, B, 768]
        else:
            x = x_layer

        # 2. 文本特征维度对齐并增加序列维度
        # text_ctx: [B, 512] -> [B, 512] -> [B, 768] -> [1, B, 768] (作为 KV 序列长度为1)
        kv = self.text_proj(text_ctx).unsqueeze(0)

        # 3. 归一化 (MultiheadAttention batch_first=False, 输入需为 [S, B, D])
        q = self.ln_q(x)
        kv = self.ln_kv(kv)

        # 4. 交叉注意力
        # Q: [196, B, 768], K: [1, B, 768], V: [1, B, 768]
        out, _ = self.attn(q, kv, kv, need_weights=False)

        # 5. 残差与投影
        out = self.proj(out)
        x_new = x + out

        # 6. 合并回原始形状
        if self.only_patches:
            return torch.cat([cls_tok, x_new], dim=0)  # [197, B, 768]
        else:
            return x_new