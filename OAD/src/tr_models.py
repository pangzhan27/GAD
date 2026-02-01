import copy

import torch
assert torch.__version__ >= '1.6.0'
import torch.nn as nn
import torch.nn.functional as F


def layer_norm(d_model, condition=True):
    return nn.LayerNorm(d_model) if condition else None

class DotProductAttention(nn.Module):

    def __init__(self, dropout=0.0, alpha=1.0, num_heads=8):
        super(DotProductAttention, self).__init__()

        self.dropout = dropout
        self.alpha = alpha
        self.num_heads = num_heads
        self.decay_mat = None

    def forward(self, q, k, v, attn_mask=None, knn=False, ratio = 0.75):
        B, N1, N2 = q.shape[0], q.shape[-2], k.shape[-2]
        attn_output_weights1 = torch.bmm(q, k.transpose(1, 2))
        # a = attn_output_weights1.detach().cpu().float().numpy()

        if attn_mask is not None:
            attn_output_weights1 += attn_mask

        if self.alpha != 1.0 and self.decay_mat is None:
            self.decay_mat = torch.vander(torch.tensor([self.alpha] * self.num_heads), N=v.shape[1])
            self.decay_mat = self.decay_mat.to(v.device)
        if self.alpha != 1.0:
            bs = attn_output_weights1.shape[0] // self.num_heads
            decay_mat = self.decay_mat.unsqueeze(0).repeat(bs, 1, 1)
            decay_mat = decay_mat.view(-1, *decay_mat.shape[2:])
            attn_output_weights = torch.log(decay_mat[:, None, :] + 1e-10) + attn_output_weights1

        if knn:
            mask=torch.zeros(B,N1,N2,device=q.device,requires_grad=False)
            index=torch.topk(attn_output_weights,k=int(N2 * ratio),dim=-1,largest=True)[1]
            mask.scatter_(-1,index,1.)
            attn_output_weights = torch.where(mask > 0, attn_output_weights, torch.full_like(attn_output_weights, float('-inf')))


        attn_output_weights = F.softmax(attn_output_weights1, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights,
                                        p=self.dropout,
                                        training=self.training)
        attn_output = torch.bmm(attn_output_weights, v)

        return attn_output

class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, kdim=None, vdim=None,
                 attention_type='dotproduct', decay_alpha=1.0):
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        if self._qkv_same_embed_dim:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        else:
            raise RuntimeError('Do not support q, k, v have different dimensions')

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0)
            nn.init.constant_(self.out_proj.bias, 0)


        if attention_type == 'dotproduct':
            self.attention = DotProductAttention(dropout, decay_alpha, num_heads)
        else:
            raise RuntimeError('attention_type should be [dotproduct | linear]')

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None, knn=False, ratio = 0.75):
        tsz, bsz, embed_dim = q.shape[0], q.shape[1], q.shape[2]
        # a1 = k.detach().cpu().float().numpy()
        # b1 = q.detach().cpu().float().numpy()
        # c1 = self.in_proj_bias.detach().cpu().float().numpy()
        # d1 = self.in_proj_weight.detach().cpu().float().numpy()

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, \
            'embed_dim must be divisible by num_heads'
        scaling = float(head_dim) ** -0.5

        _b = self.in_proj_bias
        _start = None
        _end = embed_dim
        _w = self.in_proj_weight[:_end, :]
        if _b is not None:
            _b = _b[:_end]
        q = F.linear(q, _w, _b)

        _b = self.in_proj_bias
        _start = embed_dim
        _end = embed_dim * 2
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k = F.linear(k, _w, _b)

        _b = self.in_proj_bias
        _start = embed_dim * 2
        _end = None
        _w = self.in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]
        v = F.linear(v, _w, _b)

        q = q * scaling

        q = q.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).repeat(bsz, 1, 1)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn_mask = attn_mask.reshape(-1, *attn_mask.shape[2:])

        if key_padding_mask is not None:
            if key_padding_mask.ndim == 4:
                key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, tsz, 1, 1, 1)
                key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1, 1)
                key_padding_mask = key_padding_mask.reshape(*key_padding_mask.shape[:3], -1)
            else:
                key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, tsz, 1)
                key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            key_padding_mask = key_padding_mask.reshape(-1, *key_padding_mask.shape[2:])
            # b = key_padding_mask.detach().cpu().numpy()

        if attn_mask is not None and key_padding_mask is not None:
            # @ added by Pang to deal with different shape
            if attn_mask.shape != key_padding_mask.shape:
                mask = torch.cat((key_padding_mask, attn_mask), dim=-1)
            else:
                mask = attn_mask + key_padding_mask
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        attn_output = self.attention(q, k, v, mask, knn=knn, ratio=ratio)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tsz, bsz,
                                                                    self.embed_dim)
        # a = k.detach().cpu().float().numpy()
        # b = v.detach().cpu().float().numpy()
        # c = q.detach().cpu().float().numpy()
        return self.out_proj(attn_output), None


# class MultiheadAttention(nn.Module):
#
#     def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, kdim=None, vdim=None,
#                  attention_type='dotproduct', decay_alpha=1.0):
#         super(MultiheadAttention, self).__init__()
#
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.kdim = kdim if kdim is not None else embed_dim
#         self.vdim = vdim if vdim is not None else embed_dim
#         self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
#
#         if self._qkv_same_embed_dim:
#             self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
#         else:
#             raise RuntimeError('Do not support q, k, v have different dimensions')
#
#         if bias:
#             self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
#         else:
#             self.register_parameter('in_proj_bias', None)
#
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
#
#         if self._qkv_same_embed_dim:
#             nn.init.xavier_uniform_(self.in_proj_weight)
#
#         if self.in_proj_bias is not None:
#             nn.init.constant_(self.in_proj_bias, 0)
#             nn.init.constant_(self.out_proj.bias, 0)
#
#
#         if attention_type == 'dotproduct':
#             self.attention = DotProductAttention(dropout, decay_alpha, num_heads)
#         else:
#             raise RuntimeError('attention_type should be [dotproduct | linear]')
#
#     def forward(self, q, k, v, attn_mask=None, key_padding_mask=None, knn=False, ratio = 0.75):
#         tsz, bsz, embed_dim = q.shape[0], q.shape[1], q.shape[2]
#         a1 = k.detach().cpu().float().numpy()
#         b1 = q.detach().cpu().float().numpy()
#         c1 = self.in_proj_bias.detach().cpu().float().numpy()
#         d1 = self.in_proj_weight.detach().cpu().float().numpy()
#
#         head_dim = embed_dim // self.num_heads
#         assert head_dim * self.num_heads == embed_dim, \
#             'embed_dim must be divisible by num_heads'
#         scaling = float(head_dim) ** -0.5
#
#         _b = self.in_proj_bias
#         _start = None
#         _end = embed_dim
#         _w = self.in_proj_weight[:_end, :]
#         if _b is not None:
#             _b = _b[:_end]
#         q = F.linear(q, _w, _b)
#
#         _b = self.in_proj_bias
#         _start = embed_dim
#         _end = embed_dim * 2
#         _w = self.in_proj_weight[_start:_end, :]
#         if _b is not None:
#             _b = _b[_start:_end]
#         k = F.linear(k, _w, _b)
#
#         _b = self.in_proj_bias
#         _start = embed_dim * 2
#         _end = None
#         _w = self.in_proj_weight[_start:, :]
#         if _b is not None:
#             _b = _b[_start:]
#         v = F.linear(v, _w, _b)
#
#         q = q * scaling
#
#         q = q.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
#         k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
#         v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
#
#         if attn_mask is not None:
#             attn_mask = attn_mask.unsqueeze(0).repeat(bsz, 1, 1)
#             attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
#             attn_mask = attn_mask.reshape(-1, *attn_mask.shape[2:])
#
#         if key_padding_mask is not None:
#             if key_padding_mask.ndim == 4:
#                 key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, tsz, 1, 1, 1)
#                 key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1, 1)
#                 key_padding_mask = key_padding_mask.reshape(*key_padding_mask.shape[:3], -1)
#             else:
#                 key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, tsz, 1)
#                 key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
#             key_padding_mask = key_padding_mask.reshape(-1, *key_padding_mask.shape[2:])
#             b = key_padding_mask.detach().cpu().numpy()
#
#         if attn_mask is not None and key_padding_mask is not None:
#             # @ added by Pang to deal with different shape
#             if attn_mask.shape != key_padding_mask.shape:
#                 mask = torch.cat((key_padding_mask, attn_mask), dim=-1)
#             else:
#                 mask = attn_mask + key_padding_mask
#         elif attn_mask is not None:
#             mask = attn_mask
#         elif key_padding_mask is not None:
#             mask = key_padding_mask
#         else:
#             mask = None
#
#         attn_output = self.attention(q, k, v, mask, knn=knn, ratio=ratio)
#         attn_output = attn_output.transpose(0, 1).contiguous().view(tsz, bsz,
#                                                                     self.embed_dim)
#         a = k.detach().cpu().float().numpy()
#         b = v.detach().cpu().float().numpy()
#         c = q.detach().cpu().float().numpy()
#         return self.out_proj(attn_output), None
#


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, src_key_padding_mask=None, knn=False, ratio = 0.75):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=src_mask,
                         src_key_padding_mask=src_key_padding_mask, knn=knn, ratio=ratio)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def clear_cache(self):
        if len(self.layers) != 1:
            raise RuntimeError('Number of layers cannot larger than 1 for stream inference')

        self.layers[0].clear_cache()

    def stream_inference(self, tgt, memory, pos, tgt_mask=None,
                         memory_mask=None, tgt_key_padding_mask=None,
                         memory_key_padding_mask=None, cache_num=1, cache_id=0):
        output = tgt

        if len(self.layers) != 1:
            raise RuntimeError('Number of layers cannot larger than 1 for stream inference')

        output = self.layers[0].stream_inference(output, memory, pos,
                                                 tgt_mask=tgt_mask,
                                                 memory_mask=memory_mask,
                                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                                 memory_key_padding_mask=memory_key_padding_mask,
                                                 cache_num=cache_num, cache_id=cache_id)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, knn=False, ratio = 0.75):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         knn=knn, ratio=ratio)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, knn=False, ratio = 0.75):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, knn=knn, ratio=ratio)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='relu', attention_type='dotproduct',
                 decay_alpha=1.0):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                            attention_type=attention_type)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                                 attention_type=attention_type,
                                                 decay_alpha=decay_alpha)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        ############################
        # Cache for stream inference
        ############################
        self.tgt_cache = None

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def clear_cache(self):
        self.tgt_cache = None
        self.multihead_attn.clear_cache()

    def stream_inference(self, tgt, memory, pos, tgt_mask=None, memory_mask=None,
                         tgt_key_padding_mask=None, memory_key_padding_mask=None,
                         cache_num=1, cache_id=0):
        if self.tgt_cache is None:
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            self.tgt_cache = tgt
        else:
            tgt = self.tgt_cache
        tgt2 = self.multihead_attn.stream_inference(tgt, memory, memory, pos, attn_mask=memory_mask,
                                                    key_padding_mask=memory_key_padding_mask,
                                                    cache_num=cache_num, cache_id=cache_id)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                knn=False, ratio = 0.75):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask, knn=False)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask, knn=knn, ratio=ratio)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu

    raise RuntimeError('activation should be relu/gelu, not {}'.format(activation))
