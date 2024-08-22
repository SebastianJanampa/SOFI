import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from . import multi_head_attention as mha
import copy

from typing import Optional, List

class MSCC(nn.Module):
    def __init__(self, ctrlc, args):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.ctrlc = ctrlc
        self.backbone = self.ctrlc.backbone

        if args.coarse_frozen:
            for n, p in self.named_parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = ctrlc.transformer.d_model, ctrlc.transformer.nhead

        self.transformer = Transformer(
                                    d_model=args.hidden_dim,
                                    dropout=args.dropout,
                                    nhead=args.nheads,
                                    num_encoder_layers=args.enc_layers,
                                    num_decoder_layers=args.dec_layers,
                                    dim_feedforward=args.dim_feedforward,
                                    return_intermediate_dec= args.aux_loss
                                    )

        self.zvp_embed = nn.Linear(hidden_dim, 3)
        self.fovy_embed = nn.Linear(hidden_dim, 1)
        self.hl_embed = nn.Linear(hidden_dim, 3)
        self.vline_class_embed = nn.Linear(hidden_dim, 1)
        self.hline_class_embed = nn.Linear(hidden_dim, 1)
        
        self.input_proj = nn.Conv2d(self.backbone.num_channels[-2], hidden_dim, kernel_size=1)        
        line_dim = 6        
        self.aux_loss = args.aux_loss   

    def forward(self, samples: NestedTensor, extra_samples):
        """ The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels            
        """
        extra_info = {}

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.ctrlc.backbone(samples)

        src1, mask1 = features[-1].decompose()
        assert mask1 is not None
        lines_ = extra_samples['lines']
        lmask = ~extra_samples['line_mask'].squeeze(2).bool()
        
        # vlines [bs, n, 3]
        lines = self.ctrlc._to_structure_tensor(lines_)
        lines_proj = self.ctrlc.input_line_proj(lines)

        hs1, _, _, _, _ = (
            self.ctrlc.transformer(src=self.ctrlc.input_proj(src1), mask=mask1,
                             query_embed=self.ctrlc.query_embed.weight,
                             tgt=lines_proj, 
                             tgt_key_padding_mask=lmask,
                             pos_embed=pos[-1]))


        src2, mask2 = features[-2].decompose()

        hs2, memory2, enc_attn, dec_self_attn, dec_cross_attn = (
            self.transformer(src=self.input_proj(src2), mask=mask2,
                             query_embed=hs1[-1],
                             tgt=lines_proj, 
                             tgt_key_padding_mask=lmask,
                             pos_embed=pos[-2]))

        outputs_zvp = self.zvp_embed(hs2[:,:,0,:]) # [n_dec_layer, bs, 3]
        outputs_zvp = F.normalize(outputs_zvp, p=2, dim=-1)

        outputs_fovy = self.fovy_embed(hs2[:,:,1,:]) # [n_dec_layer, bs, 1]
        outputs_fovy = outputs_fovy.sigmoid()*np.pi # 0 ~ 180

        outputs_hl = self.hl_embed(hs2[:,:,2,:]) # [n_dec_layer, bs, 3]
        outputs_hl = F.normalize(outputs_hl, p=2, dim=-1)

        outputs_vline_class = self.vline_class_embed(hs2[:,:,3:,:])
        outputs_hline_class = self.hline_class_embed(hs2[:,:,3:,:])

        # if self.training:

        #     outputs_zvp_coarse = self.ctrlc.zvp_embed(hs1[:,:,0,:]) # [n_dec_layer, bs, 3]
        #     outputs_zvp_coarse = F.normalize(outputs_zvp_coarse, p=2, dim=-1)

        #     outputs_fovy_coarse = self.ctrlc.fovy_embed(hs1[:,:,1,:]) # [n_dec_layer, bs, 1]
        #     outputs_fovy_coarse = outputs_fovy_coarse.sigmoid()*np.pi # 0 ~ 180

        #     outputs_hl_coarse = self.ctrlc.hl_embed(hs1[:,:,2,:]) # [n_dec_layer, bs, 3]
        #     outputs_hl_coarse = F.normalize(outputs_hl_coarse, p=2, dim=-1)

        #     outputs_vline_class_coarse = self.ctrlc.vline_class_embed(hs1[:,:,3:,:])
        #     outputs_hline_class_coarse = self.ctrlc.hline_class_embed(hs1[:,:,3:,:])
            
        #     outputs_zvp = torch.cat([outputs_zvp_coarse, outputs_zvp], dim =0)
        #     outputs_fovy = torch.cat([outputs_fovy_coarse, outputs_fovy], dim =0)
        #     outputs_hl = torch.cat([outputs_hl_coarse, outputs_hl], dim =0)
        #     outputs_vline_class = torch.cat([outputs_vline_class_coarse, outputs_vline_class], dim =0)
        #     outputs_hline_class = torch.cat([outputs_hline_class_coarse, outputs_hline_class], dim =0)

        out = {
            'pred_zvp': outputs_zvp[-1], 
            'pred_fovy': outputs_fovy[-1],
            'pred_hl': outputs_hl[-1],
            'pred_vline_logits': outputs_vline_class[-1], # [bs, n, 1]
            'pred_hline_logits': outputs_hline_class[-1], # [bs, n, 1]
        }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_zvp, 
                                                    outputs_fovy, 
                                                    outputs_hl, 
                                                    outputs_vline_class,
                                                    outputs_hline_class)
        return out#, extra_info

    @torch.jit.unused
    def _set_aux_loss(self, outputs_zvp, outputs_fovy, outputs_hl, 
                      outputs_vline_class, outputs_hline_class):
        return [{'pred_zvp': a, 'pred_fovy': b, 'pred_hl': c, 
                 'pred_vline_logits': d, 'pred_hline_logits': e}
                for a, b, c, d, e in zip(outputs_zvp[:-1], 
                                      outputs_fovy[:-1], 
                                      outputs_hl[:-1], 
                                      outputs_vline_class[:-1],
                                      outputs_hline_class[:-1])]

    def _to_structure_tensor(self, params):    
        (a,b,c) = torch.unbind(params, dim=-1)
        return torch.stack([a*a, a*b,
                            b*b, b*c,
                            c*c, c*a], dim=-1)
    
    def _evaluate_whls_zvp(self, weights, vlines):
        vlines = F.normalize(vlines, p=2, dim=-1)
        u, s, v = torch.svd(weights * vlines)
        return v[:, :, :, -1]


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, tgt, tgt_key_padding_mask, pos_embed):
        # flatten NxCxHxW to HWxNxC
        num_queries = query_embed.size(1)

        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.permute(1, 0, 2) 
        tgt = tgt.permute(1, 0, 2)
        query_pos = query_embed #torch.cat([query_embed, torch.zeros_like(tgt)], dim=0)
        tgt = torch.zeros_like(query_embed) # [n, bs, ch]
        tgt_key_padding_mask = torch.cat([torch.zeros(bs, 3, dtype=torch.bool, device=query_embed.device), tgt_key_padding_mask], dim=1)
        mask = mask.flatten(1)
        
        memory, enc_attns = self.encoder(src, 
                                                src_key_padding_mask=mask, 
                                                pos=pos_embed)
        hs, dec_self_attns, dec_cross_attns = self.decoder(tgt, memory,
                                            tgt_key_padding_mask=tgt_key_padding_mask,
                                            memory_key_padding_mask=mask,
                                            pos=pos_embed, 
                                            query_pos=query_pos)
        # enc_attn_weights [enc_ayer, bs, nhead, h*w, h*w]
        # dec_attn_weights [dec_ayer, bs, nhead, n_qeury, h*w]
        # import pdb; pdb.set_trace()

        return (hs.transpose(1, 2), # hs [dec_ayer, bs, n, ch]
                memory.permute(1, 2, 0).view(bs, c, h, w), 
                enc_attns.permute(1, 0, 2, 3).view(bs, self.encoder.num_layers,
                                                          h, w,  # dim of target
                                                          h, w), # dim of source
                dec_self_attns.permute(1, 0, 2, 3).view(bs, self.decoder.num_layers,
                                                        num_queries,  # dim of target
                                                        num_queries), # dim of source
                dec_cross_attns.permute(1, 0, 2, 3).view(bs, self.decoder.num_layers,
                                                        num_queries,  # dim of target
                                                        h, w))        # dim of source

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        attn_weights = []

        for layer in self.layers:
            output, attn_weight = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            attn_weights.append(attn_weight)
        attn_weights = torch.stack(attn_weights)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_weights


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        self_attn_weights = []
        cross_attn_weights = []

        for layer in self.layers:
            output, self_attn_weight, cross_attn_weight = (
                layer(output, memory, tgt_mask=tgt_mask,
                      memory_mask=memory_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask,
                      pos=pos, query_pos=query_pos)
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
            self_attn_weights.append(self_attn_weight)
            cross_attn_weights.append(cross_attn_weight)
        self_attn_weights = torch.stack(self_attn_weights)
        cross_attn_weights = torch.stack(cross_attn_weights)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), self_attn_weights, cross_attn_weights

        return output.unsqueeze(0), self_attn_weights, cross_attn_weights


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = mha.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2, attn_weight = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weight

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, attn_weight = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, attn_weight

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = mha.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = mha.MultiheadAttention(d_model, nhead, dropout=dropout)
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
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2, self_attn_weight = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, cross_attn_weight = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, self_attn_weight, cross_attn_weight

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2, self_attn_weight = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, cross_attn_weight = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, self_attn_weight, cross_attn_weight

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
