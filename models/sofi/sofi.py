# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    pos_ = []

    for i in range (pos_tensor.size(-1)):
        embed = pos_tensor[:, :, i] * scale
        pos_embed = embed[:, :, None] / dim_t
        pos_embed = torch.stack((pos_embed[:, :, 0::2].sin(), pos_embed[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_.append(pos_embed)

    if pos_tensor.size(-1) == 3:
        pos = torch.cat(pos_, dim=2)
    elif pos_tensor.size(-1) == 6:
        pos = torch.cat((pos_[0], pos_[2], pos_[4], pos_[1], pos_[3], pos_[5]), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

class SOFI(nn.Module):
    """ This is the SOFI module that performs Camera Calibrations"""
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, mixed_query,
                 aux_loss=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of line classes
            num_queries: number of object queries
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        nheads = transformer.nhead
        group_norm = hidden_dim // nheads

        self.zvp_embed = nn.Linear(hidden_dim, 3)
        self.fovy_embed = nn.Linear(hidden_dim, 1)
        self.hl_embed = nn.Linear(hidden_dim, 3)
        self.line_class_embed = nn.Linear(hidden_dim, num_classes)
        self.line_score_embed = nn.Linear(hidden_dim, 1)

        line_dim = 6
        self.input_line_geometric_proj = nn.Linear(line_dim, hidden_dim) 

        self.q_mixed = mixed_query
        if self.q_mixed:
            self.n_samples = 16
            self.line_features_proj = nn.Conv2d(self.n_samples * num_feature_levels, 1, kernel_size=1, bias=False)
        

        self.num_feature_levels = num_feature_levels
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(group_norm, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(group_norm, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(group_norm, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.line_class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.line_score_embed.bias.data = torch.ones(1) * bias_value

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers
       
        self.zvp_embed = nn.ModuleList([self.zvp_embed for _ in range(num_pred)])
        self.fovy_embed = nn.ModuleList([self.fovy_embed for _ in range(num_pred)])
        self.hl_embed = nn.ModuleList([self.hl_embed for _ in range(num_pred)])
        self.line_class_embed = nn.ModuleList([self.line_class_embed for _ in range(num_pred)])
        self.line_score_embed = nn.ModuleList([self.line_score_embed for _ in range(num_pred)])

    @torch.no_grad()
    def sampling_points(self, endpoints, n_samples=16):
        """
        input: feature map (batch_size, C, H, W)
        a, b: endpoints of a line. [batch_size, n_lines, 2]
        """
        a, b = endpoints.split(2, -1)
        a = a.cpu().numpy()
        b = b.cpu().numpy()
        c = np.linspace(a, b, n_samples)
        c = c.transpose(1, 2, 0, 3) #[batch_size, n_lines, n_samples, 2]

        grid = torch.tensor(c).to(endpoints.device)#.contiguous()
        return grid 

    def _to_structure_tensor(self, params):    
        (a,b,c) = torch.unbind(params, dim=-1)
        return torch.stack([a*a, a*b,
                            b*b, b*c,
                            c*c, c*a], dim=-1)

    def forward(self, samples: NestedTensor, extra_samples):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - TODO
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        extra_info = {}
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        x, _ = samples.decompose()
        features, pos = self.backbone(samples)

        # Line processing
        L = extra_samples['lines']
        lmask = ~extra_samples['line_mask'].squeeze(2).bool()
        
        endpoints = extra_samples['segs']
        grid = self.sampling_points(endpoints)

        # vlines [bs, n, 3]
        l_geo = self._to_structure_tensor(L)
        l_geo = self.input_line_geometric_proj(l_geo)

        l_con = None
        if self.q_mixed:
            l_con = []

        srcs = []
        masks = []

        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            src_proj = self.input_proj[l](src)
            if self.q_mixed:
                context_info = F.grid_sample(src_proj, grid, align_corners=False)
                context_info = context_info.permute(0, 3, 2, 1)
                l_con.append(context_info)

            srcs.append(src_proj)
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)           
                if self.q_mixed:
                    context_info = F.grid_sample(src, grid, align_corners=False)
                    context_info = context_info.permute(0, 3, 2, 1)
                    l_con.append(context_info)

        if self.q_mixed:
            l_con = torch.cat(l_con, dim=1)
            l_con = self.line_features_proj(l_con).squeeze(1)
        else:
            l_con = torch.zeros_like(l_geo)

        query_embeds = self.query_embed.weight

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = (
            self.transformer(srcs=srcs, masks=masks, 
                             query_embed=self.query_embed.weight, pos_embeds=pos,
                             l_geo=l_geo, l_con=l_con, l_mask=lmask
                             ))

        outputs_zvps = []
        outputs_fovys = []
        outputs_hls = []
        outputs_scores = []
        outputs_classes = []
        outputs_vline_classes = []
        outputs_hline_classes = []
        for lvl in range(hs.shape[0]):
            outputs_zvp = self.zvp_embed[lvl](hs[lvl, :, 0, :])
            
            outputs_fovy = self.fovy_embed[lvl](hs[lvl, :, 1, :])

            outputs_hl = self.hl_embed[lvl](hs[lvl, :, 2, :])

            outputs_score = self.line_score_embed[lvl](hs[lvl, :, 3:, :])
            outputs_class = self.line_class_embed[lvl](hs[lvl, :, 3:, :])

            outputs_zvps.append(outputs_zvp)
            outputs_fovys.append(outputs_fovy)
            outputs_hls.append(outputs_hl)
            outputs_scores.append(outputs_score)
            outputs_classes.append(outputs_class)

        outputs_zvp = torch.stack(outputs_zvps)
        outputs_fovy = torch.stack(outputs_fovys)
        outputs_hl = torch.stack(outputs_hls)
        outputs_score = torch.stack(outputs_scores)
        outputs_class = torch.stack(outputs_classes)

        out = {
            'pred_zvp': outputs_zvp[-1], 
            'pred_fovy': outputs_fovy[-1],
            'pred_hl': outputs_hl[-1],
            'pred_line_score_logits': outputs_score[-1], # [bs, n, 1]
            'pred_line_class_logits': outputs_class[-1], # [bs, n, 1]
        }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_zvp, 
                                                    outputs_fovy, 
                                                    outputs_hl, 
                                                    outputs_score,
                                                    outputs_class)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_zvp, outputs_fovy, outputs_hl, 
                      outputs_score, outputs_class):
        return [{'pred_zvp': a, 'pred_fovy': b, 'pred_hl': c, 
                 'pred_line_score_logits': d, 'pred_line_class_logits': e}
                for a, b, c, d, e in zip(outputs_zvp[:-1], 
                                      outputs_fovy[:-1], 
                                      outputs_hl[:-1], 
                                      outputs_score[:-1],
                                      outputs_class[:-1])]


class SetCriterion(nn.Module):
    def __init__(self, weight_dict, losses, 
                       line_pos_angle, line_neg_angle):
        super().__init__()
        self.weight_dict = weight_dict        
        self.losses = losses
        self.thresh_line_pos = np.cos(np.radians(line_pos_angle), dtype=np.float32) # near 0.0
        self.thresh_line_neg = np.cos(np.radians(line_neg_angle), dtype=np.float32) # near 0.0
        # 
        
    def loss_zvp(self, outputs, targets, **kwargs):
        assert 'pred_zvp' in outputs
        src_zvp = outputs['pred_zvp']                                    
        target_zvp = torch.stack([t['zvp'] for t in targets], dim=0)

        cos_sim = F.cosine_similarity(src_zvp, target_zvp, dim=-1).abs()      
        loss_zvp_cos = (1.0 - cos_sim).mean()
                
        losses = {'loss_zvp': loss_zvp_cos}
        return losses

    def loss_fovy(self, outputs, targets, **kwargs):
        assert 'pred_fovy' in outputs
        src_fovy = outputs['pred_fovy']                                    
        target_fovy = torch.stack([t['fovy'] for t in targets], dim=0)
        
        loss_fovy_mae = F.l1_loss(src_fovy, target_fovy)
                        
        losses = {'loss_fovy': loss_fovy_mae}
        return losses

    def compute_hl(self, hl, dim=-1, eps=1e-6):        
        (a,b,c) = torch.split(hl, 1, dim=dim) # [b,3]
        hl = torch.where(b < 0.0, hl.neg(), hl)
        (a,b,c) = torch.split(hl, 1, dim=dim) # [b,3]
        b = torch.max(b, torch.tensor(eps, device=b.device))
        # compute horizon line
        left  = (a - c)/b  # [-1.0, ( a - c)/b]
        right = (-a - c)/b # [ 1.0, (-a - c)/b]
        return left, right

    def loss_hl(self, outputs, targets, **kwargs):
        assert 'pred_hl' in outputs
        src_hl = outputs['pred_hl']                                    
        target_hl = torch.stack([t['hl'] for t in targets], dim=0)
        target_hl = F.normalize(target_hl, p=2, dim=-1)

        src_l, src_r = self.compute_hl(src_hl)
        target_l, target_r = self.compute_hl(target_hl)

        error_l = (src_l - target_l).abs()
        error_r = (src_r - target_r).abs()
        loss_hl_max = torch.max(error_l, error_r).mean()

        losses = {'loss_hl': loss_hl_max,}
        return losses

    def project_points(self, pts, f=1.0, eps=1e-7):
        # project point on z=1 plane
        device = pts.device
        (x,y,z) = torch.split(pts, 1, dim=-1)
        de = torch.max(torch.abs(z), torch.tensor(eps, device=device))
        de = torch.where(z < 0.0, de.neg(), de)
        return f*(pts/de)

    def sigmoid_focal_loss(self,inputs, targets, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                   balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        return loss

    def loss_line_score(self, outputs, targets, **kwargs):
        assert 'pred_line_score_logits' in outputs
        src_logits = outputs['pred_line_score_logits'] # [bs, n, 1]        
        target_lines = torch.stack([t['lines'] for t in targets], dim=0) # [bs, n, 3]
        target_mask = torch.stack([t['line_mask'] for t in targets], dim=0) # [bs, n, 1]
        target_zvp = torch.stack([t['zvp'] for t in targets], dim=0) # [bs, 3]
        target_zvp = target_zvp.unsqueeze(1) # [bs, 1, 3]
        target_hvps = torch.stack([t['hvps'] for t in targets], dim=0) # [bs, 2, 3]

        with torch.no_grad():
            cos_sim = F.cosine_similarity(target_lines, target_zvp, dim=-1).abs()
            # [bs, n]
            cos_sim = cos_sim.unsqueeze(-1) # [bs, n, 1]

            ones = torch.ones_like(src_logits)
            zeros = torch.zeros_like(src_logits)
            target_classes_v = torch.where(cos_sim < self.thresh_line_pos, ones, zeros)
            mask = torch.where(torch.gt(cos_sim, self.thresh_line_pos) &
                               torch.lt(cos_sim, self.thresh_line_neg), 
                               zeros, ones) 
            
            # [bs, n, 1]            
            v_mask = target_mask*mask

            target_lines = target_lines.unsqueeze(dim=2) # [bs, n, 1, 3]
            target_hvps = target_hvps.unsqueeze(dim=1) # [bs, 1, 2, 3]
            
            cos_sim = F.cosine_similarity(target_lines, target_hvps, dim=-1).abs() # [bs, n, 2]
            cos_sim = cos_sim.min(dim=-1, keepdim=True)[0] # [bs, n, 1]
            
            target_classes_h = torch.where(cos_sim < self.thresh_line_pos, ones, zeros)
            mask = torch.where(torch.gt(cos_sim, self.thresh_line_pos) &
                               torch.lt(cos_sim, self.thresh_line_neg), 
                               zeros, ones)
            # [bs, n, 1]            
            h_mask = target_mask*mask
            
            mask = torch.logical_or(v_mask, h_mask)
            target_classes = torch.logical_or(target_classes_v, target_classes_h) * 1.0

        loss_ce = self.sigmoid_focal_loss(src_logits, target_classes)
        loss_ce = mask*loss_ce
        loss_ce = loss_ce.sum(dim=1)/mask.sum(dim=1)
        losses = {'loss_line_score': loss_ce.mean()}
        return losses

    def loss_line_class(self, outputs, targets, **kwargs):
        assert 'pred_line_class_logits' in outputs
        src_logits = outputs['pred_line_class_logits'] # [bs, n, 1]        
        target_lines = torch.stack([t['lines'] for t in targets], dim=0) # [bs, n, 3]
        target_mask = torch.stack([t['line_mask'] for t in targets], dim=0) # [bs, n, 1]
        target_zvp = torch.stack([t['zvp'] for t in targets], dim=0) # [bs, 3]
        target_zvp = target_zvp.unsqueeze(1) # [bs, 1, 3]
        target_hvps = torch.stack([t['hvps'] for t in targets], dim=0) # [bs, 2, 3]

        with torch.no_grad():
            cos_sim = F.cosine_similarity(target_lines, target_zvp, dim=-1).abs()
            # [bs, n]
            cos_sim = cos_sim.unsqueeze(-1) # [bs, n, 1]

            ones = torch.ones_like(src_logits)[...,:1]
            zeros = torch.zeros_like(src_logits)[...,:1]
            twos = ones + 1
            target_classes = torch.where(cos_sim < self.thresh_line_pos, ones, zeros)
            mask = torch.where(torch.gt(cos_sim, self.thresh_line_pos) &
                               torch.lt(cos_sim, self.thresh_line_neg), 
                               zeros, ones) 
            
            # [bs, n, 1]            
            v_mask = target_mask*mask

            target_lines = target_lines.unsqueeze(dim=2) # [bs, n, 1, 3]
            target_hvps = target_hvps.unsqueeze(dim=1) # [bs, 1, 2, 3]
            
            cos_sim = F.cosine_similarity(target_lines, target_hvps, dim=-1).abs() # [bs, n, 2]
            cos_sim = cos_sim.min(dim=-1, keepdim=True)[0] # [bs, n, 1]
            
            target_classes = torch.where(cos_sim < self.thresh_line_pos, twos, target_classes)
            mask = torch.where(torch.gt(cos_sim, self.thresh_line_pos) &
                               torch.lt(cos_sim, self.thresh_line_neg), 
                               zeros, ones)
            # [bs, n, 1]            
            h_mask = target_mask*mask
            mask = torch.logical_or(v_mask, h_mask) * 1

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        
        target_classes_onehot.scatter_(2, target_classes.long(), 1)
        target_classes_onehot = target_classes_onehot[:,:,:-1]

        loss_ce = self.sigmoid_focal_loss(src_logits, target_classes_onehot, mask)
        loss_ce = mask*loss_ce
        loss_ce = loss_ce.sum(dim=1)/mask.sum(dim=1)
        losses = {'loss_line_class': loss_ce.mean()}
        return losses

    def loss_vline_labels(self, outputs, targets, **kwargs):
        # positive < thresh_pos < no label < thresh_neg < negative
        assert 'pred_vline_logits' in outputs
        src_logits = outputs['pred_vline_logits'] # [bs, n, 1]        
        target_lines = torch.stack([t['lines'] for t in targets], dim=0) # [bs, n, 3]
        target_mask = torch.stack([t['line_mask'] for t in targets], dim=0) # [bs, n, 1]
        target_zvp = torch.stack([t['zvp'] for t in targets], dim=0) # [bs, 3]
        target_zvp = target_zvp.unsqueeze(1) # [bs, 1, 3]
        
        with torch.no_grad():
            cos_sim = F.cosine_similarity(target_lines, target_zvp, dim=-1).abs()
            # [bs, n]
            cos_sim = cos_sim.unsqueeze(-1) # [bs, n, 1]

            ones = torch.ones_like(src_logits)
            zeros = torch.zeros_like(src_logits)
            target_classes = torch.where(cos_sim < self.thresh_line_pos, ones, zeros)
            mask = torch.where(torch.gt(cos_sim, self.thresh_line_pos) &
                               torch.lt(cos_sim, self.thresh_line_neg), 
                               zeros, ones) 
            #import pdb; pdb.set_trace()
            
            # [bs, n, 1]            
            mask = target_mask*mask
                
        loss_ce = F.binary_cross_entropy_with_logits(
            src_logits, target_classes, reduction='none')
        #loss_ce = mask*loss_ce
        loss_ce = loss_ce.sum(dim=1)/mask.sum(dim=1)
        
        losses = {'loss_vline_ce': loss_ce.mean(),}
        return losses
    
    def loss_hline_labels(self, outputs, targets, **kwargs):
        assert 'pred_hline_logits' in outputs
        src_logits = outputs['pred_hline_logits'] # [bs, n, 1]

        target_lines = torch.stack([t['lines'] for t in targets], dim=0) # [bs, n, 3]
        target_mask = torch.stack([t['line_mask'] for t in targets], dim=0) # [bs, n, 1]
        
        target_hvps = torch.stack([t['hvps'] for t in targets], dim=0) # [bs, 2, 3]
        
        with torch.no_grad():
            target_lines = target_lines.unsqueeze(dim=2) # [bs, n, 1, 3]
            target_hvps = target_hvps.unsqueeze(dim=1) # [bs, 1, 2, 3]
            
            cos_sim = F.cosine_similarity(target_lines, target_hvps, dim=-1).abs() # [bs, n, 2]
            cos_sim = cos_sim.min(dim=-1, keepdim=True)[0] # [bs, n, 1]
            
#             import pdb; pdb.set_trace()
            
            ones = torch.ones_like(src_logits)
            zeros = torch.zeros_like(src_logits)
            target_classes = torch.where(cos_sim < self.thresh_line_pos, ones, zeros)
            mask = torch.where(torch.gt(cos_sim, self.thresh_line_pos) &
                               torch.lt(cos_sim, self.thresh_line_neg), 
                               zeros, ones)
            # [bs, n, 1]            
            mask = target_mask*mask
        loss_ce = F.binary_cross_entropy_with_logits(
            src_logits, target_classes, reduction='none')
        loss_ce = mask*loss_ce
        loss_ce = loss_ce.sum(dim=1)/mask.sum(dim=1)
        
        losses = {'loss_hline_ce': loss_ce.mean(),}
        return losses
    
    
    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {            
            'zvp': self.loss_zvp,
            'fovy': self.loss_fovy,
            'hl': self.loss_hl,
            'line_score': self.loss_line_score,
            'line_class': self.loss_line_class,
            # 'vline_labels': self.loss_vline_labels,
            # 'hline_labels': self.loss_hline_labels,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 3
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)

    model = SOFI(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        mixed_query = args.mixed_queries,
        aux_loss=args.aux_loss,
    )
    
    weight_dict = {'loss_zvp': args.zvp_loss_coef, 
    'loss_hl': args.hl_loss_coef,
    'loss_fovy': args.fovy_loss_coef,
    }
    weight_dict['loss_line_score'] = args.line_score_loss_coef
    weight_dict['loss_line_class'] = args.line_class_loss_coef
    # weight_dict['loss_vline_ce'] = 1
    # weight_dict['loss_hline_ce'] = 1
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(6):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['zvp', 'fovy', 'hl', 'line_score', 'line_class']
   
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(weight_dict=weight_dict,
                             losses=losses,
                             line_pos_angle=args.line_pos_angle,
                             line_neg_angle=args.line_neg_angle)
    criterion.to(device)
    
    return model, criterion
