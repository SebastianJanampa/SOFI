import os
import os.path as osp
import argparse
from datetime import date
import json
import random
import time
from pathlib import Path
import numpy as np
import numpy.linalg as LA
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import csv

import torch
import torch.nn.functional as F

cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

def c(x):
    return sm.to_rgba(x)

def compute_vp_err(vp1, vp2, dim=-1):
    cos_sim = F.cosine_similarity(vp1, vp2, dim=dim).abs()
    cos_sim = np.clip(cos_sim.item(), 0.0, 1.0)    
    return np.degrees(np.arccos(cos_sim))

def compute_hl_np(hl, sz, eps=1e-6):
    (a,b,c) = hl
    if b < 0:
        a, b, c = -a, -b, -c
    b = np.maximum(b, eps)
    
    left = np.array([-1.0, (a - c)/b])        
    right = np.array([1.0, (-a - c)/b])

    # scale back to original image    
    scale = sz[1]/2
    left = scale*left
    right = scale*right
    return [np.squeeze(left), np.squeeze(right)]

def compute_up_vector(zvp, fovy, eps=1e-7):
    # image size 2 (-1~1)
    focal = 1.0/np.tan(fovy/2.0)
    
    if zvp[2] < 0:
        zvp = -zvp
    zvp = zvp / np.maximum(zvp[2], eps)
    zvp[2] = focal
    return normalize_safe_np(zvp)

def decompose_up_vector(v):
    pitch = np.arcsin(v[2])
    roll = np.arctan(-v[0]/v[1])
    return pitch, roll

def cosine_similarity(v1, v2, eps=1e-7):
    v1 = v1 / np.maximum(LA.norm(v1), eps)
    v2 = v2 / np.maximum(LA.norm(v2), eps)
    return np.sum(v1*v2)

def normalize_safe_np(v, eps=1e-7):
    return v/np.maximum(LA.norm(v), eps)

def compute_up_vector_error(pred_zvp, pred_fovy, target_up_vector):
    pred_up_vector = compute_up_vector(pred_zvp, pred_fovy)
    cos_sim = cosine_similarity(target_up_vector, pred_up_vector)

    target_pitch, target_roll = decompose_up_vector(target_up_vector)

    if cos_sim < 0:
        pred_pitch, pred_roll = decompose_up_vector(-pred_up_vector)
    else:
        pred_pitch, pred_roll = decompose_up_vector(pred_up_vector)

    err_angle = np.degrees(np.arccos(np.clip(np.abs(cos_sim),0.0, 1.0)))
    err_pitch = np.degrees(np.abs(pred_pitch - target_pitch))
    err_roll = np.degrees(np.abs(pred_roll - target_roll))
    return err_angle, err_pitch, err_roll

def compute_fovy_error(pred_fovy, target_fovy):
    pred_fovy = np.degrees(pred_fovy)
    target_fovy = np.degrees(target_fovy)
    err_fovy = np.abs(pred_fovy - target_fovy)
    return err_fovy.item()

def compute_horizon_error(pred_hl, target_hl, img_sz):
    target_hl_pts = compute_hl_np(target_hl, img_sz)
    pred_hl_pts = compute_hl_np(pred_hl, img_sz)
    err_hl = np.maximum(np.abs(target_hl_pts[0][1] - pred_hl_pts[0][1]),
                        np.abs(target_hl_pts[1][1] - pred_hl_pts[1][1]))
    err_hl /= img_sz[0] # height
    return err_hl

def compute_vline_accuracy(lines, num_lines, zvp, logits, cfg):
    thresh_line_pos = np.cos(np.radians(cfg.LOSS.LINE_POS_ANGLE), dtype=np.float32) # near 0.0
    thresh_line_neg = np.cos(np.radians(cfg.LOSS.LINE_NEG_ANGLE), dtype=np.float32) # near 0.0
    
    #import pdb; pdb.set_trace()
    
    pred_classes = (logits > 0.5).squeeze()
    
    cos_sim = F.cosine_similarity(lines, zvp, dim=-1).abs() # [n]
       
    ones = torch.ones_like(pred_classes)
    zeros = torch.zeros_like(pred_classes)
    
    target_classes = torch.where(cos_sim < thresh_line_pos, ones, zeros)
    mask = torch.where(torch.gt(cos_sim, thresh_line_pos) &
                               torch.lt(cos_sim, thresh_line_neg), 
                               zeros, ones)
    
    corrects = []
    for i in range(num_lines):
        if mask[i] > 0:
            corrects.append((pred_classes[i].item() == target_classes[i].item()))
            
#     import pdb; pdb.set_trace()
    return corrects

def compute_hline_accuracy(lines, num_lines, hvps, logits, cfg):
    thresh_line_pos = np.cos(np.radians(cfg.LOSS.LINE_POS_ANGLE), dtype=np.float32) # near 0.0
    thresh_line_neg = np.cos(np.radians(cfg.LOSS.LINE_NEG_ANGLE), dtype=np.float32) # near 0.0
    
    #import pdb; pdb.set_trace()
    
    pred_classes = (logits > 0.5).squeeze()
    
    lines = lines.unsqueeze(1)
    hvps = hvps.unsqueeze(0)
    cos_sim = F.cosine_similarity(lines, hvps, dim=-1).abs() # [n,2]
    cos_sim = cos_sim.min(dim=-1)[0]
       
    ones = torch.ones_like(pred_classes)
    zeros = torch.zeros_like(pred_classes)
    
    target_classes = torch.where(cos_sim < thresh_line_pos, ones, zeros)
    mask = torch.where(torch.gt(cos_sim, thresh_line_pos) &
                               torch.lt(cos_sim, thresh_line_neg), 
                               zeros, ones)
    
    corrects = []
    for i in range(num_lines):
        if mask[i] > 0:
            corrects.append((pred_classes[i].item() == target_classes[i].item()))
    return corrects

def to_device(data, device):
    if type(data) == dict:
        return {k: v.to(device) for k, v in data.items()}
    return [{k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()} for t in data]

def test(args):
    csvpath = osp.join(args.output_dir,'{}.csv'.format(name))
    print('Writing the evaluation results of the {} dataset into {}'.format(name, csvpath))
    with open(csvpath, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['filename',                            
                            'angle-err', 'pitch-err', 'roll-err', 'fovy-err', 'hl-err',
                            'vc-acc', 'hc-acc'])
    
    correct_vl_list = []
    correct_hl_list = []
    
    for i, (samples, extra_samples, targets) in enumerate(tqdm(data_loader_test)):
        with torch.no_grad():
            samples = samples.to(device)
            extra_samples = to_device(extra_samples, device)
            outputs, extra_info = model(samples, extra_samples)
            targets = to_device(targets, device)            
        
            pred_zvp = outputs['pred_zvp'].to('cpu')[0].numpy()
            pred_fovy = outputs['pred_fovy'].to('cpu')[0].numpy()
            pred_hl = outputs['pred_hl'].to('cpu')[0].numpy()
            
            pred_vweight = outputs['pred_vline_logits'][0].sigmoid()
            pred_hweight = outputs['pred_hline_logits'][0].sigmoid()
            
            img_sz = targets[0]['org_sz']
            filename = targets[0]['filename']
            filename = osp.splitext(filename)[0]
                
            target_up_vector = targets[0]['up_vector'].cpu().numpy()
            target_zvp = targets[0]['zvp'].cpu().numpy()
            target_fovy = targets[0]['fovy'].cpu().numpy()
            target_hl = targets[0]['hl'].cpu().numpy()
                
            segs = targets[0]['segs'].cpu().numpy()
            line_mask = targets[0]['line_mask'].cpu().numpy()
            line_mask = np.squeeze(line_mask, axis=1)
            
            num_segs = int(np.sum(line_mask))
            
            target_zvp = normalize_safe_np(target_zvp)                
            
            pred_fovy = np.squeeze(pred_fovy)
            target_fovy = np.squeeze(target_fovy)
            
            # up vector error
            err_angle, err_pitch, err_roll = (
                compute_up_vector_error(pred_zvp, pred_fovy, target_up_vector))
            
            # fovy error
            err_fovy = compute_fovy_error(pred_fovy, target_fovy)
            
            # horizon line error
            err_hl = compute_horizon_error(pred_hl, target_hl, img_sz)
            
            pred_hl_pts = compute_hl_np(pred_hl, img_sz)
            
            
            correct_vl = compute_vline_accuracy(
                    targets[0]['lines'], num_segs, targets[0]['zvp'], pred_vweight, cfg)
            correct_hl = compute_hline_accuracy(
                    targets[0]['lines'], num_segs, targets[0]['hvps'], pred_hweight, cfg)
            acc_vl = np.sum(correct_vl) / len(correct_vl)
            acc_hl = np.sum(correct_hl) / len(correct_hl)
            
            correct_vl_list += correct_vl
            correct_hl_list += correct_hl
            
            with open(csvpath, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([filename,                                    
                                    err_angle, err_pitch, err_roll, err_fovy, err_hl,
                                    acc_vl, acc_hl])
                
    print('accuracy vl', np.mean(correct_vl_list)*100.0)
    print('accuracy hl', np.mean(correct_hl_list)*100.0)
