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

def extract_hl(left, right, width):
    hl_homo = np.cross(np.append(left, 1), np.append(right, 1))
    hl_left_homo = np.cross(hl_homo, [-1, 0, -width/2]);
    hl_left = hl_left_homo[0:2]/hl_left_homo[-1];
    hl_right_homo = np.cross(hl_homo, [-1, 0, width/2]);
    hl_right = hl_right_homo[0:2]/hl_right_homo[-1];
    return hl_left, hl_right

def compute_horizon(hl, crop_sz, org_sz, eps=1e-6):
    a,b,c = hl
    if b < 0:
        a, b, c = -hl
    b = np.maximum(b, eps)    
    left = (a - c)/b
    right = (-a - c)/b

    c_left = left*(crop_sz[0]/2)
    c_right = right*(crop_sz[0]/2)

    left_tmp = np.asarray([-crop_sz[1]/2, c_left])
    right_tmp = np.asarray([crop_sz[1]/2, c_right])
    left, right = extract_hl(left_tmp, right_tmp, org_sz[1])

    return [np.squeeze(left), np.squeeze(right)]

def compute_horizon_hlw(hl, sz, eps=1e-6):
    (a,b,c) = hl
    if b < 0:
        a, b, c = -a, -b, -c
    b = np.maximum(b, eps)

    scale = sz[1]/2
    left = np.array([-scale, (scale*a - c)/b])        
    right = np.array([scale, (-scale*a - c)/b])

    return [np.squeeze(left), np.squeeze(right)]

def normalize_safe_np(v, eps=1e-7):
    return v/np.maximum(LA.norm(v), eps)

def compute_horizon_error(pred_hl, target_hl, img_sz, crop_sz):
    target_hl_pts = compute_hl_np(target_hl, img_sz)
    pred_hl_pts = compute_hl_np(pred_hl, img_sz)
    err_hl = np.maximum(np.abs(target_hl_pts[0][1] - pred_hl_pts[0][1]),
                        np.abs(target_hl_pts[1][1] - pred_hl_pts[1][1]))
    err_hl /= img_sz[0] # height
    return err_hl

def to_device(data, device):
    if type(data) == dict:
        return {k: v.to(device) for k, v in data.items()}
    return [{k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()} for t in data]

def test_hlw(model, data_loader_test, device, args):
    csvpath = osp.join(args.output_dir,'{}.csv'.format(args.append_word))
    print('Writing the evaluation results of the {} dataset into {}'.format(args.append_word, csvpath))
    with open(csvpath, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['filename', 'hl-err', 'num_segs'])
        
    model.eval()
    for i, (samples, extra_samples, targets) in enumerate(tqdm(data_loader_test)):
        with torch.no_grad():
            samples = samples.to(device)
            extra_samples = to_device(extra_samples, device)
            outputs = model(samples, extra_samples)
        
            pred_zvp = outputs['pred_zvp'].to('cpu')[0].numpy()
            pred_fovy = outputs['pred_fovy'].to('cpu')[0].numpy()
            pred_hl = outputs['pred_hl'].to('cpu')[0].numpy()
                
            img_sz = targets[0]['org_sz']
            crop_sz = targets[0]['crop_sz']
            num_segs = targets[0]['num_segs']
            filename = targets[0]['filename']
            filename = osp.splitext(filename)[0]
                
            target_hl = targets[0]['hl'].numpy()
            
            # horizon line error
            target_hl_pts = compute_horizon_hlw(target_hl, img_sz)
            pred_hl_pts = compute_horizon(pred_hl, crop_sz, img_sz)
            
            err_hl = np.maximum(np.abs(target_hl_pts[0][1] - pred_hl_pts[0][1]),
                        np.abs(target_hl_pts[1][1] - pred_hl_pts[1][1]))
            err_hl /= img_sz[0] # height
                
            with open(csvpath, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([filename, err_hl, num_segs])
            