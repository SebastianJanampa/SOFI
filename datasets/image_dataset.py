import os
import os.path as osp

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import functional as F

import numpy as np
import numpy.linalg as LA
import cv2
import json
import csv
import matplotlib.pyplot as plt
from pylsd import lsd

import datasets.transforms as T

def center_crop(img):
    sz = img.shape[0:2]
    side_length = np.min(sz)
    if sz[0] > sz[1]:
        ul_x = 0  
        ul_y = int(np.floor((sz[0]/2) - (side_length/2)))
        x_inds = [ul_x, sz[1]-1]
        y_inds = [ul_y, ul_y + side_length - 1]
    else:
        ul_x = int(np.floor((sz[1]/2) - (side_length/2)))
        ul_y = 0
        x_inds = [ul_x, ul_x + side_length - 1]
        y_inds = [ul_y, sz[0]-1]

    c_img = img[y_inds[0]:y_inds[1]+1, x_inds[0]:x_inds[1]+1, :]

    return c_img

def create_masks(image):
    masks = torch.zeros((1, height, width), dtype=torch.uint8)
    return masks

def filter_length(segs, min_line_length=10):
    lengths = LA.norm(segs[:,2:4] - segs[:,:2], axis=1)
    segs = segs[lengths > min_line_length]
    return segs[:,:4]

def normalize_segs(segs, pp, rho):    
    pp = np.array([pp[0], pp[1], pp[0], pp[1]], dtype=np.float32)    
    return rho*(segs - pp)    

def normalize_safe_np(v, axis=-1, eps=1e-6):
    de = LA.norm(v, axis=axis, keepdims=True)
    de = np.maximum(de, eps)
    return v/de

def segs2lines_np(segs):
    ones = np.ones(len(segs))
    ones = np.expand_dims(ones, axis=-1)
    p1 = np.concatenate([segs[:,:2], ones], axis=-1)
    p2 = np.concatenate([segs[:,2:], ones], axis=-1)
    lines = np.cross(p1, p2)
    return normalize_safe_np(lines)

def sample_segs_np(segs, num_sample, use_prob=True):    
    num_segs = len(segs)
    sampled_segs = np.zeros([num_sample, 4], dtype=np.float32)
    mask = np.zeros([num_sample, 1], dtype=np.float32)
    if num_sample > num_segs:
        sampled_segs[:num_segs] = segs
        mask[:num_segs] = np.ones([num_segs, 1], dtype=np.float32)
    else:    
        lengths = LA.norm(segs[:,2:] - segs[:,:2], axis=-1)
        prob = lengths/np.sum(lengths)        
        idxs = np.random.choice(segs.shape[0], num_sample, replace=True, p=prob)
        sampled_segs = segs[idxs]
        mask = np.ones([num_sample, 1], dtype=np.float32)
    return sampled_segs, mask

def sample_vert_segs_np(segs, thresh_theta=22.5):    
    lines = segs2lines_np(segs)
    (a,b) = lines[:,0],lines[:,1]
    theta = np.arctan2(np.abs(b),np.abs(a))
    thresh_theta = np.radians(thresh_theta)
    return segs[theta < thresh_theta]

class ImageDataset(Dataset):
    def __init__(self, args, image_path, return_masks=False, transform=None):
        self.input_width = args.input_width
        self.input_height = args.input_height
        self.min_line_length = args.min_line_length
        self.num_input_lines = args.num_input_lines
        self.num_input_vert_lines = args.num_input_vert_lines
        self.vert_line_angle = args.vert_line_angle
        self.return_vert_lines = args.return_vert_lines
        self.return_masks = return_masks
        self.transform = transform
        
        self.list_filename = [image_path,]
        
    def __getitem__(self, idx):
        target = {}
        extra = {}
        
        filename = self.list_filename[idx]
                
        image = cv2.imread(filename)
        assert image is not None, print(filename)
        image = image[:,:,::-1] # convert to rgb
        
        org_image = image
        org_h, org_w = image.shape[0], image.shape[1]
        org_sz = np.array([org_h, org_w]) 
        
        crop_image = center_crop(org_image)
        crop_h, crop_w = crop_image.shape[0], crop_image.shape[1]
        crop_sz = np.array([crop_h, crop_w]) 
                
        image = cv2.resize(image, dsize=(self.input_width, self.input_height))
        input_sz = np.array([self.input_height, self.input_width])
        
        # preprocess
        ratio_x = float(self.input_width)/float(org_w)
        ratio_y = float(self.input_height)/float(org_h)

        pp = (org_w/2, org_h/2)
        rho = 2.0/np.minimum(org_w,org_h)
        
        # detect line and preprocess       
        gray = cv2.cvtColor(org_image, cv2.COLOR_BGR2GRAY)
        org_segs = lsd(gray, scale=0.5)
        org_segs = filter_length(org_segs, self.min_line_length)
        num_segs = len(org_segs)
        assert len(org_segs) > 10, print(len(org_segs))
        
        segs = normalize_segs(org_segs, pp=pp, rho=rho)
        
        # whole segs
        sampled_segs, line_mask = sample_segs_np(
            segs, self.num_input_lines)
        sampled_lines = segs2lines_np(sampled_segs)
        
        # vertical directional segs
        vert_segs = sample_vert_segs_np(segs, thresh_theta=self.vert_line_angle)
        if len(vert_segs) < 2:
            vert_segs = segs
            
        sampled_vert_segs, vert_line_mask = sample_segs_np(
            vert_segs, self.num_input_vert_lines)
        sampled_vert_lines = segs2lines_np(sampled_vert_segs)
        
        if self.return_masks:
            masks = create_masks(image)
            
        image = np.ascontiguousarray(image)
        
        if self.return_vert_lines:
            target['segs'] = torch.from_numpy(np.ascontiguousarray(sampled_vert_segs)).contiguous().float()
            target['lines'] = torch.from_numpy(np.ascontiguousarray(sampled_vert_lines)).contiguous().float()
            target['line_mask'] = torch.from_numpy(np.ascontiguousarray(vert_line_mask)).contiguous().float()
        else:
            target['segs'] = torch.from_numpy(np.ascontiguousarray(sampled_segs)).contiguous().float()
            target['lines'] = torch.from_numpy(np.ascontiguousarray(sampled_lines)).contiguous().float()
            target['line_mask'] = torch.from_numpy(np.ascontiguousarray(line_mask)).contiguous().float()
                    
        if self.return_masks:
            target['masks'] = masks
        target['org_img'] = org_image
        target['org_sz'] = org_sz
        target['crop_sz'] = crop_sz
        target['input_sz'] = input_sz
        target['img_path'] = filename
        target['filename'] = filename
        
        extra['lines'] = target['lines'].clone()
        extra['line_mask'] = target['line_mask'].clone()

        return self.transform(image, extra, target)
    
    def __len__(self):
        return len(self.list_filename)   

def make_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) 

def build_image(image_path, args):
    dataset = ImageDataset(args, image_path, return_masks=args.masks, transform=make_transform())
    return dataset

