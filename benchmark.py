# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Benchmark inference speed of Deformable DETR.
"""
import os
import time
import argparse

import torch
from torch.utils.data import DataLoader, DistributedSampler

from args import get_args_parser
from models import build_model, build_ctrlc, build_ctrlcplus

from datasets import build_gsv_dataset, build_hlw_dataset, build_holicity_dataset
from util.misc import nested_tensor_from_tensor_list
import util.misc as utils


def to_device(data, device):
    if type(data) == dict:
        return {k: v.to(device) for k, v in data.items()}
    return [{k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()} for t in data]

@torch.no_grad()
def measure_average_inference_time(model, data_loader, device, num_iters=100, warm_iters=5):
    ts = []
    for iter_, (samples, extra_samples,_) in enumerate(data_loader):
        samples = samples.to(device)
        extra_samples = to_device(extra_samples, device)
        torch.cuda.synchronize()
        t_ = time.perf_counter()
        model(samples, extra_samples)
        torch.cuda.synchronize()
        t = time.perf_counter() - t_
        if iter_ >= warm_iters:
          ts.append(t)
    return sum(ts) / len(ts)


def benchmark(args):
    assert args.warm_iters < args.num_iters and args.num_iters > 0 and args.warm_iters >= 0
    assert args.batch_size > 0
    # assert args.resume is None or os.path.exists(args.resume)

    device = torch.device(args.device)

    # Model
    assert args.model in ['sofi', 'ctrlc', 'mscc']
    if args.model == 'sofi':
        model, criterion = build_model(args)
    else:
        model, criterion = build_ctrlc(args)
    model.to(device)
    model.eval()

    # Dataset
    if args.dataset == 'gsv':
        build_dataset = build_gsv_dataset
    elif args.dataset == 'hlw':
        build_dataset = build_hlw_dataset
    elif args.dataset == 'holicity':
        build_dataset = build_holicity_dataset
    # TODO
    else:
        print('Unrecognized dataset: {}'.format(args.dataset))
        exit()

    dataset = build_dataset(image_set='test', args=args)
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = DataLoader(dataset, args.batch_size, sampler=sampler,
                             drop_last=False, 
                             collate_fn=utils.collate_fn, 
                             num_workers=2)


    # if args.resume is not None:
    #     ckpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
    #     model.load_state_dict(ckpt['model'])
    t = measure_average_inference_time(model, data_loader, device, args.num_iters, args.warm_iters)
    return 1.0 * args.batch_size / t 


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FPS evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.model == 'sofi':
        print(f'Results of {args.model}: num_feature_levels {args.num_feature_levels}, dec_n_points {args.dec_n_points}, enc_n_points {args.enc_n_points}, batch_size {args.batch_size}, backbone:{args.backbone}')
    else:
        print(f'Results of {args.model}, batch_size {args.batch_size}')
    fps = benchmark(args)
    print(f'Inference Speed: {fps:.1f} FPS')

