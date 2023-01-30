# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Anchor DETR (https://github.com/megvii-research/AnchorDETR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from functools import partial

import paddle
# from torch.utils.data import DataLoader

import datasets.samplers as samplers
from util.collate_fn import build_collate_fn
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
from paddle.io import BatchSampler

class HybridBatchSampler(BatchSampler):
    def __init__(self, *args, **kwargs):
        self.kps_sampler = kwargs.pop("kps_sampler")
        self.kps_iter = iter(self.kps_sampler)
        super().__init__(*args, **kwargs)
        assert self.batch_size == 1

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            try:
                idx_kps = next(self.kps_iter)
            except StopIteration:
                self.kps_sampler.set_epoch(self.kps_sampler.epoch * 3)
                self.kps_iter = iter(self.kps_sampler)
                idx_kps = next(self.kps_iter)
            batch.append(-idx_kps)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


def build_dataloader(dataset_train, dataset_val, args, fleet=False):
    if args.distributed or fleet:
        if args.cache_mode:
            # sampler_train = samplers.NodeDistributedSampler(dataset_train)
            # sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
            pass
        else:
            sampler_train = paddle.io.DistributedBatchSampler(dataset_train, 
                                                              batch_size=args.batch_size, 
                                                              shuffle=True)
            sampler_val = paddle.io.DistributedBatchSampler(dataset_val, 
                                                            batch_size=args.batch_size, 
                                                            shuffle=False)
            output_sampler = sampler_train
    else:
        sampler_train = paddle.io.RandomSampler(dataset_train)
        sampler_val = paddle.io.SequenceSampler(dataset_val)

    if not fleet:
        if args.type == "coco_hybrid":
            kps_sampler_train = samplers.DistributedSampler(dataset_train.dset_kps)
            batch_sampler_train = HybridBatchSampler(
                sampler_train, args.batch_size, drop_last=True, kps_sampler=kps_sampler_train)
            output_sampler = [sampler_train, kps_sampler_train]
        else:
            batch_sampler_train = paddle.io.BatchSampler(
                sampler=sampler_train, batch_size=args.batch_size, drop_last=True)
            output_sampler = sampler_train

        collate_fn = build_collate_fn(args.COLLECT_FN)
        data_loader_train = paddle.io.DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    collate_fn=collate_fn, num_workers=args.num_workers,
                                    use_shared_memory=True)
        data_loader_val = paddle.io.DataLoader(dataset_val, 
                                               batch_size=args.batch_size, # sampler=sampler_val,
                                               drop_last=False, 
                                               collate_fn=collate_fn, 
                                               num_workers=args.num_workers,
                                               use_shared_memory=True)
        return data_loader_train, data_loader_val, output_sampler
    
    else:
        collate_fn = build_collate_fn(args.COLLECT_FN)
        data_loader_train = paddle.io.DataLoader(dataset_train, batch_sampler=sampler_train,
                                    collate_fn=collate_fn, num_workers=args.num_workers,
                                    use_shared_memory=True)
        data_loader_val = paddle.io.DataLoader(dataset_val, 
                                            #    batch_size=args.batch_size, # sampler=sampler_val,
                                            #    drop_last=False, 
                                               batch_sampler=sampler_val,
                                               collate_fn=collate_fn, 
                                               num_workers=args.num_workers,
                                               use_shared_memory=True)
        return data_loader_train, data_loader_val, output_sampler