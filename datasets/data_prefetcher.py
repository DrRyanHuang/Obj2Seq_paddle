# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import paddle

def item_cuda(item, device, white_list=[]):
    if isinstance(item, list):
        return [item_cuda(i, device) for i in item]
    elif isinstance(item, dict):
        return {k: item_cuda(v, device) for k, v in item.items() if k not in white_list}
    elif isinstance(item, paddle.Tensor):
        # return item.to(device, non_blocking=True)
        return paddle.to_tensor(item)
    elif isinstance(item, int):
        return item
    else:
        raise TypeError

def to_cuda(samples, targets, device):
    samples = samples.to(device, non_blocking=True)
    targets = item_cuda(targets, device)
    return samples, targets

class data_prefetcher():
    def __init__(self, loader, device, prefetch=False):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        # if prefetch and self.device=='cuda':
        if prefetch:
            self.stream = paddle.device.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_samples, self.next_targets = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with paddle.device.cuda.stream_guard(self.stream):
            self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

    def next(self):
        # if self.prefetch and self.device=='cuda':
        if self.prefetch:
            paddle.device.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            # if samples is not None:
            #     samples.record_stream(paddle.device.cuda.current_stream())
            # if targets is not None:
            #     for t in targets:
            #         for k, v in t.items():
            #             v.record_stream(paddle.device.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples = None
                targets = None
        return samples, targets
