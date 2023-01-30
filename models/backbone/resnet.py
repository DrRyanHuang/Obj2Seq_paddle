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
"""
Backbone modules.
"""
from collections import OrderedDict

import paddle
import paddle.nn.functional as F
from paddle import ParamAttr
# import torch
# import torch.nn.functional as F
# import torchvision
from paddle import nn
# from paddle.vision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from . import _resnet

from util.misc import NestedTensor, is_main_process


class IntermediateLayerGetter(nn.LayerDict):
    """
    Layer wrapper that returns intermediate layers from a model.
    It has a strong assumption that the layers have been registered into the model in the
    same order as they are used. This means that one should **not** reuse the same nn.Layer
    twice in the forward if you want this to work.
    Additionally, it is only able to query sublayer that are directly assigned to the model.
    So if `model` is passed, `model.feature1` can be returned, but not `model.feature1.layer2`.
    Args:
        model (nn.Layer): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names of the layers for
        which the activations will be returned as the key of the dict, and the value of the
        dict is the name of the returned activation (which the user can specify).
    Examples:
        .. code-block:: python
        import paddle
        m = paddle.vision.models.resnet18(pretrained=False)
        # extract layer1 and layer3, giving as names `feat1` and feat2`
        new_m = paddle.vision.models._utils.IntermediateLayerGetter(m,
            {'layer1': 'feat1', 'layer3': 'feat2'})
        out = new_m(paddle.rand([1, 3, 224, 224]))
        print([(k, v.shape) for k, v in out.items()])
        # [('feat1', [1, 64, 56, 56]), ('feat2', [1, 256, 14, 14])]
    """

    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Layer, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():

            if (isinstance(module, nn.Linear) and x.ndim == 4) or (
                len(module.sublayers()) > 0
                and isinstance(module.sublayers()[0], nn.Linear)
                and x.ndim == 4
            ):
                x = paddle.flatten(x, 1)

            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


# class FrozenBatchNorm2D(nn.Layer):
#     """
#     BatchNorm2d where the batch statistics and the affine parameters are fixed.

#     Copy-paste from torchvision.misc.ops with added eps before rqsrt,
#     without which any other models than torchvision.models.resnet[18,34,50,101]
#     produce nans.
#     """

#     def __init__(self, n, eps=1e-5):
#         super(FrozenBatchNorm2D, self).__init__()
#         self.register_buffer("weight", paddle.ones([n]))
#         self.register_buffer("bias", paddle.zeros([n]))
#         self.register_buffer("running_mean", paddle.zeros([n]))
#         self.register_buffer("running_var", paddle.ones([n]))
#         self.eps = eps

#     def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
#                               missing_keys, unexpected_keys, error_msgs):
#         num_batches_tracked_key = prefix + 'num_batches_tracked'
#         if num_batches_tracked_key in state_dict:
#             del state_dict[num_batches_tracked_key]

#         super(FrozenBatchNorm2D, self)._load_from_state_dict(
#             state_dict, prefix, local_metadata, strict,
#             missing_keys, unexpected_keys, error_msgs)

#     def forward(self, x):
#         # move reshapes to the beginning
#         # to make it fuser-friendly
#         w = self.weight.reshape([1, -1, 1, 1])
#         b = self.bias.reshape([1, -1, 1, 1])
#         rv = self.running_var.reshape([1, -1, 1, 1])
#         rm = self.running_mean.reshape([1, -1, 1, 1])
#         eps = self.eps
#         scale = w * (rv + eps).rsqrt()
#         bias = b - rm * scale
#         return x * scale + bias

class FrozenBatchNorm2D(nn.BatchNorm2D):
    def __init__(self, num_channels):
        weight_attr = ParamAttr(learning_rate=0.0, trainable=False)
        bias_attr = ParamAttr(learning_rate=0.0, trainable=False)
        super(FrozenBatchNorm2D, self).__init__(
            num_channels, weight_attr=weight_attr, bias_attr=bias_attr, use_global_stats=True
        )
        

class BackboneBase(nn.Layer):

    def __init__(self, backbone: nn.Layer, train_backbone: bool, return_interm_layers: bool):
        super(BackboneBase, self).__init__()
        # for name, parameter in backbone.named_parameters():
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.stop_gradient = True
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out = []
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].cast("float32"), size=x.shape[-2:])[0]
            out.append(NestedTensor(x, mask))
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, args):
        name = args.name
        pretrained = is_main_process() and args.pretrained
        train_backbone = args.train_backbone
        return_interm_layers = args.num_feature_levels > 1
        dilation = args.dilation
        norm_layer = NORM_DICT[args.norm_layer]

        backbone = getattr(_resnet, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained, norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


NORM_DICT = {
    "FrozenBN": FrozenBatchNorm2D,
    "BN": nn.BatchNorm2D,
}


def build_backbone(args):
    # include
    ## args.name
    ## args.dilation
    ## args.train_backbone
    ## args.num_feature_levels
    backbone = Backbone(args)
    return backbone
