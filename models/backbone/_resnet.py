import enum
from enum import Enum, EnumMeta
from dataclasses import dataclass, fields
from functools import partial
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple,
                    Type, TypeVar, Union, cast)

import paddle.nn as nn
from paddle.vision.transforms import functional as F

import paddle
from paddle import Tensor

from ..initializer import constant_, kaiming_normal_
# from ..transforms._presets import ImageClassification
# from ..utils import _log_api_usage_once
# from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
# from ._utils import _ovewrite_named_param, handle_legacy_interface


class InterpolationMode(Enum):
    """Interpolation modes
    Available interpolation methods are ``nearest``, ``bilinear``, ``bicubic``, ``box``, ``hamming``, and ``lanczos``.
    """

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"
    

class ImageClassification(nn.Layer):
    def __init__(
        self,
        *,
        crop_size: int,
        resize_size: int = 256,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.crop_size = [crop_size]
        self.resize_size = [resize_size]
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation

    def forward(self, img: Tensor) -> Tensor:
        img = F.resize(img, self.resize_size, interpolation=self.interpolation)
        img = F.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = F.to_tensor(img)
        img = F.convert_image_dtype(img, paddle.float32)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
            f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are first rescaled to "
            f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and ``std={self.std}``."
        )

V = TypeVar("V")
def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


class Weights:
    """
    This class is used to group important attributes associated with the pre-trained weights.

    Args:
        url (str): The location where we find the weights.
        transforms (Callable): A callable that constructs the preprocessing method (or validation preset transforms)
            needed to use the model. The reason we attach a constructor method rather than an already constructed
            object is because the specific object might have memory and thus we want to delay initialization until
            needed.
        meta (Dict[str, Any]): Stores meta-data related to the weights of the model and its configuration. These can be
            informative attributes (for example the number of parameters/flops, recipe link/methods used in training
            etc), configuration parameters (for example the `num_classes`) needed to construct the model or important
            meta-data (for example the `classes` of a classification model) needed to use the model.
    """

    url: str
    transforms: Callable
    meta: Dict[str, Any]


T = TypeVar("T", bound=Enum)
class StrEnumMeta(EnumMeta):
    auto = enum.auto

    def from_str(self: Type[T], member: str) -> T:  # type: ignore[misc]
        try:
            return self[member]
        except KeyError:
            # TODO: use `add_suggestion` from torchvision.prototype.utils._internal to improve the error message as
            #  soon as it is migrated.
            raise ValueError(f"Unknown value '{member}' for {self.__name__}.") from None


class StrEnum(Enum, metaclass=StrEnumMeta):
    pass


class WeightsEnum(StrEnum):
    """
    This class is the parent class of all model weights. Each model building method receives an optional `weights`
    parameter with its associated pre-trained weights. It inherits from `Enum` and its values should be of type
    `Weights`.

    Args:
        value (Weights): The data class entry with the weight information.
    """

    def __init__(self, value: Weights):
        self._value_ = value

    @classmethod
    def verify(cls, obj: Any) -> Any:
        if obj is not None:
            if type(obj) is str:
                obj = cls.from_str(obj.replace(cls.__name__ + ".", ""))
            elif not isinstance(obj, cls):
                raise TypeError(
                    f"Invalid Weight class provided; expected {cls.__name__} but received {obj.__class__.__name__}."
                )
        return obj

    def get_state_dict(self, progress: bool) -> Mapping[str, Any]:
        return load_state_dict_from_url(self.url, progress=progress)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self._name_}"

    def __getattr__(self, name):
        # Be able to fetch Weights attributes directly
        for f in fields(Weights):
            if f.name == name:
                return object.__getattribute__(self.value, name)
        return super().__getattr__(name)


__all__ = [
    "ResNet",
    "ResNet18_Weights",
    "ResNet34_Weights",
    "ResNet50_Weights",
    "ResNet101_Weights",
    "ResNet152_Weights",
    "ResNeXt50_32X4D_Weights",
    "ResNeXt101_32X8D_Weights",
    "ResNeXt101_64X4D_Weights",
    "Wide_ResNet50_2_Weights",
    "Wide_ResNet101_2_Weights",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2D:
    """3x3 convolution with padding"""
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias_attr=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2D:
    """1x1 convolution"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False)


class BasicBlock(nn.Layer):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Layer] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Layer]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Layer] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Layer]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Layer):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Layer]] = None,
        **kwargs,
    ) -> None:
        super(ResNet, self).__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2D(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias_attr=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for name, m in self.named_sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for name, m in self.named_sublayers():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}


class ResNet18_Weights(WeightsEnum):
    # IMAGENET1K_V1 = Weights(
    #     url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
    #     transforms=partial(ImageClassification, crop_size=224),
    #     meta={
    #         **_COMMON_META,
    #         "num_params": 11689512,
    #         "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
    #         "_metrics": {
    #             "ImageNet-1K": {
    #                 "acc@1": 69.758,
    #                 "acc@5": 89.078,
    #             }
    #         },
    #         "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
    #     },
    # )
    # DEFAULT = IMAGENET1K_V1
    pass


class ResNet34_Weights(WeightsEnum):
    # IMAGENET1K_V1 = Weights(
    #     url="https://download.pytorch.org/models/resnet34-b627a593.pth",
    #     transforms=partial(ImageClassification, crop_size=224),
    #     meta={
    #         **_COMMON_META,
    #         "num_params": 21797672,
    #         "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
    #         "_metrics": {
    #             "ImageNet-1K": {
    #                 "acc@1": 73.314,
    #                 "acc@5": 91.420,
    #             }
    #         },
    #         "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
    #     },
    # )
    # DEFAULT = IMAGENET1K_V1
    pass
    

class ResNet50_Weights(WeightsEnum):
    # IMAGENET1K_V1 = Weights(
    #     url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
    #     transforms=partial(ImageClassification, crop_size=224),
    #     meta={
    #         **_COMMON_META,
    #         "num_params": 25557032,
    #         "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
    #         "_metrics": {
    #             "ImageNet-1K": {
    #                 "acc@1": 76.130,
    #                 "acc@5": 92.862,
    #             }
    #         },
    #         "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
    #     },
    # )
    # IMAGENET1K_V2 = Weights(
    #     url="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
    #     transforms=partial(ImageClassification, crop_size=224, resize_size=232),
    #     meta={
    #         **_COMMON_META,
    #         "num_params": 25557032,
    #         "recipe": "https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621",
    #         "_metrics": {
    #             "ImageNet-1K": {
    #                 "acc@1": 80.858,
    #                 "acc@5": 95.434,
    #             }
    #         },
    #         "_docs": """
    #             These weights improve upon the results of the original paper by using TorchVision's `new training recipe
    #             <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
    #         """,
    #     },
    # )
    # DEFAULT = IMAGENET1K_V2
    pass


class ResNet101_Weights(WeightsEnum):
    # IMAGENET1K_V1 = Weights(
    #     url="https://download.pytorch.org/models/resnet101-63fe2227.pth",
    #     transforms=partial(ImageClassification, crop_size=224),
    #     meta={
    #         **_COMMON_META,
    #         "num_params": 44549160,
    #         "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
    #         "_metrics": {
    #             "ImageNet-1K": {
    #                 "acc@1": 77.374,
    #                 "acc@5": 93.546,
    #             }
    #         },
    #         "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
    #     },
    # )
    # IMAGENET1K_V2 = Weights(
    #     url="https://download.pytorch.org/models/resnet101-cd907fc2.pth",
    #     transforms=partial(ImageClassification, crop_size=224, resize_size=232),
    #     meta={
    #         **_COMMON_META,
    #         "num_params": 44549160,
    #         "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
    #         "_metrics": {
    #             "ImageNet-1K": {
    #                 "acc@1": 81.886,
    #                 "acc@5": 95.780,
    #             }
    #         },
    #         "_docs": """
    #             These weights improve upon the results of the original paper by using TorchVision's `new training recipe
    #             <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
    #         """,
    #     },
    # )
    # DEFAULT = IMAGENET1K_V2
    pass

class ResNet152_Weights(WeightsEnum):
    # IMAGENET1K_V1 = Weights(
    #     url="https://download.pytorch.org/models/resnet152-394f9c45.pth",
    #     transforms=partial(ImageClassification, crop_size=224),
    #     meta={
    #         **_COMMON_META,
    #         "num_params": 60192808,
    #         "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
    #         "_metrics": {
    #             "ImageNet-1K": {
    #                 "acc@1": 78.312,
    #                 "acc@5": 94.046,
    #             }
    #         },
    #         "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
    #     },
    # )
    # IMAGENET1K_V2 = Weights(
    #     url="https://download.pytorch.org/models/resnet152-f82ba261.pth",
    #     transforms=partial(ImageClassification, crop_size=224, resize_size=232),
    #     meta={
    #         **_COMMON_META,
    #         "num_params": 60192808,
    #         "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
    #         "_metrics": {
    #             "ImageNet-1K": {
    #                 "acc@1": 82.284,
    #                 "acc@5": 96.002,
    #             }
    #         },
    #         "_docs": """
    #             These weights improve upon the results of the original paper by using TorchVision's `new training recipe
    #             <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
    #         """,
    #     },
    # )
    # DEFAULT = IMAGENET1K_V2
    pass


class ResNeXt50_32X4D_Weights(WeightsEnum):
    # IMAGENET1K_V1 = Weights(
    #     url="https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    #     transforms=partial(ImageClassification, crop_size=224),
    #     meta={
    #         **_COMMON_META,
    #         "num_params": 25028904,
    #         "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
    #         "_metrics": {
    #             "ImageNet-1K": {
    #                 "acc@1": 77.618,
    #                 "acc@5": 93.698,
    #             }
    #         },
    #         "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
    #     },
    # )
    # IMAGENET1K_V2 = Weights(
    #     url="https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
    #     transforms=partial(ImageClassification, crop_size=224, resize_size=232),
    #     meta={
    #         **_COMMON_META,
    #         "num_params": 25028904,
    #         "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
    #         "_metrics": {
    #             "ImageNet-1K": {
    #                 "acc@1": 81.198,
    #                 "acc@5": 95.340,
    #             }
    #         },
    #         "_docs": """
    #             These weights improve upon the results of the original paper by using TorchVision's `new training recipe
    #             <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
    #         """,
    #     },
    # )
    # DEFAULT = IMAGENET1K_V2
    pass

class ResNeXt101_32X8D_Weights(WeightsEnum):
    # IMAGENET1K_V1 = Weights(
    #     url="https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    #     transforms=partial(ImageClassification, crop_size=224),
    #     meta={
    #         **_COMMON_META,
    #         "num_params": 88791336,
    #         "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
    #         "_metrics": {
    #             "ImageNet-1K": {
    #                 "acc@1": 79.312,
    #                 "acc@5": 94.526,
    #             }
    #         },
    #         "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
    #     },
    # )
    # IMAGENET1K_V2 = Weights(
    #     url="https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth",
    #     transforms=partial(ImageClassification, crop_size=224, resize_size=232),
    #     meta={
    #         **_COMMON_META,
    #         "num_params": 88791336,
    #         "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
    #         "_metrics": {
    #             "ImageNet-1K": {
    #                 "acc@1": 82.834,
    #                 "acc@5": 96.228,
    #             }
    #         },
    #         "_docs": """
    #             These weights improve upon the results of the original paper by using TorchVision's `new training recipe
    #             <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
    #         """,
    #     },
    # )
    # DEFAULT = IMAGENET1K_V2
    pass

class ResNeXt101_64X4D_Weights(WeightsEnum):
    # IMAGENET1K_V1 = Weights(
    #     url="https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth",
    #     transforms=partial(ImageClassification, crop_size=224, resize_size=232),
    #     meta={
    #         **_COMMON_META,
    #         "num_params": 83455272,
    #         "recipe": "https://github.com/pytorch/vision/pull/5935",
    #         "_metrics": {
    #             "ImageNet-1K": {
    #                 "acc@1": 83.246,
    #                 "acc@5": 96.454,
    #             }
    #         },
    #         "_docs": """
    #             These weights were trained from scratch by using TorchVision's `new training recipe
    #             <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
    #         """,
    #     },
    # )
    # DEFAULT = IMAGENET1K_V1
    pass


class Wide_ResNet50_2_Weights(WeightsEnum):
    # IMAGENET1K_V1 = Weights(
    #     url="https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    #     transforms=partial(ImageClassification, crop_size=224),
    #     meta={
    #         **_COMMON_META,
    #         "num_params": 68883240,
    #         "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
    #         "_metrics": {
    #             "ImageNet-1K": {
    #                 "acc@1": 78.468,
    #                 "acc@5": 94.086,
    #             }
    #         },
    #         "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
    #     },
    # )
    # IMAGENET1K_V2 = Weights(
    #     url="https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
    #     transforms=partial(ImageClassification, crop_size=224, resize_size=232),
    #     meta={
    #         **_COMMON_META,
    #         "num_params": 68883240,
    #         "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
    #         "_metrics": {
    #             "ImageNet-1K": {
    #                 "acc@1": 81.602,
    #                 "acc@5": 95.758,
    #             }
    #         },
    #         "_docs": """
    #             These weights improve upon the results of the original paper by using TorchVision's `new training recipe
    #             <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
    #         """,
    #     },
    # )
    # DEFAULT = IMAGENET1K_V2
    pass

class Wide_ResNet101_2_Weights(WeightsEnum):
    # IMAGENET1K_V1 = Weights(
    #     url="https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
    #     transforms=partial(ImageClassification, crop_size=224),
    #     meta={
    #         **_COMMON_META,
    #         "num_params": 126886696,
    #         "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
    #         "_metrics": {
    #             "ImageNet-1K": {
    #                 "acc@1": 78.848,
    #                 "acc@5": 94.284,
    #             }
    #         },
    #         "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
    #     },
    # )
    # IMAGENET1K_V2 = Weights(
    #     url="https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth",
    #     transforms=partial(ImageClassification, crop_size=224, resize_size=232),
    #     meta={
    #         **_COMMON_META,
    #         "num_params": 126886696,
    #         "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
    #         "_metrics": {
    #             "ImageNet-1K": {
    #                 "acc@1": 82.510,
    #                 "acc@5": 96.020,
    #             }
    #         },
    #         "_docs": """
    #             These weights improve upon the results of the original paper by using TorchVision's `new training recipe
    #             <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
    #         """,
    #     },
    # )
    # DEFAULT = IMAGENET1K_V2
    pass

# @register_model()
# @handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


# @register_model()
# @handle_legacy_interface(weights=("pretrained", ResNet34_Weights.IMAGENET1K_V1))
def resnet34(*, weights: Optional[ResNet34_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet34_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet34_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet34_Weights
        :members:
    """
    weights = ResNet34_Weights.verify(weights)

    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)


# @register_model()
# @handle_legacy_interface(weights=("pretrained", ResNet50_Weights.IMAGENET1K_V1))
def resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet50_Weights
        :members:
    """
    # weights = ResNet50_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


# @register_model()
# @handle_legacy_interface(weights=("pretrained", ResNet101_Weights.IMAGENET1K_V1))
def resnet101(*, weights: Optional[ResNet101_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-101 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet101_Weights
        :members:
    """
    weights = ResNet101_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


# @register_model()
# @handle_legacy_interface(weights=("pretrained", ResNet152_Weights.IMAGENET1K_V1))
def resnet152(*, weights: Optional[ResNet152_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-152 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet152_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet152_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet152_Weights
        :members:
    """
    weights = ResNet152_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 8, 36, 3], weights, progress, **kwargs)


# @register_model()
# @handle_legacy_interface(weights=("pretrained", ResNeXt50_32X4D_Weights.IMAGENET1K_V1))
def resnext50_32x4d(
    *, weights: Optional[ResNeXt50_32X4D_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt50_32X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNext50_32X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt50_32X4D_Weights
        :members:
    """
    weights = ResNeXt50_32X4D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


# @register_model()
# @handle_legacy_interface(weights=("pretrained", ResNeXt101_32X8D_Weights.IMAGENET1K_V1))
def resnext101_32x8d(
    *, weights: Optional[ResNeXt101_32X8D_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """ResNeXt-101 32x8d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt101_32X8D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNeXt101_32X8D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt101_32X8D_Weights
        :members:
    """
    weights = ResNeXt101_32X8D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 8)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


# @register_model()
# @handle_legacy_interface(weights=("pretrained", ResNeXt101_64X4D_Weights.IMAGENET1K_V1))
def resnext101_64x4d(
    *, weights: Optional[ResNeXt101_64X4D_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """ResNeXt-101 64x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt101_64X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNeXt101_64X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt101_64X4D_Weights
        :members:
    """
    weights = ResNeXt101_64X4D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 64)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


# @register_model()
# @handle_legacy_interface(weights=("pretrained", Wide_ResNet50_2_Weights.IMAGENET1K_V1))
def wide_resnet50_2(
    *, weights: Optional[Wide_ResNet50_2_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """Wide ResNet-50-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        weights (:class:`~torchvision.models.Wide_ResNet50_2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Wide_ResNet50_2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.Wide_ResNet50_2_Weights
        :members:
    """
    weights = Wide_ResNet50_2_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


# @register_model()
# @handle_legacy_interface(weights=("pretrained", Wide_ResNet101_2_Weights.IMAGENET1K_V1))
def wide_resnet101_2(
    *, weights: Optional[Wide_ResNet101_2_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """Wide ResNet-101-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-101 has 2048-512-2048
    channels, and in Wide ResNet-101-2 has 2048-1024-2048.

    Args:
        weights (:class:`~torchvision.models.Wide_ResNet101_2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Wide_ResNet101_2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.Wide_ResNet101_2_Weights
        :members:
    """
    weights = Wide_ResNet101_2_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


# The dictionary below is internal implementation detail and will be removed in v0.15
# from ._utils import _ModelURLs

# model_urls = _ModelURLs(
#     {
#         "resnet18": ResNet18_Weights.IMAGENET1K_V1.url,
#         "resnet34": ResNet34_Weights.IMAGENET1K_V1.url,
#         "resnet50": ResNet50_Weights.IMAGENET1K_V1.url,
#         "resnet101": ResNet101_Weights.IMAGENET1K_V1.url,
#         "resnet152": ResNet152_Weights.IMAGENET1K_V1.url,
#         "resnext50_32x4d": ResNeXt50_32X4D_Weights.IMAGENET1K_V1.url,
#         "resnext101_32x8d": ResNeXt101_32X8D_Weights.IMAGENET1K_V1.url,
#         "wide_resnet50_2": Wide_ResNet50_2_Weights.IMAGENET1K_V1.url,
#         "wide_resnet101_2": Wide_ResNet101_2_Weights.IMAGENET1K_V1.url,
#     }
# )