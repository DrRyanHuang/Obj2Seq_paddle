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
Transforms and data augmentation for both image + bbox.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import random
import numpy as np
import uuid

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import PIL
from PIL import Image
import paddle
import paddle.vision.transforms as T
from paddle.vision.transforms import functional as F


from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return isinstance(img, paddle.Tensor)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def _get_image_size(img):
    if _is_pil_image(img):
        return img.size
    elif _is_numpy_image(img):
        return img.shape[:2][::-1]
    elif _is_tensor_image(img):
        if len(img.shape) == 3:
            return img.shape[1:][::-1]  # chw -> wh
        elif len(img.shape) == 4:
            return img.shape[2:][::-1]  # nchw -> wh
        else:
            raise ValueError(
                "The dim for input Tensor should be 3-D or 4-D, but received {}".format(
                    len(img.shape)
                )
            )
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


def crop(image, target_old, region):
    cropped_image = F.crop(image, *region)

    target = target_old.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = np.array([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = np.array([w, h], dtype=np.float32)
        cropped_boxes = boxes - np.array([j, i, j, i])
        cropped_boxes = np.minimum(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clip(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(axis=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        
        # Fix No Object BUG
        # If Not, it throw fucking stupid `Segmentation fault` when distributed training with fleet
        if not (area > 1).any(): # 如果裁剪出的图片没有目标，直接返回原图吧!
            return image, target_old
        
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    if "keypoints" in target:
        keypoints = target['keypoints']
        keypoints = keypoints - np.array([j, i, 0.])
        target['keypoints'] = keypoints
        fields.append("keypoints")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            # keep = np.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], axis=1)
            keep = area > 1
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]
        # print(keep)

    return cropped_image, target


# class BaseOperator(object):
#     def __init__(self, name=None):
#         if name is None:
#             name = self.__class__.__name__
#         self._id = name + '_' + str(uuid.uuid4())[-6:]

#     def apply(self, sample, context=None):
#         """ Process a sample.
#         Args:
#             sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
#             context (dict): info about this sample processing
#         Returns:
#             result (dict): a processed sample
#         """
#         return sample

#     def __call__(self, sample, context=None):
#         """ Process a sample.
#         Args:
#             sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
#             context (dict): info about this sample processing
#         Returns:
#             result (dict): a processed sample
#         """
#         if isinstance(sample, Sequence):
#             for i in range(len(sample)):
#                 sample[i] = self.apply(sample[i], context)
#         else:
#             sample = self.apply(sample, context)
#         return sample

#     def __str__(self):
#         return str(self._id)


# class RandomSizeCrop(BaseOperator):
#     """
#     Cut the image randomly according to `min_size` and `max_size`
#     """

#     def __init__(self, min_size, max_size):
#         super(RandomSizeCrop, self).__init__()
#         self.min_size = min_size
#         self.max_size = max_size

#         from paddle.vision.transforms.functional import crop as paddle_crop
#         self.paddle_crop = paddle_crop

#     @staticmethod
#     def get_crop_params(img_shape, output_size):
#         """Get parameters for ``crop`` for a random crop.
#         Args:
#             img_shape (list|tuple): Image's height and width.
#             output_size (list|tuple): Expected output size of the crop.
#         Returns:
#             tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
#         """
#         h, w = img_shape
#         th, tw = output_size

#         if h + 1 < th or w + 1 < tw:
#             raise ValueError(
#                 "Required crop size {} is larger then input image size {}".
#                 format((th, tw), (h, w)))

#         if w == tw and h == th:
#             return 0, 0, h, w

#         i = random.randint(0, h - th + 1)
#         j = random.randint(0, w - tw + 1)
#         return i, j, th, tw

#     def crop(self, sample, region):
#         image_shape = sample['image'].shape[:2]
#         sample['image'] = self.paddle_crop(sample['image'], *region)

#         keep_index = None
#         # apply bbox
#         if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
#             sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], region)
#             bbox = sample['gt_bbox'].reshape([-1, 2, 2])
#             area = (bbox[:, 1, :] - bbox[:, 0, :]).prod(axis=1)
#             keep_index = np.where(area > 0)[0]
#             sample['gt_bbox'] = sample['gt_bbox'][keep_index] if len(
#                 keep_index) > 0 else np.zeros(
#                     [0, 4], dtype=np.float32)
#             sample['gt_class'] = sample['gt_class'][keep_index] if len(
#                 keep_index) > 0 else np.zeros(
#                     [0, 1], dtype=np.float32)
#             if 'gt_score' in sample:
#                 sample['gt_score'] = sample['gt_score'][keep_index] if len(
#                     keep_index) > 0 else np.zeros(
#                         [0, 1], dtype=np.float32)
#             if 'is_crowd' in sample:
#                 sample['is_crowd'] = sample['is_crowd'][keep_index] if len(
#                     keep_index) > 0 else np.zeros(
#                         [0, 1], dtype=np.float32)

#         # # apply polygon
#         # if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
#         #     sample['gt_poly'] = self.apply_segm(sample['gt_poly'], region,
#         #                                         image_shape)
#         #     if keep_index is not None:
#         #         sample['gt_poly'] = sample['gt_poly'][keep_index]
#         # # apply gt_segm
#         # if 'gt_segm' in sample and len(sample['gt_segm']) > 0:
#         #     i, j, h, w = region
#         #     sample['gt_segm'] = sample['gt_segm'][:, i:i + h, j:j + w]
#         #     if keep_index is not None:
#         #         sample['gt_segm'] = sample['gt_segm'][keep_index]

#         return sample

#     def apply_bbox(self, bbox, region):
#         i, j, h, w = region
#         region_size = np.asarray([w, h])
#         crop_bbox = bbox - np.asarray([j, i, j, i])
#         crop_bbox = np.minimum(crop_bbox.reshape([-1, 2, 2]), region_size)
#         crop_bbox = crop_bbox.clip(min=0)
#         return crop_bbox.reshape([-1, 4]).astype('float32')

#     # ------- No need to deal with segm -------
#     # def apply_segm(self, segms, region, image_shape):
#     #     def _crop_poly(segm, crop):
#     #         xmin, ymin, xmax, ymax = crop
#     #         crop_coord = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
#     #         crop_p = np.array(crop_coord).reshape(4, 2)
#     #         crop_p = Polygon(crop_p)

#     #         crop_segm = list()
#     #         for poly in segm:
#     #             poly = np.array(poly).reshape(len(poly) // 2, 2)
#     #             polygon = Polygon(poly)
#     #             if not polygon.is_valid:
#     #                 exterior = polygon.exterior
#     #                 multi_lines = exterior.intersection(exterior)
#     #                 polygons = shapely.ops.polygonize(multi_lines)
#     #                 polygon = MultiPolygon(polygons)
#     #             multi_polygon = list()
#     #             if isinstance(polygon, MultiPolygon):
#     #                 multi_polygon = copy.deepcopy(polygon)
#     #             else:
#     #                 multi_polygon.append(copy.deepcopy(polygon))
#     #             for per_polygon in multi_polygon:
#     #                 inter = per_polygon.intersection(crop_p)
#     #                 if not inter:
#     #                     continue
#     #                 if isinstance(inter, (MultiPolygon, GeometryCollection)):
#     #                     for part in inter:
#     #                         if not isinstance(part, Polygon):
#     #                             continue
#     #                         part = np.squeeze(
#     #                             np.array(part.exterior.coords[:-1]).reshape(1,
#     #                                                                         -1))
#     #                         part[0::2] -= xmin
#     #                         part[1::2] -= ymin
#     #                         crop_segm.append(part.tolist())
#     #                 elif isinstance(inter, Polygon):
#     #                     crop_poly = np.squeeze(
#     #                         np.array(inter.exterior.coords[:-1]).reshape(1, -1))
#     #                     crop_poly[0::2] -= xmin
#     #                     crop_poly[1::2] -= ymin
#     #                     crop_segm.append(crop_poly.tolist())
#     #                 else:
#     #                     continue
#     #         return crop_segm

#     #     def _crop_rle(rle, crop, height, width):
#     #         if 'counts' in rle and type(rle['counts']) == list:
#     #             rle = mask_util.frPyObjects(rle, height, width)
#     #         mask = mask_util.decode(rle)
#     #         mask = mask[crop[1]:crop[3], crop[0]:crop[2]]
#     #         rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
#     #         return rle

#     #     i, j, h, w = region
#     #     crop = [j, i, j + w, i + h]
#     #     height, width = image_shape
#     #     crop_segms = []
#     #     for segm in segms:
#     #         if is_poly(segm):
#     #             import copy
#     #             import shapely.ops
#     #             from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
#     #             # Polygon format
#     #             crop_segms.append(_crop_poly(segm, crop))
#     #         else:
#     #             # RLE format
#     #             import pycocotools.mask as mask_util
#     #             crop_segms.append(_crop_rle(segm, crop, height, width))
#     #     return crop_segms

#     def apply(self, sample, context=None):
#         h = random.randint(self.min_size,
#                            min(sample['image'].shape[0], self.max_size))
#         w = random.randint(self.min_size,
#                            min(sample['image'].shape[1], self.max_size))

#         region = self.get_crop_params(sample['image'].shape[:2], [h, w])
#         return self.crop(sample, region)



KPT_FLIP_INDEX = [
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
]


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * np.array([-1., 1., -1., 1.]) + np.array([w, 0., w, 0.])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1) # TODO: Fix no clip

    if "keypoints" in target:
        keypoints = target['keypoints']
        keypoints = keypoints * np.array([-1, 1, 1]) + np.array([w, 0, 0])
        keypoints = keypoints[:, KPT_FLIP_INDEX]
        target['keypoints'] = keypoints

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if w < h:
           ow = size
           oh = int(size * h / w)
           if max_size is not None and oh > max_size:
               oh = max_size
               ow = int( max_size * w / h)
        else:
            oh = size
            ow = int(size * w / h)
            if max_size is not None and ow > max_size:
               ow = max_size
               oh = int( max_size * h / w)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * np.array([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = np.array([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].astype("float32"), size, mode="nearest")[:, 0] > 0.5

    if "keypoints" in target:
        keypoints = target["keypoints"]
        scaled_keypoints = keypoints * np.array([ratio_width, ratio_height, 1])
        target["keypoints"] = scaled_keypoints

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = np.array(padded_image[::-1])
    if "masks" in target:
        target['masks'] = F.pad(target['masks'], (0, padding[0], 0, padding[1])) # TODO
    if "keypoints" in target:
        keypoints = target['keypoints']
        keypoints = keypoints + np.array([padding[0], padding[1], 0])
        target['keypoints'] = keypoints
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop._get_param(None, img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size
    
    @staticmethod
    def _get_param(img: PIL.Image.Image, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = _get_image_size(img)
        th, tw = output_size
        
        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".
                format((th, tw), (h, w)))
        
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th + 1)
        j = random.randint(0, w - tw + 1)
        return i, j, th, tw # 左上角(y x), h和w

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        # 这样写浪费我一天时间 (x) 问题不是这里
        # region = T.RandomCrop._get_param(None, img, [h, w]) 
        region = self._get_param(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
    """

    def __init__(self, output_size=1333, aug_scale_min=0.3, aug_scale_max=2.0):
        self.desired_size = np.array(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def rescale_target(self, scaled_size, image_size, target):
        # compute rescaled targets
        image_scale = scaled_size / image_size
        ratio_height, ratio_width = image_scale

        target = target.copy()
        target["size"] = scaled_size

        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * np.array([ratio_width, ratio_height, ratio_width, ratio_height])
            target["boxes"] = scaled_boxes

        if "area" in target:
            area = target["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target["area"] = scaled_area

        if "masks" in target:
            masks = target['masks']
            masks = interpolate(
                masks[:, None].astype("float32"), scaled_size, mode="nearest")[:, 0] > 0.5
            target['masks'] = masks

        if "keypoints" in target:
            keypoints = target["keypoints"]
            scaled_keypoints = keypoints * np.array([ratio_width, ratio_height, 1])
            target["keypoints"] = scaled_keypoints
        return target

    def crop_target(self, region, target):
        i, j, h, w = region
        fields = ["labels", "area", "iscrowd"]

        target = target.copy()
        target["size"] = np.array([h, w])

        if "boxes" in target:
            boxes = target["boxes"]
            max_size = np.array([w, h], dtype=paddle.float32)
            cropped_boxes = boxes - np.array([j, i, j, i])
            cropped_boxes = np.minimum(cropped_boxes.reshape(-1, 2, 2), max_size)
            cropped_boxes = cropped_boxes.clip(min=0)
            area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(axis=1)
            target["boxes"] = cropped_boxes.reshape(-1, 4)
            target["area"] = area
            fields.append("boxes")

        if "masks" in target:
            # FIXME should we update the area here if there are no boxes?
            target['masks'] = target['masks'][:, i:i + h, j:j + w]
            fields.append("masks")

        if "keypoints" in target:
            keypoints = target['keypoints']
            keypoints = keypoints - np.array([j, i, 0])
            target['keypoints'] = keypoints
            fields.append("keypoints")

        # remove elements for which the boxes or masks that have zero area
        if "boxes" in target or "masks" in target:
            # favor boxes selection when defining which elements to keep
            # this is compatible with previous implementation
            if "boxes" in target:
                cropped_boxes = target['boxes'].reshape(-1, 2, 2)
                keep = np.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], axis=1)
            else:
                keep = target['masks'].flatten(1).any(1)

            for field in fields:
                target[field] = target[field][keep]
        return target

    def pad_target(self, padding, target):
        target = target.copy()
        if "masks" in target:
            target['masks'] = F.pad(target['masks'], (0, padding[1], 0, padding[0]))
        if "keypoints" in target:
            keypoints = target['keypoints']
            keypoints = keypoints + np.array([padding[1], padding[0], 0])
            target['keypoints'] = keypoints
        return target

    def __call__(self, image, target=None):
        image_size = image.size
        image_size = np.array(image_size[::-1])

        out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().int()

        scaled_image = F.resize(image, scaled_size.tolist())

        if target is not None:
            target = self.rescale_target(scaled_size, image_size, target)

        # randomly crop or pad images
        if random_scale > 1:
            # Selects non-zero random offset (x, y) if scaled image is larger than desired_size.
            max_offset = scaled_size - out_desired_size
            offset = (max_offset * torch.rand(2)).floor().int()
            region = (offset[0].item(), offset[1].item(),
                      out_desired_size[0].item(), out_desired_size[1].item())
            output_image = F.crop(scaled_image, *region)
            if target is not None:
                target = self.crop_target(region, target)
        else:
            padding = out_desired_size - scaled_size
            output_image = F.pad(scaled_image, [0, 0, padding[1].item(), padding[0].item()])
            if target is not None:
                target = self.pad_target(padding, target)

        return output_image, target


class RandomDistortion(object):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, prob=0.5):
        self.prob = prob
        self.tfm = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, target=None):
        if np.random.random() < self.prob:
            return self.tfm(img), target
        else:
            return img, target


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class ImgToNumpyArray(object):
    def __call__(self, img, target):
        if isinstance(img, np.ndarray):
            return img, target
        if isinstance(img, paddle.Tensor):
            return img.numpy(), target
        return F.to_tensor(img).numpy(), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / np.array([w, h, w, h], dtype=np.float32)
            target["boxes"] = boxes.astype("float32")
        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = keypoints / np.array([w, h, 1], dtype=np.float32)
            target["keypoints"] = keypoints
        return image, target


class GenerateClassificationResults(object):
    def __init__(self, num_cats):
        self.num_cats = num_cats

    def __call__(self, image, target):
        multi_labels = np.unique(target["labels"])
        multi_label_onehot = np.zeros(self.num_cats, dtype="float32")
        multi_label_onehot[multi_labels] = 1
        multi_label_weights = np.ones_like(multi_label_onehot)

        # filter crowd items
        keep = target["iscrowd"] == 0
        fields = ["labels", "area", "iscrowd"]
        if "boxes" in target:
            fields.append("boxes")
        if "masks" in target:
            fields.append("masks")
        if "keypoints" in target:
            fields.append("keypoints")
        for field in fields:
            target[field] = target[field][keep]

        if 'neg_category_ids' in target:
            # TODO:LVIS
            not_exhaustive_category_ids = [self.json_category_id_to_contiguous_id[idx] for idx in img_info['not_exhaustive_category_ids'] if idx in self.json_category_id_to_contiguous_id]
            neg_category_ids = [self.json_category_id_to_contiguous_id[idx] for idx in img_info['neg_category_ids'] if idx in self.json_category_id_to_contiguous_id]
            multi_label_onehot[not_exhaustive_category_ids] = 1
            multi_label_weights = multi_label_onehot.clone()
            multi_label_weights[neg_category_ids] = 1
        else:
            sample_prob = np.zeros_like(multi_label_onehot) - 1
            sample_prob[np.unique(target["labels"])] = 1
        target["multi_label_onehot"] = multi_label_onehot
        target["multi_label_weights"] = multi_label_weights
        target["force_sample_probs"] = sample_prob

        return image, target


class RearrangeByCls(object):
    def __init__(self, keep_keys=["size", "orig_size", "image_id", "multi_label_onehot", "multi_label_weights", "force_sample_probs"], min_keypoints_train=0):
        # TODO: min_keypoints_train is deperacated
        self.min_keypoints_train = min_keypoints_train
        self.keep_keys = keep_keys

    def __call__(self, image, target=None):
        target["class_label"] = np.unique(target["labels"])

        new_target = {}
        for icls in np.unique(target["labels"]):
            icls = icls.item()
            new_target[icls] = {}
            where = target["labels"] == icls

            new_target[icls]["boxes"] = target["boxes"][where]
            if icls == 0 and "keypoints" in target:
                new_target[icls]["keypoints"] = target["keypoints"][where]

        for key in self.keep_keys:
            new_target[key] = target[key]
        return image, new_target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
