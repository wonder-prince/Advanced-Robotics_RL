"""Inference fallback when the original C++/CUDA extension is not installed.

Python prefers a compiled _C extension over this module when one is present.
The Week9 non-DCN CenterMask demo needs NMS and ROIAlign forward only.
Training and deformable operators still require the original native extension.
"""

import torch
from torchvision import ops

IS_TORCHVISION_FALLBACK = True


def _continuous_boxes(boxes):
    # The original kernels use inclusive pixel coordinates (x2 - x1 + 1).
    converted = boxes.clone()
    converted[:, 2:] += 1
    return converted


def nms(boxes, scores, threshold):
    return ops.nms(_continuous_boxes(boxes), scores, threshold)


def ml_nms(boxes, scores, labels, threshold):
    return ops.batched_nms(
        _continuous_boxes(boxes), scores, labels.to(dtype=torch.int64), threshold
    )


def roi_align_forward(features, rois, spatial_scale, height, width, sampling_ratio):
    return ops.roi_align(
        features,
        rois,
        output_size=(height, width),
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
        aligned=False,
    )


def roi_align_backward(*args):
    raise RuntimeError(
        "The Week9 torchvision fallback supports inference only. "
        "Build the original C++/CUDA extension for training."
    )
