"""Focused regressions for the Week9 inference path."""

import numpy as np
import pytest
import torch

from maskrcnn_benchmark import _C
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import ROIAlign
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from predictor import COCODemo
from week8_experiment import check_finite_head_outputs, parse_args, render, serialize_instances


def test_fallback_nms_preserves_inclusive_pixels_and_separates_classes():
    if not getattr(_C, "IS_TORCHVISION_FALLBACK", False):
        pytest.skip("test targets the portable fallback")
    boxes = torch.tensor([[0., 0., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 0.]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    labels = torch.tensor([1., 1., 2.])
    assert _C.ml_nms(boxes, scores, labels, 0.4).tolist() == [0, 2]
    assert _C.nms(boxes[:2], scores[:2], 0.4).tolist() == [0]
    assert _C.ml_nms(boxes[:0], scores[:0], labels[:0], 0.4).numel() == 0


def test_roi_align_uses_original_unaligned_coordinate_convention():
    feature = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    rois = torch.tensor([[0., 0., 0., 2., 2.]])
    pooled = ROIAlign((1, 1), spatial_scale=1., sampling_ratio=2)(feature, rois)
    torch.testing.assert_close(pooled, torch.tensor([[[[5.]]]]))
    empty = ROIAlign((1, 1), spatial_scale=1., sampling_ratio=2)(feature, rois[:0])
    assert empty.shape == (0, 1, 1, 1)


def test_fcos_candidates_accept_noncontiguous_class_scores():
    from maskrcnn_benchmark.modeling.rpn.fcos.inference import FCOSPostProcessor
    postprocessor = FCOSPostProcessor(0.03, 100, 0.6, 50, 0, 3, 1)
    predictions = postprocessor.forward_for_single_feature_map(
        torch.tensor([[2., 2.], [4., 2.], [2., 4.], [4., 4.]]),
        torch.full((1, 2, 2, 2), 10.),
        torch.ones(1, 4, 2, 2),
        torch.zeros(1, 1, 2, 2),
        [(8, 8)],
    )[0]
    assert len(predictions) == 8
    assert set(predictions.get_field("labels").tolist()) == {1, 2}
    torch.testing.assert_close(predictions.get_field("scores"), torch.full((8,), 0.5 * torch.sigmoid(torch.tensor(10.)).item()))


def test_empty_model_output_remains_a_valid_empty_prediction():
    demo = object.__new__(COCODemo)
    demo.cfg = cfg.clone()
    demo.cfg.MODEL.DEVICE = "cpu"
    demo.cfg.MODEL.MASK_ON = True
    demo.device = demo.cpu_device = torch.device("cpu")
    demo.transforms = lambda image: torch.zeros(3, image.shape[0], image.shape[1])
    demo.model = lambda image_list: []
    demo.masker = Masker(threshold=0.5, padding=1)
    prediction = demo.compute_prediction(np.zeros((32, 48, 3), dtype=np.uint8))
    assert len(prediction) == 0
    assert prediction.size == (48, 32)
    assert prediction.get_field("mask").shape == (0, 1, 32, 48)
    demo.confidence_threshold = 0.9
    selected = demo.select_top_predictions(prediction)
    assert len(selected) == 0
    assert serialize_instances(demo, selected) == []


def test_confidence_threshold_uses_strict_greater_than_and_mask_scores():
    demo = object.__new__(COCODemo)
    prediction = BoxList(torch.tensor([[0., 0., 3., 3.]] * 4), (4, 4), "xyxy")
    prediction.add_field("scores", torch.tensor([0.1, 0.5, 0.6, 0.95]))
    prediction.add_field("labels", torch.ones(4, dtype=torch.int64))
    counts = []
    for threshold in (0.1, 0.5, 0.9):
        demo.confidence_threshold = threshold
        counts.append(len(demo.select_top_predictions(prediction)))
    assert counts == [3, 2, 1]
    prediction.add_field("mask_scores", torch.tensor([0.9, 0.2, 0.1, 0.3]))
    demo.confidence_threshold = 0.5
    selected = demo.select_top_predictions(prediction)
    assert len(selected) == 1
    torch.testing.assert_close(selected.get_field("scores"), torch.tensor([0.1]))


def test_frozen_batchnorm_accepts_only_the_unused_training_counter():
    from maskrcnn_benchmark.layers.batch_norm import FrozenBatchNorm2d
    layer = FrozenBatchNorm2d(2)
    state = torch.nn.BatchNorm2d(2).state_dict()
    layer.load_state_dict(state, strict=True)
    image = torch.randn(1, 2, 3, 3)
    torch.testing.assert_close(layer(image), image)
    state["unexpected_tensor"] = torch.tensor(1)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        layer.load_state_dict(state, strict=True)


def test_mobilenet_frozen_batchnorm_keeps_near_zero_variance_finite():
    from maskrcnn_benchmark.layers.batch_norm import FrozenBatchNorm2d
    from maskrcnn_benchmark.modeling.backbone.mobilenet import conv_bn
    layer = conv_bn(3, 2, 1)[1]
    assert isinstance(layer, FrozenBatchNorm2d)
    state = torch.nn.BatchNorm2d(2).state_dict()
    state["running_var"] = torch.tensor([-1.4133e-12, 0.])
    layer.load_state_dict(state, strict=True)
    reference = torch.nn.BatchNorm2d(2, eps=1e-5).eval()
    reference.load_state_dict(state)
    image = torch.randn(1, 2, 3, 3)
    assert torch.isfinite(layer(image)).all()
    torch.testing.assert_close(layer(image), reference(image))


def test_strict_checkpoint_loading_rejects_incomplete_model(tmp_path, monkeypatch):
    import predictor
    monkeypatch.setattr(predictor, "build_detection_model", lambda config: torch.nn.Linear(2, 2))
    checkpoint = tmp_path / "incomplete.pth"
    torch.save({"model": {"weight": torch.ones(2, 2)}}, checkpoint)
    config = cfg.clone()
    config.MODEL.DEVICE = "cpu"
    config.MODEL.WEIGHT = str(checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        COCODemo(config, confidence_threshold=0.5, strict_weights=True)


def test_strict_checkpoint_ignores_stale_last_checkpoint(tmp_path, monkeypatch):
    import predictor
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(predictor, "build_detection_model", lambda config: model)
    checkpoint = tmp_path / "valid.pth"
    torch.save({"model": {"module.weight": torch.ones(2, 2), "module.bias": torch.zeros(2)}}, checkpoint)
    (tmp_path / "last_checkpoint").write_text("missing-other-model.pth")
    config = cfg.clone()
    config.MODEL.DEVICE = "cpu"
    config.MODEL.WEIGHT = str(checkpoint)
    config.OUTPUT_DIR = str(tmp_path)
    COCODemo(config, confidence_threshold=0.5, strict_weights=True)
    torch.testing.assert_close(model.weight, torch.ones(2, 2))


def test_masks_are_saved_as_roundtrippable_coco_rle_and_text_renders():
    from pycocotools import mask as mask_utils
    demo = object.__new__(COCODemo)
    demo.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    predictions = BoxList(torch.tensor([[0., 0., 7., 7.]]), (8, 8), "xyxy")
    predictions.add_field("labels", torch.tensor([1]))
    predictions.add_field("scores", torch.tensor([0.9]))
    masks = torch.zeros(1, 1, 8, 8, dtype=torch.uint8)
    masks[:, :, 2:6, 2:6] = 1
    predictions.add_field("mask", masks)
    instances = serialize_instances(demo, predictions)
    decoded = mask_utils.decode(instances[0]["segmentation"])
    np.testing.assert_array_equal(decoded, masks[0, 0].numpy())
    result = render(demo, np.zeros((8, 8, 3), dtype=np.uint8), predictions)
    assert result.any()


def test_nonfinite_model_outputs_cannot_be_reported_as_zero_detections():
    with pytest.raises(FloatingPointError, match="Nonfinite FCOS"):
        check_finite_head_outputs(None, (), ([torch.tensor([float("nan")])],))


@pytest.mark.parametrize("args", [["--conf_th", "1.1"], ["--thresholds", "-0.1"], ["--repeats", "0"]])
def test_invalid_experiment_parameters_fail_before_running(args):
    with pytest.raises(SystemExit):
        parse_args(args)
