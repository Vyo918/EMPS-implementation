# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Sequence
import torch

from idm_vton.detectron2.layers.nms import batched_nms
from idm_vton.detectron2.structures.instances import Instances

from idm_vton.densepose.vis.bounding_box import BoundingBoxVisualizer, ScoredBoundingBoxVisualizer
from idm_vton.densepose.vis.densepose import DensePoseResultsVisualizer

from .base import CompoundVisualizer

Scores = Sequence[float]


def extract_scores_from_instances(instances: Instances, select=None):
    if instances.has("scores"):
        return instances.scores if select is None else instances.scores[select]
    return None


def extract_boxes_xywh_from_instances(instances: Instances, select=None):
    if instances.has("pred_boxes"):
        boxes_xywh = instances.pred_boxes.tensor.clone()
        boxes_xywh[:, 2] -= boxes_xywh[:, 0]
        boxes_xywh[:, 3] -= boxes_xywh[:, 1]
        return boxes_xywh if select is None else boxes_xywh[select]
    return None


def create_extractor(visualizer: object):
    """
    Create an extractor for the provided visualizer
    """
    if isinstance(visualizer, CompoundVisualizer):
        extractors = [create_extractor(v) for v in visualizer.visualizers]
        return CompoundExtractor(extractors)
    elif isinstance(visualizer, DensePoseResultsVisualizer):
        return DensePoseResultExtractor()
    elif isinstance(visualizer, ScoredBoundingBoxVisualizer):
        return CompoundExtractor([extract_boxes_xywh_from_instances, extract_scores_from_instances])
    elif isinstance(visualizer, BoundingBoxVisualizer):
        return extract_boxes_xywh_from_instances
    else:
        logger = logging.getLogger(__name__)
        logger.error(f"Could not create extractor for {visualizer}")
        return None


class BoundingBoxExtractor(object):
    """
    Extracts bounding boxes from instances
    """

    def __call__(self, instances: Instances):
        boxes_xywh = extract_boxes_xywh_from_instances(instances)
        return boxes_xywh


class ScoredBoundingBoxExtractor(object):
    """
    Extracts bounding boxes from instances
    """

    def __call__(self, instances: Instances, select=None):
        scores = extract_scores_from_instances(instances)
        boxes_xywh = extract_boxes_xywh_from_instances(instances)
        if (scores is None) or (boxes_xywh is None):
            return (boxes_xywh, scores)
        if select is not None:
            scores = scores[select]
            boxes_xywh = boxes_xywh[select]
        return (boxes_xywh, scores)


class DensePoseResultExtractor(object):
    """
    Extracts DensePose result from instances
    """

    def __call__(self, instances: Instances, select=None):
        boxes_xywh = extract_boxes_xywh_from_instances(instances)
        if instances.has("pred_densepose") and (boxes_xywh is not None):
            dpout = instances.pred_densepose
            if select is not None:
                dpout = dpout[select]
                boxes_xywh = boxes_xywh[select]
            return dpout.to_result(boxes_xywh)
        else:
            return None


class CompoundExtractor(object):
    """
    Extracts data for CompoundVisualizer
    """

    def __init__(self, extractors):
        self.extractors = extractors

    def __call__(self, instances: Instances, select=None):
        datas = []
        for extractor in self.extractors:
            data = extractor(instances, select)
            datas.append(data)
        return datas


class NmsFilteredExtractor(object):
    """
    Extracts data in the format accepted by NmsFilteredVisualizer
    """

    def __init__(self, extractor, iou_threshold):
        self.extractor = extractor
        self.iou_threshold = iou_threshold

    def __call__(self, instances: Instances, select=None):
        scores = extract_scores_from_instances(instances)
        boxes_xywh = extract_boxes_xywh_from_instances(instances)
        if boxes_xywh is None:
            return None
        select_local_idx = batched_nms(
            boxes_xywh,
            scores,
            torch.zeros(len(scores), dtype=torch.int32),
            iou_threshold=self.iou_threshold,
        ).squeeze()
        select_local = torch.zeros(len(boxes_xywh), dtype=torch.bool, device=boxes_xywh.device)
        select_local[select_local_idx] = True
        select = select_local if select is None else (select & select_local)
        return self.extractor(instances, select=select)


class ScoreThresholdedExtractor(object):
    """
    Extracts data in the format accepted by ScoreThresholdedVisualizer
    """

    def __init__(self, extractor, min_score):
        self.extractor = extractor
        self.min_score = min_score

    def __call__(self, instances: Instances, select=None):
        scores = extract_scores_from_instances(instances)
        if scores is None:
            return None
        select_local = scores > self.min_score
        select = select_local if select is None else (select & select_local)
        data = self.extractor(instances, select=select)
        return data
