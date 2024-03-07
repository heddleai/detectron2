#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import os
import os.path as osp
from typing import Any, ClassVar, Dict, List
import torch

from detectron2.config import CfgNode, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances

from densepose import add_densepose_config
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import (
    DensePoseOutputsTextureVisualizer,
    DensePoseOutputsVertexVisualizer,
    get_texture_atlases,
)
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from densepose.vis.densepose_results_textures import (
    DensePoseResultsVisualizerWithTexture,
    get_texture_atlas,
)
from densepose.vis.extractor import (
    CompoundExtractor,
    create_extractor,
)

DOC = """Apply Net - a tool to print / visualize DensePose results
"""

LOGGER_NAME = "apply_net"
logger = logging.getLogger(LOGGER_NAME)

class DenseposeAction:
    """
    Show action that visualizes selected entries on an image
    """

    COMMAND: ClassVar[str] = "show"
    VISUALIZERS: ClassVar[Dict[str, object]] = {
        "dp_contour": DensePoseResultsContourVisualizer,
        "dp_segm": DensePoseResultsFineSegmentationVisualizer,
        "dp_u": DensePoseResultsUVisualizer,
        "dp_v": DensePoseResultsVVisualizer,
        "dp_iuv_texture": DensePoseResultsVisualizerWithTexture,
        "dp_cse_texture": DensePoseOutputsTextureVisualizer,
        "dp_vertex": DensePoseOutputsVertexVisualizer,
        "bbox": ScoredBoundingBoxVisualizer,
    }

    @classmethod
    def setup(cls: type, output_dir: str):
        cls.add_arguments()
        logger.info(f"Loading config from {cls.cfg}")
        opts = []
        cfg = cls.setup_config(cls.cfg, cls.model, opts)
        predictor = DefaultPredictor(cfg)
        cls.predictor = predictor
        cls.context = cls.create_context(cls.cfg)
        cls.output_dir = output_dir

    @classmethod
    def execute(cls: type, input: List[str]):
        if len(input) == 0:
            logger.warning(f"No input images for {input}")
            return
        good = []
        for file_name in input:
            img = read_image(file_name, format="BGR")  # predictor expects BGR image.
            with torch.no_grad():
                outputs = cls.predictor(img)["instances"]
                is_good = cls.execute_on_outputs(cls.context, {"file_name": file_name, "image": img}, outputs)
                if is_good:
                    good.append(file_name)
        return good

    @classmethod
    def add_arguments(cls: type):
        cls.cfg = osp.join(__file__, "../configs/densepose_rcnn_R_50_FPN_s1x.yaml")
        cls.model = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
        cls.visualizations = "dp_segm"
        cls.min_score = 0.8
        cls.nms_thresh = None
        cls.texture_atlas = None
        cls.texture_atlases_map = None

    @classmethod
    def get_config(
        cls: type, config_fpath: str, model_fpath: str, opts: List[str]
    ):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_fpath)
        if opts:
            cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = model_fpath
        cfg.freeze()
        return cfg

    @classmethod
    def setup_config(
        cls: type, config_fpath: str, model_fpath: str, opts: List[str]
    ):
        opts.append("MODEL.ROI_HEADS.SCORE_THRESH_TEST")
        opts.append(str(cls.min_score))
        if cls.nms_thresh is not None:
            opts.append("MODEL.ROI_HEADS.NMS_THRESH_TEST")
            opts.append(str(cls.nms_thresh))
        cfg = cls.get_config(config_fpath, model_fpath, opts)
        return cfg

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        import cv2
        import numpy as np
        if len(outputs) == 0:
            return False
        visualizer = context["visualizer"]
        extractor = context["extractor"]
        image_fpath = entry["file_name"]
        image_dir = os.path.basename(os.path.dirname(image_fpath))
        logger.info(f"Processing {image_fpath}")
        image = cv2.cvtColor(entry["image"], cv2.COLOR_BGR2GRAY)
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        
        data = extractor(outputs)
        image_vis = visualizer.visualize(image, data)

        out_dir = os.path.join(cls.output_dir, image_dir)
        out_fname = os.path.join(out_dir, os.path.basename(image_fpath))
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cv2.imwrite(out_fname, image_vis)
        logger.info(f"Output saved to {out_fname}")
        context["entry_idx"] += 1
        return True

    @classmethod
    def create_context(cls: type, cfg: CfgNode) -> Dict[str, Any]:
        vis_specs = cls.visualizations.split(",")
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            texture_atlas = get_texture_atlas(cls.texture_atlas)
            texture_atlases_dict = get_texture_atlases(cls.texture_atlases_map)
            vis = cls.VISUALIZERS[vis_spec](
                cfg=cfg,
                texture_atlas=texture_atlas,
                texture_atlases_dict=texture_atlases_dict,
            )
            visualizers.append(vis)
            extractor = create_extractor(vis)
            extractors.append(extractor)
        visualizer = CompoundVisualizer(visualizers)
        extractor = CompoundExtractor(extractors)
        context = {
            "extractor": extractor,
            "visualizer": visualizer,
            "entry_idx": 0,
        }
        return context

