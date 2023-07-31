# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import torch
import SimpleITK as sitk

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

from diffusiondet.predictor import VisualizationDemo
from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from diffusiondet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer
from diffusiondet.util.mip import createMIP_transverse_nifti

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        default="datasets/coco/val2017",
        help="A file or directory to use as input."
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

class DiffusionDetPredictor(DefaultPredictor):
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT

        self.cfg = cfg.clone()
        self.cpu_device = torch.device("cpu")

        # print(self.cfg)

    def normalize(self, img_np):
        # clip percentile
        mean_intensity = -160
        std_intensity = 333
        lower_bound = -900
        upper_bound = 1600
        img_np = np.clip(img_np, lower_bound, upper_bound)
        img_np = (img_np - mean_intensity) / std_intensity
        return img_np

    def predict_one_slice(self, slice_img_np, slice_idx):
        predictions = self(slice_img_np)
        print(predictions)
        exit()
        return boxes, scores, labels
    
    def __call__(self, original_image):
        """
        """
        img_np = sitk.GetArrayFromImage(original_image)

        # normalize img_np
        img_np = self.normalize(img_np)

        print(img_np.shape)
        # get mip of img_np and mask_np
        slice_num = int(10/0.625) # 10 mm / 0.625 mm per slice
        mip_img_np = createMIP_transverse_nifti(img_np, slices_num=slice_num)
        print(mip_img_np.shape)
        
        all_boxes = []
        all_scores = []
        all_labels = []
        for slice_idx in range(mip_img_np.shape[0]):
            img_slice_np = mip_img_np[slice_idx]
            boxes, scores, labels = self.predict_one_slice(img_slice_np, slice_idx)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

def main(cfg, args):
    # load all volumes in the input directory
    paths = glob.glob(os.path.join(args.input, "*_vol.nii.gz"))
    print("Found {} volumes in the input directory".format(len(paths)))
    print(paths[0])

    predictor = DiffusionDetPredictor(cfg)

    for path in tqdm.tqdm(paths):
        img_sitk = sitk.ReadImage(path)
        boxes, scores, labels = predictor(img_sitk)

if __name__ == "__main__":

    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    main(cfg, args)

    

