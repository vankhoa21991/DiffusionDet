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
import pickle
import SimpleITK as sitk
from ensemble_boxes import *

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from nndet.inference.detection import batched_nms_model, batched_wbc_ensemble

from diffusiondet.predictor import VisualizationDemo
from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from diffusiondet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer
from diffusiondet.util.mip import createMIP_transverse_nifti

def setup_cfg(args):
    # load config from file and command-line arguments
    os.makedirs(args.output, exist_ok=True)
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
        help="A file or directory to save output_lidc10 visualizations. "
        "If not given, will show output_lidc10 in an OpenCV window.",
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

def normalize(img_np):
    # clip percentile
    mean_intensity = -160
    std_intensity = 333
    lower_bound = -900
    upper_bound = 1600
    img_np = np.clip(img_np, lower_bound, upper_bound)
    img_np = (img_np - mean_intensity) / std_intensity
    return img_np


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

    def __call__(self, original_image):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]

            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


def main(cfg, args, threshold=0.2):
    # load all volumes in the input directory
    paths = glob.glob(os.path.join(args.input, "*_vol.nii.gz"))
    print("Found {} volumes in the input directory".format(len(paths)))
    print(paths[0])

    # predictor = DiffusionDetPredictor(cfg)
    predictor = DefaultPredictor(cfg)

    for path in tqdm.tqdm(paths):
        uid = path.split("/")[-1].split("_")[0]
        img_sitk = sitk.ReadImage(path)
        spacing = img_sitk.GetSpacing()
        origin = img_sitk.GetOrigin()
        direction = img_sitk.GetDirection()
        size = img_sitk.GetSize()
        img_np = sitk.GetArrayFromImage(img_sitk)

        # normalize img_np
        img_np = normalize(img_np)

        print(img_np.shape)
        # get mip of img_np and mask_np
        slice_num = int(30 / 0.625)  # 10 mm / 0.625 mm per slice
        print(slice_num) # 16
        mip_img_np = createMIP_transverse_nifti(img_np, slices_num=slice_num)
        print(mip_img_np.shape)

        all_boxes = []
        all_scores = []
        all_labels = []
        for slice_idx in tqdm.tqdm(range(mip_img_np.shape[0])):
            try:
                img_slice_np = mip_img_np[slice_idx]
                # scale to 0-255
                img_slice_np = (img_slice_np - img_slice_np.min()) / (img_slice_np.max() - img_slice_np.min()) * 255
                img_slice_np = np.expand_dims(img_slice_np, axis=2)
                predictions = predictor(img_slice_np)
                instances = predictions['instances']
                new_instances = instances[instances.scores > threshold]
                boxes = new_instances.pred_boxes.tensor.cpu().numpy()
                scores = new_instances.scores.cpu().numpy()
                labels = new_instances.pred_classes.cpu().numpy()

                if len(boxes) == 0:
                    continue
                else:
                    boxes3d = np.zeros((len(boxes), 6))
                    for i in range(len(boxes)):
                        boxes3d[i] = np.append([slice_idx, slice_idx+slice_num], boxes[i]) # (z0, z1, x0, y0, x1, y1)
                        # transpose to (z0, y0, z1, y1, x0, x1)
                        # boxes3d[i] = boxes3d[i][[0, 3, 1, 5, 2, 4]]
                        boxes3d[i] = boxes3d[i][[2, 3, 0, 4, 5, 1]]
                        all_boxes.append(boxes3d[i])
                        all_scores.append(scores[i])
                        all_labels.append(labels[i])
            except Exception as e:
                print(e)
                print("Error in slice {} of serie uid {}".format(slice_idx, uid) )
                continue

        # process all_boxes
        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # top k=100
        _, idx = torch.from_numpy(all_scores).sort(descending=True)
        idx = idx[:100]
        all_boxes = all_boxes[idx]
        all_scores = all_scores[idx]
        all_labels = all_labels[idx]

        # apply nms
        # all_boxes, all_scores, all_labels, _ = batched_nms_model(torch.from_numpy(all_boxes), torch.from_numpy(all_scores), torch.from_numpy(all_labels), torch.from_numpy(all_labels), iou_thresh=0.01)

        # apply wbc
        # num_models = 1
        # n_exp_preds = torch.tensor([num_models] * len(all_boxes)).to(torch.from_numpy(all_boxes))
        # weights = torch.tensor([1.0/len(all_boxes)] * len(all_boxes)).to(torch.from_numpy(all_boxes))
        # all_boxes_new, all_scores_new, all_labels_new = batched_wbc_ensemble(torch.from_numpy(all_boxes), torch.from_numpy(all_scores), torch.from_numpy(all_labels), weights,
        #                                                          iou_thresh=0.005,
        #                                                          score_thresh=0.0,
        #                                                          n_exp_preds=n_exp_preds)

        # convert to numpy
        # all_boxes_new = all_boxes_new.numpy()
        # all_scores_new = all_scores_new.numpy()
        # all_labels_new = all_labels_new.numpy()


        # apply wbs
        iou_thr = 0.005
        skip_box_thr = 0.0001
        sigma = 0.1
        weights = [1]

        # all_boxes_new = []
        # scale to 0-1
        all_boxes_new = [[b[0]/size[0], b[1]/size[1], b[2]/size[2],b[3]/size[0], b[4]/size[1], b[5]/size[2]]for b in list(all_boxes)]
        all_boxes_new, all_scores_new, all_labels_new = weighted_boxes_fusion_3d([all_boxes_new], [list(all_scores)], [list(all_labels)], weights=weights,
                                                      iou_thr=iou_thr, skip_box_thr=skip_box_thr)

        # scale back to original size
        all_boxes_new = [[b[2]*size[2], b[1]*size[1], b[5]*size[2], b[4]*size[1], b[0]*size[0], b[3]*size[0]]for b in list(all_boxes_new)]
        all_boxes_new = np.array(all_boxes_new)
        all_scores_new = np.array(all_scores_new)
        all_labels_new = np.array(all_labels_new)


        result = {
            "pred_boxes": all_boxes_new,
            "pred_scores": all_scores_new,
            "pred_labels": all_labels_new,
            "itk_spacing": spacing,
            "itk_origin": origin,
            "itk_direction": direction,
            "original_size_of_raw_data": size,
            "restore": True
        }

        # save result
        save_path = os.path.join(args.output, uid + "_boxes.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(result, f)
        print("Saved result to {}".format(save_path))



if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    main(cfg, args)

    

