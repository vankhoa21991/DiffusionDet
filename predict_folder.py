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
<<<<<<< Updated upstream
=======
import pickle
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======
    os.makedirs(args.output, exist_ok=True)
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
=======
def normalize(img_np):
    # clip percentile
    mean_intensity = -160
    std_intensity = 333
    lower_bound = -900
    upper_bound = 1600
    img_np = np.clip(img_np, lower_bound, upper_bound)
    img_np = (img_np - mean_intensity) / std_intensity
    return img_np


>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
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
=======
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


def main(cfg, args, threshold=0.5):
>>>>>>> Stashed changes
    # load all volumes in the input directory
    paths = glob.glob(os.path.join(args.input, "*_vol.nii.gz"))
    print("Found {} volumes in the input directory".format(len(paths)))
    print(paths[0])

<<<<<<< Updated upstream
    predictor = DiffusionDetPredictor(cfg)

    for path in tqdm.tqdm(paths):
        img_sitk = sitk.ReadImage(path)
        boxes, scores, labels = predictor(img_sitk)

if __name__ == "__main__":

=======
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
        slice_num = int(10 / 0.625)  # 10 mm / 0.625 mm per slice
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
                        boxes3d[i] = np.append(boxes[i], [slice_idx, slice_idx+slice_num])

                all_boxes.append(boxes3d)
                all_scores.append(scores)
                all_labels.append(labels)
            except Exception as e:
                print(e)
                print("Error in slice {} of serie uid {}".format(slice_idx, uid) )
                continue

        # process all_boxes
        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        result = {
            "pred_boxes": all_boxes,
            "pred_scores": all_scores,
            "pred_labels": all_labels,
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
>>>>>>> Stashed changes
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    main(cfg, args)

    

