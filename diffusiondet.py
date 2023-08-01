import cv2
import torch
from diffusiondet.util.misc import nested_tensor_from_tensor_list
from detectron2.layers import batched_nms
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, create_ddp_model, \
    AMPTrainer, SimpleTrainer, hooks
from detectron2.structures import Boxes, ImageList, Instances
from train_net_luna16 import Trainer, get_cfg, setup
from icecream import ic
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
def load_label(json_path):
    # load json file
    import json
    with open(json_path, 'r') as f:
        label = json.load(f)
    return label

def plot_image_and_boxes(image, boxes, noise_boxes, title=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots(1, figsize=(10, 10))
    "open image from path"
    image = plt.imread(image)

    ax.imshow(image, cmap='gray')
    if title:
        ax.set_title(title)

    # for box in noise_boxes:
    #     x1, y1, x2, y2 = box
    #     w, h = x2 - x1, y2 - y1
    #     rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='b', facecolor='none')
    #     ax.add_patch(rect)

    for box in boxes:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

def main(args):
    label_train = load_label("/home/vankhoa@median.cad/datasets/MedDec/Task016_Luna/coco/coco_format/train_annotations.json")
    annotation = label_train['annotations']

    cfg = setup(args)
    trainer = Trainer(cfg)
    model = trainer.build_model(cfg)
    data_loader = trainer.build_train_loader(cfg)

    for iter, batch in enumerate(data_loader):
        print(batch[0].keys())
        ic([x["file_name"] for x in batch])
        ic([x["height"] for x in batch])
        ic([x["width"] for x in batch])
        ic([x["image_id"] for x in batch])
        ic([x["instances"] for x in batch])

        image_ids = [x["image_id"] for x in batch]

        annotation_raw = [x for x in annotation if x['image_id'] in image_ids]
        ic([(x['image_id'], x['bbox'] )for x in annotation_raw])

        gt_instances = [x["instances"].to(model.device) for x in batch]
        targets, x_boxes, noises, t = model.prepare_targets(gt_instances)

        print(batch)
        images, images_whwh = model.preprocess_image(batch)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        print(images, images_whwh)
        # Feature Extraction.
        src = model.backbone(images.tensor)
        print([src[k].shape for k in src.keys()])

        features = list()
        for f in model.in_features:
            feature = src[f]
            features.append(feature)

        t = t.squeeze(-1)
        x_boxes = x_boxes * images_whwh[:, None, :]
        print(targets, x_boxes.shape, noises.shape, t)
        outputs_class, outputs_coord = model.head(features, x_boxes, t, None)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        loss_dict = model.criterion(output, targets)

        ic(targets[0].keys())
        ic(targets[0]['boxes'])
        ic(targets[0]['labels'])
        ic(targets[0]['boxes_xyxy'])
        ic(targets[0]['image_size_xyxy'])
        ic(targets[0]['image_size_xyxy_tgt'])
        ic(targets[0]['area'])
        ic(x_boxes.shape)
        ic(noises.shape)
        ic(t)
        ic(outputs_class.shape)
        ic(outputs_coord.shape)
        boxes = gt_instances[0]._fields['gt_boxes'].tensor.cpu().numpy()
        noise_boxes = x_boxes[0].cpu().numpy()
        plot_image_and_boxes(batch[0]['file_name'], boxes, noise_boxes, title=None)
        break
    pass

def ddim_sample(model, batched_inputs, backbone_feats, images_whwh, images, clip_denoised=True, do_postprocess=True):
    batch = images_whwh.shape[0]
    shape = (batch, model.num_proposals, 4)
    total_timesteps, sampling_timesteps, eta, objective = model.num_timesteps, model.sampling_timesteps, model.ddim_sampling_eta, model.objective

    # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    img = torch.randn(shape, device=model.device) # random boxes

    ensemble_score, ensemble_label, ensemble_coord = [], [], []
    x_start = None
    for time, time_next in time_pairs:
        time_cond = torch.full((batch,), time, device=model.device, dtype=torch.long)
        self_cond = x_start if model.self_condition else None

        preds, outputs_class, outputs_coord = model.model_predictions(backbone_feats, images_whwh, img, time_cond,
                                                                     self_cond, clip_x_start=clip_denoised)
        pred_noise, x_start = preds.pred_noise, preds.pred_x_start

        if model.box_renewal:  # filter
            score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
            threshold = 0.5
            score_per_image = torch.sigmoid(score_per_image)
            value, _ = torch.max(score_per_image, -1, keepdim=False)
            keep_idx = value > threshold
            num_remain = torch.sum(keep_idx)

            pred_noise = pred_noise[:, keep_idx, :]
            x_start = x_start[:, keep_idx, :]
            img = img[:, keep_idx, :]
        if time_next < 0:
            img = x_start
            continue

        alpha = model.alphas_cumprod[time]
        alpha_next = model.alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(img)

        img = x_start * alpha_next.sqrt() + \
              c * pred_noise + \
              sigma * noise

        if model.box_renewal:  # filter
            # replenish with randn boxes
            img = torch.cat((img, torch.randn(1, model.num_proposals - num_remain, 4, device=img.device)), dim=1)
        if model.use_ensemble and model.sampling_timesteps > 1:
            box_pred_per_image, scores_per_image, labels_per_image = model.inference(outputs_class[-1],
                                                                                    outputs_coord[-1],
                                                                                    images.image_sizes)
            ensemble_score.append(scores_per_image)
            ensemble_label.append(labels_per_image)
            ensemble_coord.append(box_pred_per_image)

    if model.use_ensemble and model.sampling_timesteps > 1:
        box_pred_per_image = torch.cat(ensemble_coord, dim=0)
        scores_per_image = torch.cat(ensemble_score, dim=0)
        labels_per_image = torch.cat(ensemble_label, dim=0)
        if model.use_nms:
            keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
            box_pred_per_image = box_pred_per_image[keep]
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]

        result = Instances(images.image_sizes[0])
        result.pred_boxes = Boxes(box_pred_per_image)
        result.scores = scores_per_image
        result.pred_classes = labels_per_image
        results = [result]
    else:
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        box_cls = output["pred_logits"]
        box_pred = output["pred_boxes"]
        results = model.inference(box_cls, box_pred, images.image_sizes)


    if do_postprocess:
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results




def predict(args):
    cfg = setup(args)
    model = build_model(cfg)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    threshold = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    path = '/home/vankhoa@median.cad/datasets/MedDec/Task016_Luna/coco/coco_format/train/images/1.3.6.1.4.1.14519.5.2.1.6279.6001.107109359065300889765026303943_64.jpg'

    original_image = read_image(path, format="L")

    height, width = original_image.shape[:2]
    image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))

    inputs = {"image": image, "height": height, "width": width}
    # predictions = model([inputs])[0]

    print([inputs])
    images, images_whwh = model.preprocess_image([inputs])
    if isinstance(images, (list, torch.Tensor)):
        images = nested_tensor_from_tensor_list(images)
    print(images, images_whwh)
    # Feature Extraction.
    src = model.backbone(images.tensor)
    print([src[k].shape for k in src.keys()])

    features = list()
    for f in model.in_features:
        feature = src[f]
        features.append(feature)

    # predictions = model.ddim_sample([inputs], features, images_whwh, images)[0]

    predictions = ddim_sample(model, [inputs], features, images_whwh, images)[0]

    instances = predictions['instances']
    new_instances = instances[instances.scores > threshold]
    predictions = {'instances': new_instances}

    print(predictions)

    original_image = original_image[:, :, ::-1]
    instance_mode = ColorMode.IMAGE
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )
    visualizer = Visualizer(original_image, metadata, instance_mode=instance_mode)

    instances = predictions["instances"].to('cpu')
    visualized_output = visualizer.draw_instance_predictions(predictions=instances)

    WINDOW_NAME = "test"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    img = visualized_output.get_image()[:, :, ::-1]
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow(WINDOW_NAME, img2)
    cv2.waitKey(0)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        # predict,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )