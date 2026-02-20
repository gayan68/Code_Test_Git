from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
import numpy as np
import argparse
import os
import torch

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("device_count =", torch.cuda.device_count())
print("GPU0 name =", torch.cuda.get_device_name(0))

def get_args_parser():
    parser = argparse.ArgumentParser('YOLO', add_help=False)
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--data_yaml', type=str, default="")
    parser.add_argument('--img_path', type=str, default="")
    parser.add_argument('--gt_xml', type=str, default="")
    parser.add_argument('--save_boxes_root', type=str, default="")
    parser.add_argument('--results_log', type=str, default="results_log.txt")
    parser.add_argument("--target_dataset_name", type=str,  default="READ_2016")
    parser.add_argument("--text", type=str,  default="")

    return parser.parse_args()

args = get_args_parser()

weights = args.model_path
data = args.data_yaml
save_path = args.save_boxes_root
model_name = weights.split('/')[-3]

model = YOLO(weights)  # or .pt you want to tune


ious = [x / 100 for x in range(30, 96, 5)]
confs = [x / 100 for x in range(30, 96, 5)]


best_f1, best_iou, best_conf = 0, 0, 0
for iou in ious:
    for conf in confs:
        
        yolo_args = dict(
            verbose=False,
            model=weights,
            data=data,
            split="val",     # or "val"
            imgsz=1280,
            conf=conf,
            iou=iou,
            max_det=2000,
            device="cuda",
        )
        
        validator = DetectionValidator(args=yolo_args)
        validator(model=model.model)          # run validation
        metrics = validator.metrics

        precision = metrics.box.p
        recall = metrics.box.r
        mAP = metrics.box.map

        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_iou = iou
            best_conf = conf
            best_mAP_at_bestF1 = mAP

        print(f"conf={conf:.2f}, iou={iou:.2f} → F1={f1:.4f}, mAP={mAP:.4f}")

        torch.cuda.empty_cache()

print(f"Best: mAP: {best_mAP_at_bestF1}; F1: {best_f1}; IoU: {best_iou}; Conf: {best_conf}")


with open(args.results_log, "a") as f:
    f.write(f"{args.target_dataset_name};{best_iou};{best_conf};{best_f1};{best_mAP_at_bestF1};{args.text}\n")