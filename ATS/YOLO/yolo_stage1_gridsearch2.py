from ultralytics import YOLO
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

conf_grid = [x / 100 for x in range(30, 96, 5)]
iou_grid  = [x / 100 for x in range(30, 96, 5)]


best_f1 = 0.0
best_iou = 0.0
best_conf = 0.0
best_mAP_at_bestF1 = 0.0

for conf in conf_grid:
    for iou in iou_grid:
        r = model.val(
            data=data,
            split="val",
            conf=conf,
            iou=iou,
            imgsz=1280,
            batch=16,
            device=0,
            plots=False,
            save=False,
            verbose=False,
        )

        # r.box / r.seg exist depending on task; choose what you care about.
        # For detection:
        map = float(r.box.map)     # mAP@0.5:0.95
        p     = float(r.box.p)
        rec   = float(r.box.r)
        f1    = (2 * p * rec / (p + rec)) if (p + rec) > 0 else 0.0

        # Track best
        if f1 > best_f1:
            best_f1 = f1
            best_iou = iou
            best_conf = conf
            best_mAP_at_bestF1 = map

        print(f"conf={conf:.2f}, iou={iou:.2f} => mAP={map:.4f}, F1={f1:.4f}")

print(f"Best F1: {best_f1}, Best IoU: {best_iou}, Best Conf: {best_conf}")

with open(args.results_log, "a") as f:
    f.write(f"{args.target_dataset_name};{best_iou};{best_conf};{best_f1};{best_mAP_at_bestF1};{args.text}\n")