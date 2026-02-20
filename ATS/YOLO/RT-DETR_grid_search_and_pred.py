from ultralytics import RTDETR
import matplotlib.pyplot as plt
import cv2
from ultralytics.models.yolo.detect.val import DetectionValidator
import os
import argparse
import torch

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("device_count =", torch.cuda.device_count())
print("GPU0 name =", torch.cuda.get_device_name(0))

def get_args_parser():
    parser = argparse.ArgumentParser('RTDETR', add_help=False)
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--data_yaml', type=str, default="")
    parser.add_argument('--img_path', type=str, default="")
    parser.add_argument('--save_boxes_root', type=str, default="")
    parser.add_argument('--results_log', type=str, default="results_log.txt")
    parser.add_argument("--target_dataset_name", type=str,  default="READ_2016")
    parser.add_argument("--text", type=str,  default="")

    return parser.parse_args()


args = get_args_parser()

weights = args.model_path
data = args.data_yaml
save_path = args.save_boxes_root

model = RTDETR(weights)

ious  = [x / 100 for x in range(30, 96, 5)]
confs = [x / 100 for x in range(5,  86, 5)]  # wider, includes lower confs

best_score, best_iou, best_conf = -1, None, None

for iou in ious:
    for conf in confs:
        metrics = model.val(
            verbose=False,
            data=data,
            split="val",     # or "val"
            imgsz=1280,
            max_det=2000,
            device="cuda",
            conf=conf,
            iou=iou,
            plots=False,     # recommended for stability
            single_cls=True, # if READ is 1 class
            batch=1,         # safer VRAM
            half=False,       # fp16 on GPU
            batch = 16
        )
        val_mAP = metrics.box.map
        if val_mAP > best_mAP:
            best_mAP = val_mAP
            best_iou = iou
            best_conf = conf

test_mAP = metrics.box.map


# Calculate mAP for Test Set
metrics = model.val(
    verbose=False,
    data=data,
    split="test",     # or "val"
    imgsz=1280,
    max_det=2000,
    device="cuda",
    conf=best_conf,
    iou=best_iou,
    plots=False,     # recommended for stability
    single_cls=True, # if READ is 1 class
    batch=1,         # safer VRAM
    half=True,       # fp16 on GPU
)

test_mAP = metrics.box.map
test_map50 = metrics.box.map50
test_map75 = metrics.box.map75


model_name = weights.split('/')[-3]
with open(args.results_log, "a") as f:
    f.write(f"{model_name};{test_map50};{test_map75};{test_mAP};{args.text}\n")

torch.cuda.empty_cache()

pred_boxes_dir = os.path.join(args.save_boxes_root, model_name, args.target_dataset_name)
os.makedirs(pred_boxes_dir, exist_ok=True)

images = [os.path.join(args.img_path, fname) for fname in os.listdir(args.img_path)]

for image_path in images:
    img_id = os.path.basename(image_path).split('.')[0]
    
    results = model.predict(    
        source=image_path,   # image / dir / video
        verbose=False,
        imgsz=1280,
        max_det=2000,
        device="cuda",
        conf=best_conf,
        iou=best_iou,
        plots=False,     # recommended for stability
        single_cls=True, # if READ is 1 class
        batch=1,         # safer VRAM
        half=True,       # fp16 on GPU
)
    
    r = results[0]  # first (and only) image
    boxes_xyxy = r.boxes.xyxy.cpu().numpy()   # shape [N, 4] -> [x1, y1, x2, y2]

    with open(os.path.join(pred_boxes_dir, f"{img_id}.txt"), "w") as f:
        for box in boxes_xyxy:
            line = " ".join(str(int(x)) for x in box)
            f.write(line + "\n")

    torch.cuda.empty_cache()
print(f"Saved boxes to: {pred_boxes_dir}")