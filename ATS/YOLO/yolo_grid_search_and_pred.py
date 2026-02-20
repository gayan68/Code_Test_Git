from ultralytics import YOLO
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
    parser = argparse.ArgumentParser('YOLO', add_help=False)
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

model = YOLO(weights)

# ious = [x / 100 for x in range(30, 96, 5)]
# confs = [x / 100 for x in range(30, 96, 5)]

# ious = [0.6, 0.7]
# confs = [0.6, 0.7]

# results = []
# best_mAP, best_iou, best_conf = 0, 0, 0
# for iou in ious:
#     for conf in confs:
        
#         yolo_args = dict(
#             verbose=False,
#             model=weights,
#             data=data,
#             split="val",     # or "val"
#             imgsz=1280,
#             conf=conf,
#             iou=iou,
#             max_det=2000,
#             device="cuda",
#         )
        
#         validator = DetectionValidator(args=yolo_args)
#         validator(model=model.model)          # run validation
#         metrics = validator.metrics
        
#         mAP = metrics.box.map
#         # results.append([iou, conf, mAP])

#         if mAP > best_mAP:
#             best_mAP = mAP
#             best_iou = iou
#             best_conf = conf

#         torch.cuda.empty_cache()

# print(f"Best: mAP: {best_mAP}")
# print(f"Best: iou: {best_iou}")
# print(f"Best: conf: {best_conf}")

# Calculate mAP for Test Set
yolo_args = dict(
    verbose=False,
    model=weights,
    data=data,
    split="test",     # or "val"
    imgsz=1280,
    # conf=best_conf,
    # iou=best_iou,
    max_det=2000,
    device="cuda",
)

validator = DetectionValidator(args=yolo_args)
validator(model=model.model)          # run validation
metrics = validator.metrics

test_mAP = metrics.box.map
test_map50 = metrics.box.map50
test_map75 = metrics.box.map75

print(f"mAP: {test_mAP}")

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
        max_det=300,
        device="cuda"
    )
    
    r = results[0]  # first (and only) image
    boxes_xyxy = r.boxes.xyxy.cpu().numpy()   # shape [N, 4] -> [x1, y1, x2, y2]

    with open(os.path.join(pred_boxes_dir, f"{img_id}.txt"), "w") as f:
        for box in boxes_xyxy:
            line = " ".join(str(int(x)) for x in box)
            f.write(line + "\n")

    torch.cuda.empty_cache()
print(f"Saved boxes to: {pred_boxes_dir}")