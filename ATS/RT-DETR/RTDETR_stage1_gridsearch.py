import matplotlib.pyplot as plt
import cv2
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import xml.etree.ElementTree as ET
from torchvision.ops import nms
import os
import argparse
import torch
import numpy as np
from ultralytics import RTDETR
from f1_score_for_bb  import compute_f1 

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("device_count =", torch.cuda.device_count())
print("GPU0 name =", torch.cuda.get_device_name(0))

def get_args_parser():
    parser = argparse.ArgumentParser('RTDETR', add_help=False)
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

model = RTDETR(weights)
show_labels = False

def polygon_to_bbox(poly):
    xs = [point[0] for point in poly]
    ys = [point[1] for point in poly]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return [x_min, y_min, x_max, y_max]

def get_GT_Boxes_onePage(gt_path, image_id):
    tree = ET.parse(os.path.join(gt_path, f"{image_id}.xml"))
    root = tree.getroot()
    ns = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    # w = int(root.find("pc:Page", ns).attrib['imageWidth'])
    # h = int(root.find("pc:Page", ns).attrib['imageHeight'])
    
    ## Extract GT Points ######################
    gt_points = []
    for line in root.findall(".//pc:TextLine", ns):
        coords_elem = line.find("pc:Coords", ns)
        points_str = coords_elem.attrib['points']
        points = [tuple(map(int, p.split(','))) for p in points_str.split()]
        gt_points.append(points)

    gt_boxes = []
    for poly in gt_points:
        gt_boxes.append(polygon_to_bbox(poly))
    return  gt_boxes

def predict_boxes_for_image(image_paths, conf=0.001, imgsz=1280, max_det=300):
    prediction = []

    for img_path in image_paths:
        result = model.predict(
            img_path,
            conf=conf,
            imgsz=imgsz,
            max_det=max_det,
            verbose=False,
            device=0
        )

        r = result[0]
        if r.boxes is None or len(r.boxes) == 0:
            boxes = np.zeros((0, 6), dtype=np.float32)
            confidences = np.zeros((0,), dtype=np.float32)
        else:
            boxes = r.boxes.data.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()

        image = cv2.imread(img_path)
        img_h, img_w = image.shape[:2]
        page_size = (img_w, img_h)

        prediction.append({
            "image_path": img_path,
            "boxes": boxes[:, :4].tolist() if len(boxes) > 0 else [],
            "confidences": confidences.tolist() if len(confidences) > 0 else [],
            "page_size": page_size
        })

    return prediction

    
def nms_filter_results(predictions, conf, nms_active=False, nms_threshold=0.5):
    # print(f"Displaying results with conf={conf}")

    metric = MeanAveragePrecision(iou_type="bbox")
    f1_scores = []

    for pred in predictions:
        img_path = pred["image_path"]
        boxes = np.array(pred["boxes"])
        confidences = np.array(pred["confidences"])
        page_size = pred["page_size"]

        keep = confidences >= conf
        boxes_f = boxes[keep]
        scores_f = confidences[keep]

        apply_nms = nms_active and len(boxes_f) > 0
        if apply_nms:
            boxes_t = torch.tensor(boxes_f[:, :4], dtype=torch.float32)
            scores_t = torch.tensor(scores_f, dtype=torch.float32)
            keep = nms(boxes_t, scores_t, nms_threshold).cpu().numpy()

            # print(f"NMS active, originally found {len(boxes)} boxes, {len(keep)} boxes remain after NMS.")
            boxes_f = boxes_f[keep]
            scores_f = scores_f[keep]

        img_id = os.path.splitext(os.path.basename(img_path))[0]
        gt_boxes = get_GT_Boxes_onePage(args.gt_xml, img_id)
        pred_boxes = boxes_f[:, 0:4].tolist() if len(boxes_f) > 0 else []

        f1_scores.append(compute_f1(gt_boxes, pred_boxes, iou_thresh=0.5, page_size=page_size)["f1"])

        targets = [{
            "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
            "labels": torch.zeros((len(gt_boxes),), dtype=torch.int64),
        }]

        preds = [{
            "boxes": torch.tensor(boxes_f[:, 0:4], dtype=torch.float32),
            "scores": torch.tensor(scores_f, dtype=torch.float32),
            "labels": torch.zeros((len(scores_f),), dtype=torch.int64),
        }]

        metric.update(preds, targets)

    result_ap = metric.compute()
    f1_score = float(np.mean(f1_scores)) if len(f1_scores) else 0.0
    return result_ap, f1_score

                
                
############################# Main Code #############################


image_paths = []
for root, dirs, files in os.walk(args.img_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.JPG')):
            image_paths.append(os.path.join(root, file))


nms_thresholds  = [x / 100 for x in range(30, 96, 5)]
confs = [x / 100 for x in range(30, 96, 5)]

predictions = predict_boxes_for_image(image_paths, conf=0.001, imgsz=1280, max_det=300)

best_f1, best_nms_threshold, best_conf, best_mAP_at_bestF1 = 0, 0, 0, 0
for conf in confs :
    print(f"Evaluating for  conf={conf}...")

    result_ap, result_f1 = nms_filter_results(predictions, conf, nms_active=False, nms_threshold=0.5)
    res_mAP = result_ap["map"].item() # round(result_ap["map"].item(), 4)

    print(f"conf: {conf} | mAP: {res_mAP} | F1: {result_f1} | best_F1: {best_f1}")

    if result_f1 > best_f1:
        best_f1 = result_f1
        best_conf = conf
        best_mAP_at_bestF1 = res_mAP

best_f1 = 0
for nms_threshold in nms_thresholds :
    print(f"Evaluating for nms_threshold={nms_threshold}")

    result_ap, result_f1 = nms_filter_results(predictions, best_conf, nms_active=True, nms_threshold=nms_threshold)
    res_mAP = result_ap["map"].item() # round(result_ap["map"].item(), 4)

    print(f"iou_thresh: {nms_threshold} | conf: {best_conf} | mAP: {res_mAP} | F1: {result_f1} | best_F1: {best_f1}")

    if result_f1 > best_f1:
        best_f1 = result_f1
        best_nms_threshold = nms_threshold
        best_mAP_at_bestF1 = res_mAP

print(f"Best F1: {best_f1}, Best NMS Threshold: {best_nms_threshold}, Best Conf: {best_conf}")

with open(args.results_log, "a") as f:
    f.write(f"{args.target_dataset_name};{best_nms_threshold};{best_conf};{best_f1};{best_mAP_at_bestF1};{args.text}\n")