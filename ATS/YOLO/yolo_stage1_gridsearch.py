import matplotlib.pyplot as plt
import cv2
from ultralytics.models.yolo.detect.val import DetectionValidator
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import xml.etree.ElementTree as ET
from torchvision.ops import nms
import os
import argparse
import torch
import numpy as np
from ultralytics import YOLO
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

model = YOLO(weights)
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

def nms_filter_results(image_paths, conf, iou, nms_active:bool=False, nms_threshold:float=0.5, imgsz=1280, max_det=300):
    print(f"Displaying results with conf={conf} and iou={iou}")
    pred_boxes_dir = os.path.join(args.save_boxes_root, model_name, args.target_dataset_name)
    os.makedirs(pred_boxes_dir, exist_ok=True)

    metric = MeanAveragePrecision(iou_type="bbox")
    f1_scores = []

    for img_path in image_paths:
        result = model.predict(img_path, 
                               conf=conf, 
                               iou=iou, 
                               imgsz=imgsz,
                               max_det=max_det,
                               verbose=False,
                               device=0)
        r = result[0]
        if r.boxes is None or len(r.boxes) == 0:
            boxes = np.zeros((0, 6), dtype=np.float32)
            confidences = np.zeros((0,), dtype=np.float32)
            nms_active = False  # No boxes to apply NMS on
        else:
            boxes = r.boxes.data.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()



        if nms_active:
            boxes_t = result[0].boxes.data[:, :4].cpu()
            scores_t = result[0].boxes.conf.cpu()
            nms_indices = nms(boxes_t, scores_t, nms_threshold).cpu().numpy()

            n_all_boxes = len(boxes)
            n_nms_boxes = len(nms_indices)

            print(f"NMS active, originally found {n_all_boxes} boxes, {n_nms_boxes} boxes remain after NMS.")

            boxes = boxes[nms_indices]
            confidences = confidences[nms_indices]
        
        image = cv2.imread(img_path)
        img_h, img_w = image.shape[:2]
        page_size = (img_w, img_h)

        img_id = os.path.basename(img_path).split('.')[0]
        gt_boxes = get_GT_Boxes_onePage(args.gt_xml, img_id)
        pred_boxes = boxes[:, 0:4].tolist() if len(boxes) > 0 else []

        f1_scores.append(compute_f1(gt_boxes, pred_boxes, iou_thresh=0.5, page_size=page_size)["f1"])
        

        ### Construct input for mAP..
        targets = [{
            "boxes": torch.tensor(gt_boxes, dtype=torch.float),
            "labels": torch.tensor([0] * len(gt_boxes))
        }]
    
        if len(boxes) == 0:
            preds = [{
                "boxes": torch.empty((0, 4), dtype=torch.float),   # empty tensor
                "scores": torch.empty((0,), dtype=torch.float),
                "labels": torch.empty((0,), dtype=torch.int64)
            }]
        else:
            preds = [{
                "boxes": torch.tensor(boxes[:,0:4]),
                "scores": torch.tensor(confidences),  
                "labels": torch.tensor([0] * len(confidences))
            }]
        metric.update(preds, targets)


    result_ap = metric.compute()
    f1_score = np.mean(f1_scores, axis=0)
    return result_ap, f1_score
                
                
############################# Main Code #############################


image_paths = []
for root, dirs, files in os.walk(args.img_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.JPG')):
            image_paths.append(os.path.join(root, file))


ious = [x / 100 for x in range(30, 96, 5)]
confs = [x / 100 for x in range(30, 96, 5)]

best_f1, best_iou, best_conf, best_mAP_at_bestF1 = 0, 0, 0, 0
for iou in ious:
    for conf in confs:
        print(f"Evaluating for IoU={iou} and conf={conf}...")
        # yolo_args = dict(

        result_ap, result_f1 = nms_filter_results(image_paths, conf=conf, iou=iou, nms_active=False, nms_threshold=0.5, imgsz=1280, max_det=300)
        res_mAP = result_ap["map"].item() # round(result_ap["map"].item(), 4)

        print(f"iou_thresh: {iou}| conf: {conf} | mAP: {res_mAP} | F1: {result_f1} | best_F1: {best_f1}")

        if result_f1 > best_f1:
            best_f1 = result_f1
            best_iou = iou
            best_conf = conf
            best_mAP_at_bestF1 = res_mAP

print(f"Best F1: {best_f1}, Best IoU: {best_iou}, Best Conf: {best_conf}")

with open(args.results_log, "a") as f:
    f.write(f"{args.target_dataset_name};{best_iou};{best_conf};{best_f1};{best_mAP_at_bestF1};{args.text}\n")