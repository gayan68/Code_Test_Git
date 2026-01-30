import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import matplotlib.patches as patches
import glob
import os
import numpy as np
import shutil
import random


def compute_intersection(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection
    ix_min = max(x1_min, x2_min)
    iy_min = max(y1_min, y2_min)
    ix_max = min(x1_max, x2_max)
    iy_max = min(y1_max, y2_max)

    iw = max(0, ix_max - ix_min)
    ih = max(0, iy_max - iy_min)
    inter = iw * ih

    # Union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - inter

    impact1 = inter/area1
    impact2 = inter/area2

    impact = max(impact1, impact2)

    return impact

def read_xml_page_gt(path_file):
    xml_root = ET.parse(path_file).getroot()

    # Remove namespace if present
    # strip_namespace(xml_root)

    bb_xyxy = []
    labels = []

    for node in xml_root.iter():

        if "TextLine" in node.tag:
            for node_c in node.iter():
                if "Coords" in node_c.tag:
                    polygon_pts = node_c.attrib["points"]

                    polygon_pts = polygon_pts.split(sep=" ")

                    nb_points = len(polygon_pts)

                    x_min = 999999
                    x_max = 0
                    y_min = 999999
                    y_max = 0

                    # Draw line polygon
                    print(polygon_pts)
                    for i in range(nb_points):
                        coordinate_p_i = polygon_pts[i]  # str 380,282
                        coordinate_p_i = coordinate_p_i.split(sep=",")

                        # Min max x
                        if int(coordinate_p_i[0]) < x_min:
                            x_min = int(coordinate_p_i[0])
                        if int(coordinate_p_i[0]) > x_max:
                            x_max = int(coordinate_p_i[0])
                        # Min max y
                        if int(coordinate_p_i[1]) < y_min:
                            y_min = int(coordinate_p_i[1])
                        if int(coordinate_p_i[1]) > y_max:
                            y_max = int(coordinate_p_i[1])

                    bb_xyxy.append([x_min, y_min, x_max, y_max])

                if "TextEquiv" in node_c.tag:
                    #print(node_c)
                    for node_txt in node_c.iter():
                        if "Unicode" in node_txt.tag:
                            txt_gt_line = node_txt.text
                            if txt_gt_line is None:
                                txt_gt_line = ""
                            labels.append(txt_gt_line)

 #   return bb_xyxy, labels
    
def read_xml_page_gt2(page_id, xml_root, img_root=None):
    gt_path = f"{xml_root}/{page_id}.xml"
    
    # Load XML
    xml_file = gt_path
    ns = {"pc": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
        
    # Load image
    if img_root is not None:
        image_path = f"{img_root}/{page_id}.jpg"
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
    else:
        page = root.find("pc:Page", ns)
        h = int(page.get("imageHeight"))
        w = int(page.get("imageWidth"))
    
    idx = 0
    bb_xyxy, labels = [], []
    for line in root.findall(".//pc:TextLine", ns):
        line_id = line.get("id")
        coords = line.find("pc:Coords", ns)
        
        if coords is not None:
            pts = coords.get("points")
            polygon = []
            for p in pts.split():
                x, y = p.split(",")
                polygon.append((int(x), int(y)))
    
            pts = np.array(polygon)
            x_min = int(pts[:, 0].min())
            y_min = int(pts[:, 1].min())
            x_max = int(pts[:, 0].max())
            y_max = int(pts[:, 1].max())

            #To avoid out-of-range errors:
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            bb_xyxy.append([x_min, y_min, x_max, y_max])

            # Capture TextEquiv
            text_equiv_elem = line.find("pc:TextEquiv/pc:Unicode", ns)
            text = text_equiv_elem.text if (text_equiv_elem is not None and text_equiv_elem.text is not None) else ""
            labels.append(text)

    return bb_xyxy, labels

def read_xyxy(bb_path):
    boxes = []
    with open(bb_path, "r") as file:
        for line in file:
            x_min, y_min, x_max, y_max = map(int, line.strip().split())
            boxes.append([x_min, y_min, x_max, y_max])
    return boxes

def match_find_label_then_save(pred_bbs, gt_bbs, labels, page_id, pred_img_lines, image_save_path, label_save_path, iou_thresh=0.8):

    for idx, pred_bb in enumerate(pred_bbs):
        line_id = f"{page_id}_{idx:02}.png"
        for gt_bb, label in zip(gt_bbs, labels):
            if compute_intersection(pred_bb, gt_bb) > iou_thresh:
                # save image line
                shutil.copy(f"{pred_img_lines}/{line_id}", f"{image_save_path}/{line_id}")
                # save GT Text
                with open(f"{label_save_path}/{page_id}_{idx:02}.txt", "w") as f:
                    f.write(label)



###########################################

split = "val"
split_mask = "masked_val"
model = "H077"

root = "/home/x_gapat/PROJECTS"
pred_img_lines = f"{root}/logs/Hi-SAM_Doc/sample_output/{model}/NorHandV3/{split_mask}/images"
gt_dir = f"{root}/DATASETS/NorHandv3/{split}/PAGE"
image_save_path = f"{root}/Croped_image_lines2/{model}/NorHandV3/{split}/images"
label_save_path = f"{root}/Croped_image_lines2/{model}/NorHandV3/{split}/gt_text"

os.makedirs(image_save_path, exist_ok=True)
os.makedirs(label_save_path, exist_ok=True)
print(gt_dir)
page_ids = glob.glob(gt_dir + '/*.xml', recursive=True)
page_ids = [os.path.splitext(os.path.basename(f))[0] for f in page_ids]

for page_id in ["fgsf001_4"]:
    print(page_id)
    boxes = read_xyxy(f"{pred_bb_dir}/{page_id}.txt")
    
    gt_bb_xyxy, labels = read_xml_page_gt2(page_id, gt_dir)
    match_find_label_then_save(boxes, gt_bb_xyxy, labels, page_id, pred_img_lines, image_save_path, label_save_path)
