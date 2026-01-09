import glob
import os
import numpy as np
import xml.etree.ElementTree as ET


def extract_lines(page_id,  xml_root, save_dir_txt):
    gt_path = f"{xml_root}/{page_id}.xml"
    
    # Load XML
    xml_file = gt_path
    ns = {"pc": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
    
    tree = ET.parse(xml_file)
    root = tree.getroot()

    xyxy=[]
    yolo_boxes = []
    idx = 0
    for line in root.findall(".//pc:TextLine", ns):
        line_id = line.get("id")
        coords = line.find("pc:Coords", ns)
        
        if coords is not None:
            # Capture TextEquiv
            text_equiv_elem = line.find("pc:TextEquiv/pc:Unicode", ns)
            text = text_equiv_elem.text if (text_equiv_elem is not None and text_equiv_elem.text is not None) else ""
            save_path_txt = os.path.join(save_dir_txt, f"{page_id}_{line_id}.txt")

            if text.strip():
                with open(save_path_txt, "w", encoding="utf-8") as f:
                    f.write(text + "\n")
    
            idx += 1

def get_file_names(xml_root):
    xml_files = glob.glob(os.path.join(xml_root, "*.xml"))
    xml_names = [os.path.splitext(os.path.basename(f))[0] for f in xml_files]
    return xml_names            

root = "/home/x_gapat/PROJECTS"

split = "train"
root = f"{root}/DATASETS/NorHandv3_mini_v3"
save_dir = f"{root}/DATASETS/NorHandv3_mini_v3"

xml_root = f"{root}/{split}/gt"
save_dir_txt = f"{save_dir}/{split}/line_splits/gt_text"
print(save_dir_txt)

os.makedirs(save_dir_txt, exist_ok=True)

file_names = get_file_names(xml_root)

for page_id in file_names:
    extract_lines(page_id, xml_root, save_dir_txt)

print("Train split done.")

split = "val"
root = f"{root}/DATASETS/NorHandv3_mini_v3"
save_dir = f"{root}/DATASETS/NorHandv3_mini_v3"

xml_root = f"{root}/{split}/gt"
save_dir_txt = f"{save_dir}/{split}/line_splits/gt_text"
print(save_dir_txt)

os.makedirs(save_dir_txt, exist_ok=True)

file_names = get_file_names(xml_root)

for page_id in file_names:
    extract_lines(page_id, xml_root, save_dir_txt)
print("Val split done.")