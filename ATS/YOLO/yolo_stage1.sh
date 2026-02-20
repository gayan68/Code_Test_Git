#!/bin/bash
#SBATCH -o /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.out
#SBATCH -e /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.err
#SBATCH -t 3-00:00:00
#SBATCH -C thin
#SBATCH --gpus=1


# mamba init bash
# module load Mambaforge/23.3.1-1-hpc1-bdist
# mamba activate pytorch25

# Parameters
file=yolo_mAP_F1_at_IoU_stage1.py

# root=/home/x_gapat/PROJECTS
root=/home/gayapath/PROJECTS
main_script="${root}/Test/ATS/YOLO/${file}"


PATHLOG="${root}/logs/ATS/YOLO"
echo "path log :"
echo ${PATHLOG}

output_file="${PATHLOG}/YOLO_Preds/out.txt"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/Hi-SAM_Doc/

############################# Train on READ #############################
################# YOLOv8-L-READ-1
PRETRAINE_MODEL=YOLOv8-L-READ-1
CUDA_VISIBLE_DEVICES=4 python "$main_script" \
  --model_path "${PATHLOG}/saved_models/${PRETRAINE_MODEL}/weights/best.pt" \
  --data_yaml read2016.yaml \
  --results_log "${PATHLOG}/YOLO_Preds/YOLO_READ 2016.txt" \
  --img_path "${root}/DATASETS/READ_2016/test/Images" \
  --gt_xml "${root}/DATASETS/READ_2016/test/gt_xml" \
  --save_boxes_root "${PATHLOG}/YOLO_Preds_tmp" \
  --target_dataset_name "READ_2016" \
  --text TrainOn_READ2016-TestOn_READ2016 \
>> "$output_file"

CUDA_VISIBLE_DEVICES=4 python "$main_script" \
  --model_path "${PATHLOG}/saved_models/${PRETRAINE_MODEL}/weights/best.pt" \
  --data_yaml iam.yaml \
  --results_log "${PATHLOG}/YOLO_Preds/YOLO_IAM.txt" \
  --img_path "${root}/DATASETS/IAM/pages_cleaned/test/Images" \
  --gt_xml "${root}/DATASETS/IAM/pages_cleaned/test/gt_xml" \
  --save_boxes_root "${PATHLOG}/YOLO_Preds_tmp" \
  --target_dataset_name "IAM" \
  --text TrainOn_READ2016-TestOn_IAM \
>> "$output_file"

CUDA_VISIBLE_DEVICES=4 python "$main_script" \
  --model_path "${PATHLOG}/saved_models/${PRETRAINE_MODEL}/weights/best.pt" \
  --data_yaml norhand_v3_mini_v3.yaml \
  --results_log "${PATHLOG}/YOLO_Preds/YOLO_NorHandV3.txt" \
  --img_path "${root}/DATASETS/NorHandv3_mini_v3/test/Images" \
  --gt_xml "${root}/DATASETS/NorHandv3_mini_v3/test/gt_xml" \
  --save_boxes_root "${PATHLOG}/YOLO_Preds_tmp" \
  --target_dataset_name "NorHandv3_mini_v3" \
  --text TrainOn_READ2016-TestOn_NorHandv3_mini_v3 \
>> "$output_file"

################# YOLOv8-L-READ-2
PRETRAINE_MODEL=YOLOv8-L-READ-2
CUDA_VISIBLE_DEVICES=4 python "$main_script" \
  --model_path "${PATHLOG}/saved_models/${PRETRAINE_MODEL}/weights/best.pt" \
  --data_yaml read2016.yaml \
  --results_log "${PATHLOG}/YOLO_Preds/YOLO_READ 2016.txt" \
  --img_path "${root}/DATASETS/READ_2016/test/Images" \
  --gt_xml "${root}/DATASETS/READ_2016/test/gt_xml" \
  --save_boxes_root "${PATHLOG}/YOLO_Preds_tmp" \
  --target_dataset_name "READ_2016" \
  --text TrainOn_READ2016-TestOn_READ2016 \
>> "$output_file"

CUDA_VISIBLE_DEVICES=4 python "$main_script" \
  --model_path "${PATHLOG}/saved_models/${PRETRAINE_MODEL}/weights/best.pt" \
  --data_yaml iam.yaml \
  --results_log "${PATHLOG}/YOLO_Preds/YOLO_IAM.txt" \
  --img_path "${root}/DATASETS/IAM/pages_cleaned/test/Images" \
  --gt_xml "${root}/DATASETS/IAM/pages_cleaned/test/gt_xml" \
  --save_boxes_root "${PATHLOG}/YOLO_Preds_tmp" \
  --target_dataset_name "IAM" \
  --text TrainOn_READ2016-TestOn_IAM \
>> "$output_file"

CUDA_VISIBLE_DEVICES=4 python "$main_script" \
  --model_path "${PATHLOG}/saved_models/${PRETRAINE_MODEL}/weights/best.pt" \
  --data_yaml norhand_v3_mini_v3.yaml \
  --results_log "${PATHLOG}/YOLO_Preds/YOLO_NorHandV3.txt" \
  --img_path "${root}/DATASETS/NorHandv3_mini_v3/test/Images" \
  --gt_xml "${root}/DATASETS/NorHandv3_mini_v3/test/gt_xml" \
  --save_boxes_root "${PATHLOG}/YOLO_Preds_tmp" \
  --target_dataset_name "NorHandv3_mini_v3" \
  --text TrainOn_READ2016-TestOn_NorHandv3_mini_v3 \
>> "$output_file"

################# YOLOv8-L-READ-3
PRETRAINE_MODEL=YOLOv8-L-READ-3
CUDA_VISIBLE_DEVICES=4 python "$main_script" \
  --model_path "${PATHLOG}/saved_models/${PRETRAINE_MODEL}/weights/best.pt" \
  --data_yaml read2016.yaml \
  --results_log "${PATHLOG}/YOLO_Preds/YOLO_READ 2016.txt" \
  --img_path "${root}/DATASETS/READ_2016/test/Images" \
  --gt_xml "${root}/DATASETS/READ_2016/test/gt_xml" \
  --save_boxes_root "${PATHLOG}/YOLO_Preds_tmp" \
  --target_dataset_name "READ_2016" \
  --text TrainOn_READ2016-TestOn_READ2016 \
>> "$output_file"

CUDA_VISIBLE_DEVICES=4 python "$main_script" \
  --model_path "${PATHLOG}/saved_models/${PRETRAINE_MODEL}/weights/best.pt" \
  --data_yaml iam.yaml \
  --results_log "${PATHLOG}/YOLO_Preds/YOLO_IAM.txt" \
  --img_path "${root}/DATASETS/IAM/pages_cleaned/test/Images" \
  --gt_xml "${root}/DATASETS/IAM/pages_cleaned/test/gt_xml" \
  --save_boxes_root "${PATHLOG}/YOLO_Preds_tmp" \
  --target_dataset_name "IAM" \
  --text TrainOn_READ2016-TestOn_IAM \
>> "$output_file"

CUDA_VISIBLE_DEVICES=4 python "$main_script" \
  --model_path "${PATHLOG}/saved_models/${PRETRAINE_MODEL}/weights/best.pt" \
  --data_yaml norhand_v3_mini_v3.yaml \
  --results_log "${PATHLOG}/YOLO_Preds/YOLO_NorHandV3.txt" \
  --img_path "${root}/DATASETS/NorHandv3_mini_v3/test/Images" \
  --gt_xml "${root}/DATASETS/NorHandv3_mini_v3/test/gt_xml" \
  --save_boxes_root "${PATHLOG}/YOLO_Preds_tmp" \
  --target_dataset_name "NorHandv3_mini_v3" \
  --text TrainOn_READ2016-TestOn_NorHandv3_mini_v3 \
>> "$output_file"