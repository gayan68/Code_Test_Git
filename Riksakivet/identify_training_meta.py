# this code is to identify characterset, max_img_line width, and max characters length 
# in training validation and text datasets.

from datasets import load_dataset
import os


# folders =["svea_hovratt_lines",
#           "krigshovrattens_dombocker_lines",
#             "bergskollegium_relationer_och_skrivelser_lines",
#             "frihetstidens_utskottshandlingar_lines",
#             "carl_fredrik_pahlmans_resejournaler_lines",
#             "trolldomskommissionen_lines",
#             "gota_hovratt_lines",
#             "bergmastaren_i_nora_htr_lines",
#             "alvsborgs_losen_lines"
# ]

folders =["eval_htr_out_of_domain_lines"]

for folder in folders:
    for split in ['train', 'val']:
        ds_path = f"/home/x_gapat/PROJECTS/DATASETS/Riksarkivet/{split}/{folder}"
        ds = load_from_disk(ds_path)
        print(f"Dataset: {folder} - Split: {split}")
        print(f"Number of examples: {len(ds)}")
