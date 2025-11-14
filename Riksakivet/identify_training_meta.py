# this code is to identify characterset, max_img_line width, and max characters length 
# in training validation and text datasets.

from datasets import load_dataset, load_from_disk   
import os


items =[{"svea_hovratt_lines": ['train', 'val']},
          {"krigshovrattens_dombocker_lines": ['train', 'val']},
            {"bergskollegium_relationer_och_skrivelser_lines": ['train', 'val']},
            {"frihetstidens_utskottshandlingar_lines": ['train', 'val']},
            {"carl_fredrik_pahlmans_resejournaler_lines": ['train', 'val']},
            {"trolldomskommissionen_lines": ['train', 'val']},
            {"gota_hovratt_lines": ['train', 'val']},
            {"bergmastaren_i_nora_htr_lines": ['train', 'val']},
            {"alvsborgs_losen_lines": ['train', 'val']},
            {"eval_htr_out_of_domain_lines": ['test']}
]

items =[
            {"eval_htr_out_of_domain_lines": ['test']}
]


unique_characters = set()
max_img_width = 0
max_text_length = 0

for item in items:
    for folder_name, splits in item.items():
        for split in splits:
            ds_path = f"/home/x_gapat/PROJECTS/DATASETS/Riksarkivet/{split}/{folder_name}"
            ds = load_from_disk(ds_path)
            print(f"Dataset: {folder_name} - Split: {split}")
            print(f"Number of examples: {len(ds)}")

            for item in ds:
                w, h = item["image"].size
                transcription = item["transcription"]

                if max_img_width < w:
                    max_img_width = w
                
                if max_text_length < len(transcription):
                    max_text_length = len(transcription)

                unique_characters.update(set(transcription))

print("===================================")
print(f"Unique characters ({len(unique_characters)}): {''.join(sorted(unique_characters))}")
print(f"Max image line width: {max_img_width}") 
print(f"Max transcription length: {max_text_length}")

save_path = "/home/x_gapat/PROJECTS/DATASETS/Riksarkivet/identify_training_meta.txt"
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(f"Unique characters ({len(unique_characters)}): {''.join(sorted(unique_characters))}\n")
    f.write(f"Max image line width: {max_img_width}\n")
    f.write(f"Max transcription length: {max_text_length}\n")   
print(f"Training meta data saved to {save_path}")

# ----------------------------------------
# Write unique characters to a SEPARATE FILE
# ----------------------------------------

unique_char_path = save_path.replace(".txt", "_unique_chars.txt")

with open(unique_char_path, 'w', encoding='utf-8') as f:
    for ch in sorted(unique_characters):
        f.write(ch + "\n")

print(f"Unique characters saved to {unique_char_path}")