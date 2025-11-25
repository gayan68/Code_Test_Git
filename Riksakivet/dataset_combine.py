from datasets import load_dataset, concatenate_datasets
import os


folders =["svea_hovratt_lines",
          "krigshovrattens_dombocker_lines",
            "bergskollegium_relationer_och_skrivelser_lines",
            "frihetstidens_utskottshandlingar_lines",
            "trolldomskommissionen_lines",
            "bergmastaren_i_nora_htr_lines"
]


root = "/home/gayapath/PROJECTS/DATA_DGX2"

ds1  = load_dataset(
    f"Riksarkivet/{folders[0]}",
    cache_dir=f"{root}/dataset_cache"
)['train']
ds2  = load_dataset(
    f"Riksarkivet/{folders[1]}",
    cache_dir=f"{root}/dataset_cache"
)['train']
ds3  = load_dataset(
    f"Riksarkivet/{folders[2]}",
    cache_dir=f"{root}/dataset_cache"
)['train']
ds4  = load_dataset(
    f"Riksarkivet/{folders[3]}",
    cache_dir=f"{root}/dataset_cache"
)['train']
ds5  = load_dataset(
    f"Riksarkivet/{folders[4]}",
    cache_dir=f"{root}/dataset_cache"
)['train']
ds6  = load_dataset(
    f"Riksarkivet/{folders[5]}",
    cache_dir=f"{root}/dataset_cache"
)['train']

## Combine datasets
combined = concatenate_datasets([ds1, ds2, ds3, ds4, ds5, ds6])

print(f"Total examples in full dataset: {len(combined)}")
print(combined[0])

save_dir = f"{root}/Riksarkivet/train/finetune_combined_6_datasets"
os.makedirs(save_dir, exist_ok=True)
combined.save_to_disk(save_dir)