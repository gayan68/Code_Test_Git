from datasets import load_dataset, concatenate_datasets
import os


folders =["svea_hovratt_lines",
          "krigshovrattens_dombocker_lines",
            "bergskollegium_relationer_och_skrivelser_lines",
            "frihetstidens_utskottshandlingar_lines",
            "trolldomskommissionen_lines",
            "bergmastaren_i_nora_htr_lines",
            "carl_fredrik_pahlmans_resejournaler_lines",
            "gota_hovratt_lines",
            "alvsborgs_losen_lines",
            "jonkopings_radhusratt_och_magistrat_lines"
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
ds7  = load_dataset(
    f"Riksarkivet/{folders[6]}",
    cache_dir=f"{root}/dataset_cache"
)['train']
ds8  = load_dataset(
    f"Riksarkivet/{folders[7]}",
    cache_dir=f"{root}/dataset_cache"
)['train']
ds9  = load_dataset(
    f"Riksarkivet/{folders[8]}",
    cache_dir=f"{root}/dataset_cache"
)['train']
ds10  = load_dataset(
    f"Riksarkivet/{folders[9]}",
    cache_dir=f"{root}/dataset_cache"
)['train']

## Combine datasets
combined = concatenate_datasets([ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8, ds9, ds10])

print(f"Total examples in full dataset: {len(combined)}")
print(combined[0])

# save_dir = f"{root}/Riksarkivet/train/finetune_combined_10_datasets"
# os.makedirs(save_dir, exist_ok=True)
# combined.save_to_disk(save_dir)


# Use train_test_split with train_size=0.9
split = combined.train_test_split(test_size=0.05, seed=42)

train_dataset = split['train']
val_dataset = split['test']

print(f"Train examples: {len(train_dataset)}")
print(f"Validation examples: {len(val_dataset)}")

folder = "finetune_combined_10_datasets"
train_dir = f"{root}/Riksarkivet/train/{folder}"
val_dir = f"{root}/Riksarkivet/val/{folder}"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

train_dataset.save_to_disk(train_dir)
val_dataset.save_to_disk(val_dir)
