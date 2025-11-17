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

folders =["krigshovrattens_dombocker_lines"]

for folder in folders:

    dataset  = load_dataset(
        f"Riksarkivet/{folder}",
        cache_dir="/home/gayapath/PROJECTS/DATA_DGX2/dataset_cache"
    )

    print(dataset)


    full_dataset = dataset['train']
    print(f"Total examples in full dataset: {len(full_dataset)}")
    print(full_dataset[0])

    # Use train_test_split with train_size=0.9
    split = full_dataset.train_test_split(test_size=0.1, seed=42)

    train_dataset = split['train']
    val_dataset = split['test']

    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")

    train_dir = f"/home/gayapath/PROJECTS/DATA_DGX2/Riksarkivet/train/{folder}"
    val_dir = f"/home/gayapath/PROJECTS/DATA_DGX2/Riksarkivet/val/{folder}"
    # test_dir = f"/home/x_gapat/PROJECTS/DATASETS/Riksarkivet/test/{folder}"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_dataset.save_to_disk(train_dir)
    val_dataset.save_to_disk(val_dir)

    # os.makedirs(test_dir, exist_ok=True)
    # full_dataset.save_to_disk(test_dir)