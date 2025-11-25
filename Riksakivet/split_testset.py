from datasets import load_dataset
import os


folders =[  "carl_fredrik_pahlmans_resejournaler_lines",
            "gota_hovratt_lines",
            "alvsborgs_losen_lines",
            "jonkopings_radhusratt_och_magistrat_lines"
]

# folders =["eval_htr_out_of_domain_lines"]

root = "/home/gayapath/PROJECTS/DATA_DGX2"

for folder in folders:

    dataset  = load_dataset(
        f"Riksarkivet/{folder}",
        cache_dir=f"{root}/dataset_cache"
    )

    print(dataset)


    full_dataset = dataset['test']
    print(f"Total examples in full dataset: {len(full_dataset)}")
    print(full_dataset[0])

    # Use train_test_split with train_size=0.9
    # split = full_dataset.train_test_split(test_size=0.05, seed=42)

    # train_dataset = split['train']
    # val_dataset = split['test']

    # print(f"Train examples: {len(train_dataset)}")
    # print(f"Validation examples: {len(val_dataset)}")

    # train_dir = f"{root}/Riksarkivet/train/{folder}"
    # val_dir = f"{root}/Riksarkivet/val/{folder}"
    test_dir = f"{root}/Riksarkivet/test/{folder}"

    # os.makedirs(train_dir, exist_ok=True)
    # os.makedirs(val_dir, exist_ok=True)

    # train_dataset.save_to_disk(train_dir)
    # val_dataset.save_to_disk(val_dir)

    os.makedirs(test_dir, exist_ok=True)
    full_dataset.save_to_disk(test_dir)