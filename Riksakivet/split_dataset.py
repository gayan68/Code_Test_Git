from datasets import load_dataset

dataset  = load_dataset(
    "Riksarkivet/jonkopings_radhusratt_och_magistrat_lines",
    cache_dir="/home/x_gapat/PROJECTS/dataset_cache"
)

print(dataset)


full_dataset = dataset['train']

# Use train_test_split with train_size=0.9
split = full_dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = split['train']
val_dataset = split['test']

print(f"Train examples: {len(train_dataset)}")
print(f"Validation examples: {len(val_dataset)}")

train_dataset.save_to_disk("/home/x_gapat/PROJECTS/DATASETS/Riksarkivet/train")
val_dataset.save_to_disk("/home/x_gapat/PROJECTS/DATASETS/Riksarkivet/val")