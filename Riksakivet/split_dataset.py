from datasets import load_dataset
import os


folder ="goteborgs_poliskammare_fore_1900_lines"

dataset  = load_dataset(
    f"Riksarkivet/{folder}",
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

train_dir = f"/home/x_gapat/PROJECTS/DATASETS/Riksarkivet/train/{folder}"
val_dir = f"/home/x_gapat/PROJECTS/DATASETS/Riksarkivet/val/{folder}"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

train_dataset.save_to_disk(train_dir)
val_dataset.save_to_disk(val_dir)