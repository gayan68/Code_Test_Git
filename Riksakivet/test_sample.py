from datasets import load_from_disk

folder ="svea_hovratt_lines"

train_path = f"/home/x_gapat/PROJECTS/DATASETS/Riksarkivet/val/{folder}"
train_ds = load_from_disk(train_path)

print(train_ds)

# print first sample
print(train_ds[0])