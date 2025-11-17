from datasets import load_from_disk

folder ="krigshovrattens_dombocker_lines"

train_path = f"/home/gayapath/PROJECTS/DATA_DGX2/Riksarkivet/val/{folder}"
train_ds = load_from_disk(train_path)

print(train_ds)

# print first sample
print(train_ds[0])