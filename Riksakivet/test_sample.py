from datasets import load_from_disk

folder ="krigshovrattens_dombocker_lines"

path = f"/home/gayapath/PROJECTS/DATA_DGX2/Riksarkivet/val/{folder}"
ds = load_from_disk(path)

print(ds)

# print first sample
print(ds[0])