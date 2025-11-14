from datasets import load_dataset

ds = load_dataset(
    "Riksarkivet/jonkopings_radhusratt_och_magistrat_lines",
    cache_dir="/home/x_gapat/PROJECTS/dataset_cache"
)

print(ds) 

first_example = ds['train'][0]  # or ds['test'][0]
print(first_example)