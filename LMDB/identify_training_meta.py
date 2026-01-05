import lmdb
import io
from PIL import Image


lmdb_root = f"../../DATASETS/NorHandv3_mini"

train_env = lmdb.open(f"{lmdb_root}/train/lmdb", readonly=True, lock=False)
val_env = lmdb.open(f"{lmdb_root}/valid/lmdb", readonly=True, lock=False)
test_env = lmdb.open(f"{lmdb_root}/test/lmdb", readonly=True, lock=False)

unique_characters = set()
max_img_width = 0
max_text_length = 0

keys = []
for env in [train_env, val_env, test_env]:
    with env.begin() as txn:
        cursor = txn.cursor()  # create a cursor to iterate

        for key, value in cursor:
            key_str = key.decode()  # convert bytes -> string
            if key_str.startswith("label-"):
                # print(key_str)
                label = txn.get(key_str.encode()).decode()
                unique_characters.update(set(label))
                print(label)
                if max_text_length < len(label):
                    max_text_length = len(label)

            if key_str.startswith("image-"):
                img_data = txn.get(key_str.encode())
                image = Image.open(io.BytesIO(img_data)).convert("L")  # Convert to grayscale
                w, h = image.size

                if max_img_width < w:
                    max_img_width = w


print(unique_characters)
print(f"Max image line width: {max_img_width}")
print(f"Max transcription length: {max_text_length}")
save_path = f"{lmdb_root}/identify_training_meta_lmdb.txt"
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(f"Max image line width: {max_img_width}\n")
    f.write(f"Max transcription length: {max_text_length}\n")

unique_char_path = f"{lmdb_root}/charset.txt"
with open(unique_char_path, 'w', encoding='utf-8') as f:
    for ch in sorted(unique_characters):
        f.write(ch + "\n")