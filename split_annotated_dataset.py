import os
import random
import shutil

# establish ratios
train_ratio = 0.7
val_ratio = 0.2

images_dir = r"C:\Users\samay\OneDrive\Documents\GitHub\ML4Traffic\images"
labels_dir = r"C:\Users\samay\OneDrive\Documents\GitHub\ML4Traffic\labels"

images = [f for f in os.listdir(images_dir) if f.endswith(".png")]
random.shuffle(images)

n = len(images)
train_end = int(n * train_ratio)
val_end = int(n * (train_ratio + val_ratio))

splits = {
       "train": images[:train_end],
       "val": images[train_end:val_end],
       "test": images[val_end:]
}

output_dir = r"C:\Users\samay\OneDrive\Documents\GitHub\ML4Traffic\split_annotated_dataset"

for split, files in splits.items():
    images_out = os.path.join(output_dir, "images", split)
    labels_out = os.path.join(output_dir, "labels", split)
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    for img in files:
        og_img = os.path.join(images_dir, img)
        final_img = os.path.join(images_out, img)
        shutil.copy2(og_img, final_img)

        og_lbl = os.path.join(labels_dir, img.replace(".png", ".txt"))
        final_lbl = os.path.join(labels_out, img.replace(".png", ".txt"))
        if os.path.exists(og_lbl):
            shutil.copy2(og_lbl, final_lbl)
        else:
            print("WARNING: missing label for", img)

print("Dataset prepared.")
