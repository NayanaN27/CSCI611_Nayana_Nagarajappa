import os
import random
import shutil

SRC = "tt100k_yolo"
DST = "tt100k_yolo_small"

random.seed(42)

train_limit = 2000
val_limit = 400

for split, limit in [("train", train_limit), ("test", val_limit)]:
    src_img_dir = os.path.join(SRC, "images", split)
    src_lbl_dir = os.path.join(SRC, "labels", split)

    dst_img_dir = os.path.join(DST, "images", "train" if split == "train" else "val")
    dst_lbl_dir = os.path.join(DST, "labels", "train" if split == "train" else "val")

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(src_img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_files.sort()
    random.shuffle(image_files)
    selected = image_files[:limit]

    for img_name in selected:
        stem = os.path.splitext(img_name)[0]
        lbl_name = stem + ".txt"

        src_img = os.path.join(src_img_dir, img_name)
        src_lbl = os.path.join(src_lbl_dir, lbl_name)

        dst_img = os.path.join(dst_img_dir, img_name)
        dst_lbl = os.path.join(dst_lbl_dir, lbl_name)

        if os.path.exists(src_img):
            try:
                os.symlink(os.path.abspath(src_img), dst_img)
            except Exception:
                shutil.copy2(src_img, dst_img)

        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        else:
            open(dst_lbl, "w").close()

yaml_path = os.path.join(DST, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"""path: {os.path.abspath(DST)}
train: images/train
val: images/val

names:
""")
    with open(os.path.join(SRC, "data.yaml"), "r") as src_yaml:
        lines = src_yaml.readlines()

    copy = False
    for line in lines:
        if line.strip() == "names:":
            copy = True
        if copy:
            f.write(line)

print("Created small subset at:", DST)
