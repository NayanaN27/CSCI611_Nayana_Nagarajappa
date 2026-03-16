import json
import os
import shutil
from PIL import Image

ROOT = "tt100k_2021"
ANN_FILE = os.path.join(ROOT, "annotations_all.json")
OUT_ROOT = "tt100k_yolo"

os.makedirs(OUT_ROOT, exist_ok=True)

with open(ANN_FILE, "r") as f:
    data = json.load(f)

classes = data["types"]
class_to_id = {name: i for i, name in enumerate(classes)}

splits = ["train", "test", "other"]
for split in splits:
    os.makedirs(os.path.join(OUT_ROOT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "labels", split), exist_ok=True)

imgs = data["imgs"]

for img_id, info in imgs.items():
    rel_path = info["path"]
    src_img = os.path.join(ROOT, rel_path)

    if not os.path.exists(src_img):
        continue

    split = rel_path.split("/")[0]
    filename = os.path.basename(rel_path)
    stem = os.path.splitext(filename)[0]

    try:
        with Image.open(src_img) as im:
            w, h = im.size
    except Exception:
        continue

    dst_img = os.path.join(OUT_ROOT, "images", split, filename)
    if not os.path.exists(dst_img):
        try:
            os.symlink(os.path.abspath(src_img), dst_img)
        except Exception:
            shutil.copy2(src_img, dst_img)

    label_path = os.path.join(OUT_ROOT, "labels", split, stem + ".txt")

    lines = []
    for obj in info.get("objects", []):
        cat = obj.get("category")
        bbox = obj.get("bbox", {})
        if cat not in class_to_id:
            continue

        xmin = float(bbox["xmin"])
        ymin = float(bbox["ymin"])
        xmax = float(bbox["xmax"])
        ymax = float(bbox["ymax"])

        x_center = ((xmin + xmax) / 2.0) / w
        y_center = ((ymin + ymax) / 2.0) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        if bw <= 0 or bh <= 0:
            continue

        class_id = class_to_id[cat]
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))

yaml_path = os.path.join(OUT_ROOT, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"""path: {os.path.abspath(OUT_ROOT)}
train: images/train
val: images/test

names:
""")
    for i, name in enumerate(classes):
        f.write(f"  {i}: {name}\n")

print("Done.")
print("YOLO dataset created at:", OUT_ROOT)
print("YAML file:", yaml_path)
print("Number of classes:", len(classes))
