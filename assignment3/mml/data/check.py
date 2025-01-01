import os
import json
import shutil

with open("annotations/train_caption.json", "r") as f:
    data = json.load(f)

for caption in data:
    
    img_file = os.path.join("images/train2014", "COCO_train2014_" + caption["image_id"].zfill(12) + ".jpg")
    if os.path.exists(img_file):
        continue
    else:
        img_file = os.path.join("images/val2014", "COCO_val2014_" + caption["image_id"].zfill(12) + ".jpg")
        if os.path.exists(img_file):
            new_img_file = os.path.join("images/train2014", "COCO_train2014_" + caption["image_id"].zfill(12) + ".jpg")
            shutil.move(img_file, new_img_file)
            print(f"Moved {img_file} to {new_img_file}")
        else:
            print(f"File not found: {img_file}")