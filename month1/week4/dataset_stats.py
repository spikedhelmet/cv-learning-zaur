import os
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU — will use CPU")

# 1. Point it at your `drone_dataset/` directory.
# 2. Scan all label files in `train/`, `val/`, and `test/`.
# 3. Print a summary report:
#    - Total images per split (train/val/test).
#    - Total annotations (bounding boxes) per split.
#    - Class distribution: how many instances of each class across the entire dataset.
#    - Average number of objects per image.
#    - Average bounding box size (width × height in normalized coordinates).
# 4. Flag any potential issues:
#    - Images with no annotations (empty `.txt` files).
#    - Annotations with coordinates outside the valid range (< 0 or > 1).
#    - Extreme aspect ratio boxes (width/height ratio > 10 or < 0.1).

def get_folder_stats(path):
    total_bounding_boxes = 0
    empty_files = []
    class_counts = {}
    total_area = 0
    weird_boxes = []
    out_of_bound_boxes = []

    entry_count = test_images_count = sum(1 for entry in os.scandir(f"{path}/images") if entry.is_file())

    for entry in os.scandir(f"{path}/labels"):
        if entry.is_file():
            with open(entry.path, "r") as f:
                lines = f.readlines()
                total_bounding_boxes += len(lines)
                if(len(lines) == 0):
                    empty_files.append(entry)
                
                for line in lines:
                    parts = line.split()
                    class_id = parts[0]
                    if class_id not in class_counts:
                        class_counts[class_id] = 1
                    else:
                        class_counts[class_id] += 1

                    width = parts[3]
                    height = parts[4]
                    box_size = float(width) * float(height)
                    total_area += box_size

                    aspect_ratio = float(width) / float(height)
                    if aspect_ratio > 10 or aspect_ratio < 0.1:
                        weird_boxes.append(entry)

                    for part in parts[1:]:
                        if float(part) < 0.0 or float(part) > 1.0:
                            out_of_bound_boxes.append(entry)
                
    avg_box_size = total_area / total_bounding_boxes
    avg_objects_per_image = total_bounding_boxes / entry_count


    print("total boxes:", total_bounding_boxes)
    print("empty files:", len(empty_files))
    print("class counts:", class_counts)
    print("avg_box_size:",avg_box_size)
    print("avg_objects_per_image:",avg_objects_per_image)
    print("weird_boxes count:",len(weird_boxes))
    print("out of bound count:",len(out_of_bound_boxes))

# get_folder_stats("month1/week4/drone_dataset/train")
# get_folder_stats("month1/week4/drone_dataset/valid")
# get_folder_stats("month1/week4/drone_dataset/test")