import os

# img = cv2.imread('month1/week4/drone_dataset/')
# img = cv2.imread('./drone_dataset/images/cat.jpg')

# with os.scandir('month1/week4/drone_dataset/train/labels') as entries:
#     for entry in entries:
#         if entry.is_file():
#             print(f"File found: {entry.name} at {entry.path}")

train_labels_count = sum(1 for entry in os.scandir("month1/week4/drone_dataset/train/labels") if entry.is_file())
valid_labels_count = sum(1 for entry in os.scandir("month1/week4/drone_dataset/valid/labels") if entry.is_file())
test_labels_count = sum(1 for entry in os.scandir("month1/week4/drone_dataset/test/labels") if entry.is_file())
train_images_count = sum(1 for entry in os.scandir("month1/week4/drone_dataset/train/images") if entry.is_file())
valid_images_count = sum(1 for entry in os.scandir("month1/week4/drone_dataset/valid/images") if entry.is_file())
test_images_count = sum(1 for entry in os.scandir("month1/week4/drone_dataset/test/images") if entry.is_file())


for entry in os.scandir("month1/week4/drone_dataset/train/labels"):
    if entry.is_file():
        with open(entry.path, "r") as f:
            lines = f.readlines()
            print(lines)
        # 'lines' is now a list of strings, where each string is one bounding box!
