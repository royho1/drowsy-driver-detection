import os
import sys

closed_eyes_path = "./kaggle_data/train/Closed_Eyes"
open_eyes_path = "./kaggle_data/train/Open_Eyes"

for path in (closed_eyes_path, open_eyes_path):
    if not os.path.isdir(path):
        print(
            f"Error: '{path}' not found.\n"
            "Download the MRL eye dataset from Kaggle and place images under:\n"
            "  ./kaggle_data/train/Closed_Eyes\n"
            "  ./kaggle_data/train/Open_Eyes\n"
            "See README.md for the dataset link."
        )
        sys.exit(1)

closed_eyes_images = sorted(
    f for f in os.listdir(closed_eyes_path)
    if not f.startswith(".")
)
open_eyes_images = sorted(
    f for f in os.listdir(open_eyes_path)
    if not f.startswith(".")
)

if len(closed_eyes_images) < 2 or len(open_eyes_images) < 2:
    print(
        "Error: need at least two images in each of Closed_Eyes and Open_Eyes "
        "to run the sample visualization."
    )
    sys.exit(1)

import cv2
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

print(f"Closed Eyes images: {len(closed_eyes_images)}")
print(f"Open Eyes images: {len(open_eyes_images)}")

fig, ax = plt.subplots(1, 4, figsize=(12, 4))

img1 = cv2.imread(os.path.join(closed_eyes_path, closed_eyes_images[0]))
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
ax[0].imshow(img1)
ax[0].set_title("Closed Eyes")
ax[0].axis("off")

img2 = cv2.imread(os.path.join(open_eyes_path, open_eyes_images[0]))
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
ax[1].imshow(img2)
ax[1].set_title("Open Eyes")
ax[1].axis("off")

img3 = cv2.imread(os.path.join(closed_eyes_path, closed_eyes_images[1]))
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
ax[2].imshow(img3)
ax[2].set_title("Closed Eyes")
ax[2].axis("off")

img4 = cv2.imread(os.path.join(open_eyes_path, open_eyes_images[1]))
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
ax[3].imshow(img4)
ax[3].set_title("Open Eyes")
ax[3].axis("off")

plt.show()
