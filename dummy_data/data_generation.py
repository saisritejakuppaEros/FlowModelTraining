#


import os
import random
from PIL import Image
import numpy as np  
import cv2
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def generate_dummy_image(path, size=(512, 512)):
    img = Image.new("RGB", size, color="white")
    img.save(path)
    # generate some random noise
    noise = np.random.randint(0, 255, size=(size[0], size[1], 3))

    cv2.imwrite(path, noise)

def generate_dummy_caption(path, caption):
    with open(path, "w") as f:
        f.write(caption + "\n")

def generate_dataset():
    base_dirs = {
        "train_images": "dataset/train/images",
        "val_images": "dataset/val/images",
        "test_images": "dataset/test/images",
        "train_captions": "dataset/train/captions",
        "val_captions": "dataset/val/captions",
        "test_captions": "dataset/test/captions"
    }

    num_samples = {
        "train": 10,
        "val": 5,
        "test": 3
    }

    captions = [
        "A cat sitting on a mat.",
        "A dog playing with a ball.",
        "A bird flying in the sky.",
        "A person riding a bicycle.",
        "A car parked on the street.",
        "A group of people having dinner.",
        "A child reading a book.",
        "A beautiful sunset over the mountains.",
        "A boat sailing on the lake.",
        "A man walking his dog."
    ]

    # Create directories
    for key, path in base_dirs.items():
        ensure_dir(path)

    # Generate images and captions
    for split in ["train", "val", "test"]:
        img_dir = base_dirs[f"{split}_images"]
        cap_dir = base_dirs[f"{split}_captions"]
        n = num_samples[split]
        for i in range(n):
            img_path = os.path.join(img_dir, f"img_{i:03d}.jpg")
            cap_path = os.path.join(cap_dir, f"img_{i:03d}.txt")
            generate_dummy_image(img_path)
            caption = random.choice(captions)
            generate_dummy_caption(cap_path, caption)

if __name__ == "__main__":
    generate_dataset()
    print("Dummy dataset generated in 'dataset/' directory.")
