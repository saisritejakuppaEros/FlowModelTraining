from datasets import load_dataset
import os
from PIL import Image
import pandas as pd

# Load dataset
ds = load_dataset("jpawan33/kag100-image-captioning-dataset", split="train")

# Output folders
output_dir = "kag100_images"
os.makedirs(output_dir +'/images', exist_ok=True)

# Store mapping for CSV
image_caption_pairs = []

from tqdm import tqdm

# Iterate and save
for i, row in tqdm(enumerate(ds),total=len(ds)):
    image = row["image"]  # Dataset returns PIL.Image
    caption = row["text"]

    # Save image
    filename = f"image_{i:05d}.jpg"
    filepath = os.path.join(output_dir,'images' ,filename)
    image.save(filepath)

    # Store caption mapping
    image_caption_pairs.append({"filename": filename, "caption": caption})

# Save captions as CSV
df = pd.DataFrame(image_caption_pairs)
df.to_csv(os.path.join(output_dir, "captions.csv"), index=False)

print(f"Saved {len(image_caption_pairs)} images and captions to '{output_dir}'")
