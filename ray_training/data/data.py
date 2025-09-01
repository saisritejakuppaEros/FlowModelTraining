import os
import csv
import random
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Set up directories
data_dir = './data'
images_dir = os.path.join(data_dir, 'images')
captions_dir = os.path.join(data_dir, 'captions')
os.makedirs(images_dir, exist_ok=True)
os.makedirs(captions_dir, exist_ok=True)

# Dummy data parameters
num_samples = 10000
image_size = (512, 512)
csv_path = os.path.join(data_dir, 'dataset.csv')

def create_sample(i):
    img_filename = f'image_{i}.png'
    caption_filename = f'caption_{i}.txt'
    img_path = os.path.join(images_dir, img_filename)
    caption_path = os.path.join(captions_dir, caption_filename)
    aesthetic_score = round(random.uniform(1.0, 10.0), 2)

    # Create a dummy image
    img_array = np.random.randint(0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(img_path)

    # Create a dummy caption
    caption_text = f"This is a dummy caption for image {i}."
    with open(caption_path, 'w') as f:
        f.write(caption_text)

    # Return row for CSV
    return {
        'caption_path': caption_path,
        'image_path': img_path,
        'aesthetic_score': aesthetic_score
    }

# Use multiprocessing to speed up data generation with tqdm progress bar
rows = []
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(create_sample, i) for i in range(num_samples)]
    for future in tqdm(as_completed(futures), total=num_samples, desc="Generating samples"):
        rows.append(future.result())

# Write CSV file
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ['caption_path', 'image_path', 'aesthetic_score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"Dummy dataset created at {data_dir}")
