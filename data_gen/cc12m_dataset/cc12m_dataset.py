import json
import os
import csv
import re
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import tarfile
import glob
import shutil

def clean_caption(caption):
    """
    Clean caption by removing unwanted characters and formatting issues
    """
    if not caption or not isinstance(caption, str):
        return ""
    
    # Remove newlines and replace with spaces
    caption = caption.replace('\n', ' ').replace('\r', ' ')
    
    # Remove tabs and replace with spaces
    caption = caption.replace('\t', ' ')
    
    # Remove excessive punctuation (multiple consecutive punctuation marks)
    caption = re.sub(r'[,]{2,}', ',', caption)  # Multiple commas
    caption = re.sub(r'[.]{2,}', '.', caption)  # Multiple periods
    caption = re.sub(r'[!]{2,}', '!', caption)  # Multiple exclamation marks
    caption = re.sub(r'[?]{2,}', '?', caption)  # Multiple question marks
    
    # Remove special characters but keep basic punctuation
    caption = re.sub(r'[^\w\s.,!?-]', ' ', caption)
    
    # Remove multiple consecutive spaces
    caption = re.sub(r'\s+', ' ', caption)
    
    # Remove leading and trailing whitespace
    caption = caption.strip()
    
    # Final cleanup of any remaining multiple spaces
    caption = re.sub(r'\s+', ' ', caption).strip()
    
    return caption

def read_tar(path: str, path_out: str) -> list:
    """
    Extract and read tar files. Yield image and corresponding caption.
    """
    os.makedirs(path_out, exist_ok=False)
    with tarfile.open(path, 'r') as tar:
        tar.extractall(path_out)

    txts = sorted(glob.glob(os.path.join(path_out, '*txt')))
    print(f"Found {len(txts)} caption files in the tar file.")
    
    images_data = []

    for t in txts:
        try:
            with open(t, 'r') as ct:
                cap = ct.read()
            # assuming all files are in jpg
            img_path = t.replace('.txt', '.jpg')
            img = Image.open(img_path)
            images_data.append((img, cap))
        except Exception as e:
            print(e)
            
    print("Done reading the tar file")
    shutil.rmtree(path_out)
    
    return images_data

def save_images_and_csv(images_data, output_image_folder, output_csv):
    """
    Save images and captions in CSV format.
    """
    # Ensure output image folder exists
    os.makedirs(output_image_folder, exist_ok=True)

    csv_data = []
    for idx, (img, cap) in tqdm(enumerate(images_data), desc="Saving images", total=len(images_data)):
        img_filename = f"image_{idx:06d}.jpg"  # Use index for unique filename
        output_image_path = os.path.join(output_image_folder, img_filename)
        img.save(output_image_path)
        
        # Clean caption
        cleaned_caption = clean_caption(cap)
        if cleaned_caption:
            csv_data.append({'image_path': output_image_path, 'captions': cleaned_caption})
    
    # Ensure CSV directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Write to CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_path', 'captions']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write data
        for row in csv_data:
            writer.writerow(row)
    
    print(f"CSV file created: {output_csv}")
    print(f"Total entries: {len(csv_data)}")

def process_wds_dataset(wds_dir, output_image_folder, output_csv, num_proc=64):
    """
    Process CC12M dataset stored in WDS format and save images and captions into CSV.
    """
    tars = glob.glob(os.path.join(wds_dir, '*.tar'))  # Find all tar files in the provided directory
    print(f"Found {len(tars)} tar files in the dataset directory.")

    all_images_data = []

    # Process each tar file
    for tar_file in tqdm(tars, desc="Processing tar files"):
        temp_dir = os.path.join(wds_dir, 'temp')
        images_data = read_tar(tar_file, temp_dir)
        all_images_data.extend(images_data)

    # Save images and CSV
    save_images_and_csv(all_images_data, output_image_folder, output_csv)

if __name__ == "__main__":
    # Define paths
    wds_dir = '/data0/teja_works/diffusion_training/dataset_preparation/micro_diffusion/datadir/cc12m/wds'  # Path to the CC12M WDS dataset
    output_image_folder = 'dataset_output/images'  # Path to save images
    output_csv = 'dataset_output/csv_files/cc12m_train.csv'  # Path to save CSV

    # Process dataset and save images and captions
    process_wds_dataset(wds_dir, output_image_folder, output_csv)
