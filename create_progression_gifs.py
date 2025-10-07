#!/usr/bin/env python3
"""
Create MP4 videos showing the progression of generated images across epochs.
For each unique prompt, creates an MP4 video with original image on the left and generated images on the right.
"""

import os
import glob
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import time
import cv2

def get_unique_prompts_from_epoch1(samples_dir):
    """Extract the 4 unique prompts from epoch 1 caption files."""
    caption_files = glob.glob(os.path.join(samples_dir, "epoch_001_*_val_caption.txt"))
    caption_files.sort()
    
    prompts = []
    for caption_file in caption_files[:4]:  # Get first 4 unique prompts
        with open(caption_file, 'r') as f:
            prompt = f.read().strip()
            prompts.append(prompt)
    
    return prompts

def extract_prompt_name_from_filename(filename):
    """Extract the prompt name from the filename."""
    # Extract the part between 'val_generated_' and '.png'
    match = re.search(r'val_generated_(.+)\.png', filename)
    if match:
        return match.group(1)
    return None

def get_epoch_number(filename):
    """Extract epoch number from filename."""
    match = re.search(r'epoch_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def create_progression_video(prompt_name, samples_dir, output_dir):
    """Create an MP4 video showing progression for a specific prompt."""
    
    print(f"  📁 Looking for files with prompt: {prompt_name[:50]}...")
    
    # Find all files for this prompt
    pattern = f"*_val_generated_{prompt_name}.png"
    generated_files = glob.glob(os.path.join(samples_dir, pattern))
    
    pattern = f"*_val_original_{prompt_name}.png"
    original_files = glob.glob(os.path.join(samples_dir, pattern))
    
    print(f"  📊 Found {len(generated_files)} generated files, {len(original_files)} original files")
    
    if not generated_files or not original_files:
        print(f"❌ No files found for prompt: {prompt_name}")
        return
    
    # Sort by epoch number
    generated_files.sort(key=get_epoch_number)
    original_files.sort(key=get_epoch_number)
    
    # # Limit to first 50 epochs to avoid huge GIFs
    # if len(generated_files) > 50:
    #     generated_files = generated_files[:50]
    #     print(f"  ⚠️  Limited to first 50 epochs for performance")
    
    print(f"  🖼️  Processing {len(generated_files)} frames...")
    
    # Get the original image (should be the same across all epochs)
    original_img = Image.open(original_files[0])
    
    # Create frames for the GIF
    frames = []
    
    # Add progress bar for frame creation
    for i, gen_file in enumerate(tqdm(generated_files, desc="    Processing frames", unit="frame", position=1, leave=False)):
        try:
            generated_img = Image.open(gen_file)
            
            # Resize images to a smaller size for faster processing
            target_size = (256, 256)  # Smaller size for faster processing
            original_resized = original_img.resize(target_size, Image.Resampling.LANCZOS)
            generated_resized = generated_img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Create a side-by-side image
            combined_width = target_size[0] * 2
            combined_height = target_size[1] + 40  # Extra space for text
            
            combined_img = Image.new('RGB', (combined_width, combined_height), 'white')
            
            # Paste original on the left
            combined_img.paste(original_resized, (0, 20))
            
            # Paste generated on the right
            combined_img.paste(generated_resized, (target_size[0], 20))
            
            # Add text labels (simplified for speed)
            draw = ImageDraw.Draw(combined_img)
            
            # Use default font for speed
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            # Add labels
            draw.text((5, 5), "Original", fill='black', font=font)
            draw.text((target_size[0] + 5, 5), f"Epoch {get_epoch_number(gen_file)}", fill='black', font=font)
            
            # Convert PIL image to numpy array for OpenCV
            frame_array = np.array(combined_img)
            frames.append(frame_array)
            
        except Exception as e:
            print(f"    ⚠️  Error processing {gen_file}: {e}")
            continue
    
    if not frames:
        print(f"❌ No valid frames created for prompt: {prompt_name}")
        return
    
    print(f"  💾 Saving MP4 video with {len(frames)} frames...")
    
    # Create MP4 video
    output_filename = f"progression_{prompt_name.replace(' ', '_').replace(',', '').replace('.', '')}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    start_time = time.time()
    
    try:
        # Get frame dimensions
        frame_height, frame_width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 2.0  # 2 frames per second for smooth viewing
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Write frames with progress bar
        for frame in tqdm(frames, desc="    Writing video", unit="frame", position=1, leave=False):
            # Convert PIL image to OpenCV format (BGR)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        # Release video writer
        video_writer.release()
        
        elapsed_time = time.time() - start_time
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
        
        print(f"✅ Created MP4: {output_path}")
        print(f"   📊 Stats: {len(frames)} frames, {file_size:.1f}MB, {elapsed_time:.1f}s")
        
    except Exception as e:
        print(f"❌ Error saving MP4: {e}")
        return

def main():
    # Set up paths
    samples_dir = "/data0/teja_works/diffusion_training/dataset_preparation/micro_diffusion/FlowModelTraining/train/output/inference_samples"
    output_dir = "/data0/teja_works/diffusion_training/dataset_preparation/micro_diffusion/FlowModelTraining/train/output/video_progressions"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the 4 unique prompts from epoch 1
    prompts = get_unique_prompts_from_epoch1(samples_dir)
    
    print("Found unique prompts:")
    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. {prompt}")
    
    print(f"\nStarting MP4 video creation for {len(prompts)} prompts...")
    
    # Create MP4 video for each prompt with progress bar
    for i, prompt in enumerate(tqdm(prompts, desc="Creating MP4s", unit="prompt", position=0), 1):
        print(f"\n[{i}/{len(prompts)}] Creating MP4 for: {prompt[:50]}...")
        
        # Extract the prompt name from the first generated file for this prompt
        pattern = f"*_val_generated_*.png"
        all_generated = glob.glob(os.path.join(samples_dir, pattern))
        
        # Find the prompt name that corresponds to this text prompt
        prompt_name = None
        for gen_file in all_generated:
            if "Bullock_Creek" in gen_file and "Bullock Creek" in prompt:
                prompt_name = extract_prompt_name_from_filename(gen_file)
                break
            elif "rainbow_rock" in gen_file and "rainbow rock" in prompt:
                prompt_name = extract_prompt_name_from_filename(gen_file)
                break
            elif "Furnished_dining" in gen_file and "Furnished dining" in prompt:
                prompt_name = extract_prompt_name_from_filename(gen_file)
                break
            elif "Pukka_Tea" in gen_file and "Pukka Tea" in prompt:
                prompt_name = extract_prompt_name_from_filename(gen_file)
                break
        
        if prompt_name:
            create_progression_video(prompt_name, samples_dir, output_dir)
        else:
            print(f"❌ Could not find matching prompt name for: {prompt}")
    
    print(f"\n🎉 All MP4 videos completed! Check the output directory: {output_dir}")

if __name__ == "__main__":
    main()
