import json
import os
import csv
import re
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
    # This removes characters like: @#$%^&*()_+={}[]|\\:";'<>?/~`
    caption = re.sub(r'[^\w\s.,!?-]', ' ', caption)
    
    # Optional: Remove commas entirely (uncomment if you want to remove all commas)
    # caption = caption.replace(',', ' ')
    
    # Optional: Remove all punctuation except periods (uncomment if needed)
    # caption = re.sub(r'[^\w\s.]', ' ', caption)
    
    # Optional: Remove ALL punctuation (uncomment if you want completely clean text)
    # caption = re.sub(r'[^\w\s]', ' ', caption)
    
    # Remove multiple consecutive spaces
    caption = re.sub(r'\s+', ' ', caption)
    
    # Remove leading and trailing whitespace
    caption = caption.strip()
    
    # Optional: Convert to lowercase (uncomment if needed)
    # caption = caption.lower()
    
    # Remove empty parentheses, brackets, etc.
    caption = re.sub(r'\(\s*\)', '', caption)
    caption = re.sub(r'\[\s*\]', '', caption)
    caption = re.sub(r'\{\s*\}', '', caption)
    
    # Final cleanup of any remaining multiple spaces
    caption = re.sub(r'\s+', ' ', caption).strip()
    
    return caption

def process_json_file(json_file, captions_path):
    """
    Process a single JSON file and extract image path and cleaned caption
    """
    json_file_path = os.path.join(captions_path, json_file)
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract image path and detailed caption
        image_path = data.get('full_path', '')
        raw_caption = data.get('detailed_caption', '')
        
        if image_path and raw_caption:
            # Clean the caption
            cleaned_caption = clean_caption(raw_caption)
            
            # Only return if cleaned caption is not empty
            if cleaned_caption:
                return {
                    'image_path': image_path,
                    'captions': cleaned_caption
                }
        return None
        
    except json.JSONDecodeError as e:
        print(f"Error reading {json_file}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return None

def generate_csv_from_json_files(captions_path, output_csv, max_workers=None, sample_size=None):
    """
    Read all JSON files from the captions directory and create a CSV file
    with columns: image_path, captions using multithreading for faster processing
    """
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(captions_path) if f.endswith('.json')]
    
    # Limit sample size if specified
    if sample_size:
        json_files = json_files[:sample_size]
    
    # Prepare data for CSV
    csv_data = []
    
    # Determine optimal number of threads (default to CPU count)
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)
    
    print(f"Processing {len(json_files)} JSON files using {max_workers} threads...")
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_json_file, json_file, captions_path): json_file 
                         for json_file in json_files}
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_file), 
                          total=len(json_files), 
                          desc="Processing JSON files"):
            result = future.result()
            if result is not None:
                csv_data.append(result)
    
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
    
    # Show some sample cleaned captions for verification
    print("\nSample cleaned captions:")
    for i, row in enumerate(csv_data[:3]):
        print(f"{i+1}. {row['captions'][:100]}...")
    
    return output_csv

def test_caption_cleaning():
    """
    Test function to see how captions are cleaned
    """
    test_captions = [
        "This is a sample caption,,,, with multiple commas\n\nand newlines",
        "Another caption!!!! with excessive punctuation???",
        "Caption with @#$%^&*() special characters",
        "Normal caption with proper punctuation.",
        "\t\tCaption with tabs\t\tand    multiple   spaces   \n"
    ]
    
    print("Testing caption cleaning:")
    print("-" * 50)
    for i, caption in enumerate(test_captions, 1):
        cleaned = clean_caption(caption)
        print(f"Original {i}: {repr(caption)}")
        print(f"Cleaned {i}:  {repr(cleaned)}")
        print()

if __name__ == "__main__":



    import os
    os.makedirs('dataset_output/csv_files', exist_ok=True)

    # Define paths and output file name
    captions_path = '/data0/teja_works/sd3_dataset/data/extracted_dataset_captions'
    output_csv = 'dataset_output/csv_files/train.csv'
    
    # Uncomment the line below to test caption cleaning first
    # test_caption_cleaning()
    
    # Generate CSV with cleaned captions
    # You can specify sample_size to process only a subset for testing
    generate_csv_from_json_files(captions_path, output_csv, sample_size=5000)


    captions_path = '/data0/teja_works/sd3_dataset/data/extracted_dataset_captions'
    output_csv = 'dataset_output/csv_files/val.csv'
    generate_csv_from_json_files(captions_path, output_csv, sample_size=20)

    captions_path = '/data0/teja_works/sd3_dataset/data/extracted_dataset_captions'
    output_csv = 'dataset_output/csv_files/test.csv'
    generate_csv_from_json_files(captions_path, output_csv, sample_size=20)