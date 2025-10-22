import os
import glob
import shutil
import tarfile
import numpy as np
from PIL import Image, UnidentifiedImageError
from argparse import ArgumentParser, Namespace
from multiprocessing import Pool, current_process
from streaming.base import MDSWriter
from streaming.base.util import merge_index
from torchvision import transforms
from tqdm import tqdm
from typing import List, Generator, Tuple

DEFAULT_CONFIG = {
    'wds_dir': '/data0/teja_works/diffusion_training/dataset_preparation/micro_diffusion/datadir/cc12m/wds',  # Path to your WDS dataset directory
    'local_mds_dir': os.path.abspath('./mds'),  # Directory to store the mds shards (using absolute path)
    'max_image_size': 512,  # Max image size for resizing
    'min_image_size': 256,  # Min image size, images smaller than this are skipped
    'num_proc': 16,  # Number of processes for multiprocessing
    'shard_size': 256 * (2**20),  # Size limit for each shard (256MB)
    'max_retries': 3  # Maximum number of retries for failed tar processing
}


os.makedirs(DEFAULT_CONFIG['local_mds_dir'], exist_ok=True)


def current_process_index() -> int:
    p = current_process()
    return p._identity[0] - 1


def read_tar(path: str, process_idx: int) -> Generator[Tuple[Image.Image, str], None, None]:
    """Read a tar file and yield (image, caption) pairs.
    
    Args:
        path: Path to tar file
        process_idx: Process index for unique temp directory
    """
    # Create a unique temporary directory in /tmp
    temp_base = f"/tmp/cc12m_proc_{process_idx}"
    temp_dir = os.path.join(temp_base, f"tar_{os.path.basename(path).replace('.tar', '')}")
    
    try:
        os.makedirs(temp_dir, exist_ok=True)
        with tarfile.open(path, 'r') as tar:
            tar.extractall(temp_dir)

        txts = sorted(glob.glob(os.path.join(temp_dir, '*txt')))
        print(f"Found {len(txts)} images in tar file")
        
        for t in txts:
            try:
                img_path = t.replace('.txt', '.jpg')
                if not os.path.exists(img_path):
                    print(f"Warning: Missing image file for caption: {img_path}")
                    continue
                    
                with open(t, 'r') as ct:
                    cap = ct.read().strip()
                if not cap:
                    print(f"Warning: Empty caption in {t}")
                    continue
                    
                img = Image.open(img_path)
                yield img, cap
            except Exception as e:
                print(f"Error processing {t}: {e}")
                continue
                
    except Exception as e:
        print(f"Error extracting tar {path}: {e}")
        raise
        
    finally:
        print("Done reading the tar file")
        # Clean up temp files
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            if os.path.exists(temp_base) and not os.listdir(temp_base):
                shutil.rmtree(temp_base, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Failed to clean up temp directory {temp_dir}: {e}")


def write_tar(tars: List[str], config: dict) -> Tuple[bool, List[str]]:
    """Process a list of tar files and convert them to MDS format.
    
    Returns:
        Tuple[bool, List[str]]: (success status, list of failed tar files)
    """
    process_idx = current_process_index()
    failed_tars = []
    
    try:
        columns = {
            'width': 'int32',
            'height': 'int32',
            'jpg': 'jpeg',
            'caption': 'str'
        }
        
        # Create process-specific output directory using absolute path
        save_dir = os.path.join(config['local_mds_dir'], str(process_idx))
        os.makedirs(save_dir, exist_ok=True)
        
        writer = MDSWriter(
            out=save_dir,
            columns=columns,
            compression=None,
            size_limit=config['shard_size'],
            max_workers=64
        )
        
        downsize = transforms.Resize(
            config['max_image_size'],
            antialias=True,
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        
        # Create a unique temporary directory for this process
        temp_base = os.path.join(config['local_mds_dir'], 'temp')
        os.makedirs(temp_base, exist_ok=True)
        temp_dir = os.path.join(temp_base, f'wds_{process_idx}')
        
        total_processed = 0
        for tar_idx, tar in enumerate(tars):
            if not os.path.exists(tar):
                print(f"Process {process_idx}: Warning - Tar file {tar} does not exist, skipping...")
                failed_tars.append(tar)
                continue
            
            retry_count = 0
            success = False
            
            while not success and retry_count < config['max_retries']:
                try:
                    rejected, total = 0, 0
                    for img, cap in tqdm(read_tar(tar, process_idx), 
                                       desc=f"Process {process_idx} [{tar_idx + 1}/{len(tars)}]"):
                        try:
                            w, h = img.size
                            if min(w, h) > config['max_image_size']:
                                img = downsize(img)
                            if min(w, h) < config['min_image_size']:
                                rejected += 1
                                continue
                                
                            mds_sample = {
                                'jpg': img,
                                'caption': cap,
                                'width': w,
                                'height': h
                            }
                            writer.write(mds_sample)
                            total += 1
                            
                        except (UnidentifiedImageError, OSError) as e:
                            rejected += 1
                            continue
                            
                    print(f"Process {process_idx}: Completed tar {tar_idx + 1}/{len(tars)} - "
                          f"Accepted {total}, Rejected {rejected}")
                    success = True
                    total_processed += 1
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count >= config['max_retries']:
                        print(f"Process {process_idx}: Failed to process {tar} after {retry_count} attempts: {e}")
                        failed_tars.append(tar)
                    else:
                        print(f"Process {process_idx}: Retry {retry_count}/{config['max_retries']} for {tar}: {e}")
                        
                finally:
                    # Clean up temp directory after each tar file
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
        
        writer.finish()
        
        # Create a status file to indicate completion
        with open(os.path.join(save_dir, 'process_complete'), 'w') as f:
            f.write(f"Processed: {total_processed}/{len(tars)}\nFailed: {len(failed_tars)}")
        
        return len(failed_tars) == 0, failed_tars
        
    except Exception as e:
        print(f"Fatal error in process {process_idx}: {e}")
        return False, tars


def retry_failed_tars(config: dict, failed_tars_file: str) -> bool:
    """Retry processing failed tar files.
    
    Args:
        config: Configuration dictionary
        failed_tars_file: Path to file containing list of failed tars
        
    Returns:
        bool: True if all retries succeeded, False otherwise
    """
    if not os.path.exists(failed_tars_file):
        print(f"Error: Failed tars file not found: {failed_tars_file}")
        return False
        
    with open(failed_tars_file, 'r') as f:
        failed_tars = [line.strip() for line in f if line.strip()]
        
    if not failed_tars:
        print("No failed tars to retry")
        return True
        
    print(f"Retrying {len(failed_tars)} failed tar files...")
    
    # Use fewer processes for retry to reduce resource contention
    num_retry_proc = min(8, config['num_proc'], len(failed_tars))
    tars_split = np.array_split(failed_tars, num_retry_proc)
    
    all_failed_tars = []
    process_statuses = []
    
    with Pool(processes=num_retry_proc) as pool:
        with tqdm(total=len(failed_tars), desc="Retrying failed tars", unit="tar") as pbar:
            def update_progress(result):
                success, failed = result
                if not success:
                    all_failed_tars.extend(failed)
                pbar.update(len(failed) if not success else 1)
            
            results = []
            for ts in tars_split:
                result = pool.apply_async(write_tar, (list(ts), config), callback=update_progress)
                results.append(result)
            
            for i, result in enumerate(results):
                try:
                    success, failed_tars = result.get()
                    process_statuses.append(success)
                    if failed_tars:
                        all_failed_tars.extend(failed_tars)
                except Exception as e:
                    print(f"Retry process {i} failed with error: {e}")
                    process_statuses.append(False)
    
    if all_failed_tars:
        print(f"\nStill failed after retry: {len(all_failed_tars)} tar files")
        retry_failed_file = os.path.join(config['local_mds_dir'], 'retry_failed_tars.txt')
        with open(retry_failed_file, 'w') as f:
            f.write('\n'.join(all_failed_tars))
        print(f"Failed tars after retry saved to: {retry_failed_file}")
        return False
        
    return True

def main():
    config = DEFAULT_CONFIG
    print(f"Using WDS dir: {config['wds_dir']}")
    print(f"Output directory: {config['local_mds_dir']}")
    
    # Ensure output directory exists and is writable
    try:
        os.makedirs(config['local_mds_dir'], exist_ok=True)
        test_file = os.path.join(config['local_mds_dir'], 'write_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except Exception as e:
        print(f"Error: Cannot write to output directory {config['local_mds_dir']}: {e}")
        return
        
    # Get list of tar files
    tars = sorted(glob.glob(os.path.join(config['wds_dir'], '*tar')))[:200]
    if not tars:
        print(f"Error: No tar files found in {config['wds_dir']}")
        return
        
    print(f"Found {len(tars)} tar files in dataset path")
    
    # Clean up any existing temporary directories
    temp_dir = os.path.join(config['local_mds_dir'], 'temp')
    if os.path.exists(temp_dir):
        print(f"Cleaning up existing temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Split work among processes
    tars_split = np.array_split(tars, config['num_proc'])
    print(f"Splitting {len(tars)} tar files across {config['num_proc']} processes "
          f"({[len(ts) for ts in tars_split]} files per process)")
    
    all_failed_tars = []
    process_statuses = []
    
    with Pool(processes=config['num_proc']) as pool:
        with tqdm(total=len(tars), desc="Processing tar files", unit="tar") as pbar:
            def update_progress(result):
                success, failed = result
                if not success:
                    all_failed_tars.extend(failed)
                pbar.update(len(failed) if not success else 1)
            
            # Submit all tasks
            results = []
            for ts in tars_split:
                result = pool.apply_async(write_tar, (list(ts), config), callback=update_progress)
                results.append(result)
            
            # Wait for all tasks to complete
            for i, result in enumerate(results):
                try:
                    success, failed_tars = result.get()
                    process_statuses.append(success)
                    if failed_tars:
                        all_failed_tars.extend(failed_tars)
                except Exception as e:
                    print(f"Process {i} failed with error: {e}")
                    process_statuses.append(False)
    
    # Report results
    successful_processes = sum(1 for status in process_statuses if status)
    print(f"\nProcess completion summary:")
    print(f"- Successful processes: {successful_processes}/{config['num_proc']}")
    print(f"- Failed tar files: {len(all_failed_tars)}")
    
    if all_failed_tars:
        print("\nFailed tar files:")
        for tar in all_failed_tars:
            print(f"- {tar}")
        
        # Save failed tars list for retry
        failed_list_file = os.path.join(config['local_mds_dir'], 'failed_tars.txt')
        with open(failed_list_file, 'w') as f:
            f.write('\n'.join(all_failed_tars))
        print(f"\nFailed tar files list saved to: {failed_list_file}")
        return
    
    print("\nAll processes completed. Verifying and merging shard metadata...")
    try:
        # Check for process completion files
        incomplete_processes = []
        for i in range(config['num_proc']):
            complete_file = os.path.join(config['local_mds_dir'], str(i), 'process_complete')
            if not os.path.exists(complete_file):
                incomplete_processes.append(i)
        
        if incomplete_processes:
            print(f"Error: Processes {incomplete_processes} did not complete successfully")
            return
            
        # Merge metadata
        shards_metadata = [
            os.path.join(config['local_mds_dir'], str(i), 'index.json')
            for i in range(config['num_proc'])
        ]
        
        missing_shards = [f for f in shards_metadata if not os.path.exists(f)]
        if missing_shards:
            print(f"Error: Missing shard metadata files: {missing_shards}")
            return
            
        merge_index(shards_metadata, out=config['local_mds_dir'], keep_local=True)
        print("Successfully merged all shard metadata!")
        
        # Clean up temporary files
        temp_dir = os.path.join(config['local_mds_dir'], 'temp')
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"Error during metadata merge: {e}")
        return


if __name__ == '__main__':
    main()
