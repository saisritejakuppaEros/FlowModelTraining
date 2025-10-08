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
    'local_mds_dir': './mds',  # Directory to store the mds shards
    'max_image_size': 512,  # Max image size for resizing
    'min_image_size': 256,  # Min image size, images smaller than this are skipped
    'num_proc': 16  # Number of processes for multiprocessing
}


os.makedirs(DEFAULT_CONFIG['local_mds_dir'], exist_ok=True)


def current_process_index() -> int:
    p = current_process()
    return p._identity[0] - 1


def read_tar(path: str, path_out: str) -> Generator[Tuple[Image.Image, str], None, None]:
    os.makedirs(path_out, exist_ok=False)
    with tarfile.open(path, 'r') as tar:
        tar.extractall(path_out)

    txts = sorted(glob.glob(os.path.join(path_out, '*txt')))
    print(f"Found {len(txts)} images in tar file")
    
    for t in txts:
        try:
            with open(t, 'r') as ct:
                cap = ct.read()
            img = Image.open(t.replace('.txt', '.jpg'))
            yield img, cap
        except Exception as e:
            print(e)
            
    print("Done reading the tar file")
    shutil.rmtree(path_out)
    shutil.rmtree(os.path.dirname(path_out))


def write_tar(tars: List[str], config: dict):
    columns = {
        'width': 'int32',
        'height': 'int32',
        'jpg': 'jpeg',
        'caption': 'str'
    }
    
    save_dir = os.path.join(config['local_mds_dir'], str(current_process_index()))
    os.makedirs(save_dir, exist_ok=True)
    
    writer = MDSWriter(
        out=save_dir,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64
    )
    
    downsize = transforms.Resize(
        config['max_image_size'],
        antialias=True,
        interpolation=transforms.InterpolationMode.BICUBIC
    )
    
    temp_dir = os.path.join(save_dir, f'temp/wds_{current_process_index()}')
    
    for tar in tars:
        rejected, total = 0, 0
        for img, cap in tqdm(read_tar(tar, temp_dir)):
            w, h = img.size
            try:
                if min(w, h) > config['max_image_size']:
                    img = downsize(img)
                if min(w, h) < config['min_image_size']:
                    rejected += 1
                    print(
                        f'Skipping image with resolution ({h}, {w}) - '
                        f'Since at least one side has resolution below {config["min_image_size"]}'
                    )
                    continue
            except (UnidentifiedImageError, OSError) as e:
                print(f"Error {e}")

            mds_sample = {
                'jpg': img,
                'caption': cap,
                'width': w,
                'height': h
            }
            writer.write(mds_sample)
            total += 1

        print(f"Rejected {rejected}, total {total}, Tar: {tar}")
    writer.finish()


def main():
    config = DEFAULT_CONFIG
    print(f"Using WDS dir: {config['wds_dir']}")
    tars = glob.glob(os.path.join(config['wds_dir'], '*tar'))[:2]
    print(f"Total {len(tars)} tar files found in cc12m wds dataset path!")

    tars_split = np.array_split(tars, config['num_proc'])
    
    print(f"Processing {len(tars)} tar files across {config['num_proc']} processes...")
    with Pool(processes=config['num_proc']) as pool:
        # Create a progress bar for the main processing
        with tqdm(total=len(tars), desc="Processing tar files", unit="tar") as pbar:
            # Use a callback to update progress
            def update_progress(result):
                pbar.update(1)
            
            # Submit all tasks
            results = []
            for ts in tars_split:
                result = pool.apply_async(write_tar, (ts, config), callback=update_progress)
                results.append(result)
            
            # Wait for all tasks to complete
            for result in results:
                result.get()
    
    print("Merging shard metadata...")
    shards_metadata = [
        os.path.join(config['local_mds_dir'], str(i), 'index.json')
        for i in range(config['num_proc'])
    ]
    merge_index(shards_metadata, out=config['local_mds_dir'], keep_local=True)


if __name__ == '__main__':
    main()
