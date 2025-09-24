#!/usr/bin/env python3
"""
Launch script for FLUX dataset generation with multi-GPU support.
This script runs the dataset generation with multiprocessing across GPUs 5, 6, 7.
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    target_script = script_dir / "stream_flux.py"
    
    # Check if target script exists
    if not target_script.exists():
        print(f"Error: {target_script} not found!")
        sys.exit(1)
    
    # Simple direct launch
    cmd = ["python", str(target_script)]
    
    print("Launching FLUX dataset generation with multi-GPU support...")
    print(f"Command: {' '.join(cmd)}")    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True)
        print("Dataset generation completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running dataset generation: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nDataset generation interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())
