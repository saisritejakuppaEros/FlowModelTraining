#!/usr/bin/env python3
"""
Script to launch TensorBoard for viewing training logs.
This script automatically finds the most recent tensorboard logs and launches TensorBoard.
"""

import os
import argparse
import subprocess
import glob
from pathlib import Path

def find_latest_log_dir(base_dir="./tensorboard_logs"):
    """Find the most recent TensorBoard log directory."""
    if not os.path.exists(base_dir):
        print(f"❌ TensorBoard log directory '{base_dir}' does not exist.")
        print("Make sure you have run training with tensorboard_enabled: true")
        return None
    
    # Find all experiment directories
    log_dirs = glob.glob(os.path.join(base_dir, "*"))
    log_dirs = [d for d in log_dirs if os.path.isdir(d)]
    
    if not log_dirs:
        print(f"❌ No log directories found in '{base_dir}'")
        return None
    
    # Sort by modification time (most recent first)
    log_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_dir = log_dirs[0]
    
    print(f"📊 Found {len(log_dirs)} log directories")
    print(f"🚀 Using latest: {latest_dir}")
    
    return latest_dir

def launch_tensorboard(log_dir=None, port=6006, host="localhost"):
    """Launch TensorBoard with the specified log directory."""
    if log_dir is None:
        log_dir = find_latest_log_dir()
        if log_dir is None:
            return False
    
    if not os.path.exists(log_dir):
        print(f"❌ Log directory '{log_dir}' does not exist.")
        return False
    
    print(f"🖥️ Launching TensorBoard on http://{host}:{port}")
    print(f"📁 Log directory: {log_dir}")
    print("📊 TensorBoard will show:")
    print("   - Training/Validation loss curves")
    print("   - Learning rate schedule")
    print("   - Gradient norms (if enabled)")
    print("   - Weight histograms (if enabled)")
    print("   - Validation images (if available)")
    print("\n🔄 Starting TensorBoard server...")
    print("   Press Ctrl+C to stop")
    
    try:
        # Launch TensorBoard
        cmd = [
            "tensorboard",
            "--logdir", log_dir,
            "--port", str(port),
            "--host", host,
            "--reload_interval", "30",  # Reload every 30 seconds
        ]
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n👋 TensorBoard stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start TensorBoard: {e}")
        print("💡 Make sure TensorBoard is installed: pip install tensorboard")
        return False
    except FileNotFoundError:
        print("❌ TensorBoard not found. Please install it:")
        print("   pip install tensorboard")
        return False

def main():
    parser = argparse.ArgumentParser(description="Launch TensorBoard for SD3 training logs")
    parser.add_argument(
        "--logdir", 
        type=str, 
        help="Path to TensorBoard log directory (default: auto-detect latest)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=6006, 
        help="Port for TensorBoard server (default: 6006)"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost", 
        help="Host for TensorBoard server (default: localhost)"
    )
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List available log directories and exit"
    )
    
    args = parser.parse_args()
    
    if args.list:
        base_dir = "./tensorboard_logs"
        if os.path.exists(base_dir):
            log_dirs = glob.glob(os.path.join(base_dir, "*"))
            log_dirs = [d for d in log_dirs if os.path.isdir(d)]
            log_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            print(f"📊 Available log directories in {base_dir}:")
            for i, log_dir in enumerate(log_dirs):
                mtime = os.path.getmtime(log_dir)
                import datetime
                mtime_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                print(f"  {i+1}. {os.path.basename(log_dir)} (modified: {mtime_str})")
        else:
            print(f"❌ TensorBoard log directory '{base_dir}' does not exist.")
        return
    
    success = launch_tensorboard(args.logdir, args.port, args.host)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
