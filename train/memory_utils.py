"""
Memory monitoring and cleanup utilities for CUDA memory management.
"""

import torch
import gc
import psutil
import os
from typing import Optional, Dict, Any
import logging

def print_gpu_memory_usage(prefix: str = "", device: int = 0):
    """Print current GPU memory usage with detailed breakdown."""
    if not torch.cuda.is_available():
        print(f"{prefix} CUDA not available")
        return
    
    try:
        # Get memory info
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3    # GB
        
        # Get GPU properties
        props = torch.cuda.get_device_properties(device)
        total_memory = props.total_memory / 1024**3  # GB
        
        print(f"{prefix} GPU Memory Usage (Device {device}):")
        print(f"  Allocated: {allocated:.2f} GB / {total_memory:.2f} GB ({allocated/total_memory*100:.1f}%)")
        print(f"  Reserved:  {reserved:.2f} GB / {total_memory:.2f} GB ({reserved/total_memory*100:.1f}%)")
        print(f"  Max Allocated: {max_allocated:.2f} GB")
        print(f"  Max Reserved:  {max_reserved:.2f} GB")
        print(f"  Free: {total_memory - reserved:.2f} GB")
        
    except Exception as e:
        print(f"{prefix} Error getting GPU memory info: {e}")

def print_cpu_memory_usage(prefix: str = ""):
    """Print current CPU memory usage."""
    try:
        # Get process memory info
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Convert to GB
        rss = memory_info.rss / 1024**3  # Resident Set Size
        vms = memory_info.vms / 1024**3  # Virtual Memory Size
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        total_memory = system_memory.total / 1024**3
        available_memory = system_memory.available / 1024**3
        
        print(f"{prefix} CPU Memory Usage:")
        print(f"  Process RSS: {rss:.2f} GB")
        print(f"  Process VMS: {vms:.2f} GB")
        print(f"  System Total: {total_memory:.2f} GB")
        print(f"  System Available: {available_memory:.2f} GB")
        print(f"  System Used: {(total_memory - available_memory):.2f} GB ({(total_memory - available_memory)/total_memory*100:.1f}%)")
        
    except Exception as e:
        print(f"{prefix} Error getting CPU memory info: {e}")

def cleanup_gpu_memory(verbose: bool = True):
    """Clean up GPU memory with aggressive cleanup."""
    if not torch.cuda.is_available():
        if verbose:
            print("CUDA not available, skipping GPU cleanup")
        return
    
    try:
        if verbose:
            print("🧹 Cleaning up GPU memory...")
            print_gpu_memory_usage("Before cleanup:")
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Clear cache again after GC
        torch.cuda.empty_cache()
        
        if verbose:
            print_gpu_memory_usage("After cleanup:")
            
    except Exception as e:
        if verbose:
            print(f"Error during GPU cleanup: {e}")

def monitor_memory_usage(func, *args, prefix: str = "", verbose: bool = True, **kwargs):
    """
    Decorator/wrapper to monitor memory usage before and after function execution.
    
    Args:
        func: Function to execute
        *args: Arguments for the function
        prefix: Prefix for logging messages
        verbose: Whether to print detailed memory info
        **kwargs: Keyword arguments for the function
    
    Returns:
        Function result and memory usage dict
    """
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"🔍 Memory Monitor: {prefix}{func.__name__ if hasattr(func, '__name__') else 'Function'}")
        print(f"{'='*50}")
        print_gpu_memory_usage("Before execution:")
        print_cpu_memory_usage("Before execution:")
    
    # Record memory before
    memory_before = {}
    if torch.cuda.is_available():
        memory_before['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3
        memory_before['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3
    
    process = psutil.Process(os.getpid())
    memory_before['cpu_rss'] = process.memory_info().rss / 1024**3
    
    try:
        # Execute function
        result = func(*args, **kwargs)
        
        # Record memory after
        memory_after = {}
        if torch.cuda.is_available():
            memory_after['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3
            memory_after['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3
        
        memory_after['cpu_rss'] = process.memory_info().rss / 1024**3
        
        if verbose:
            print_gpu_memory_usage("After execution:")
            print_cpu_memory_usage("After execution:")
            
            # Print memory changes
            if torch.cuda.is_available():
                gpu_alloc_change = memory_after['gpu_allocated'] - memory_before['gpu_allocated']
                gpu_reserved_change = memory_after['gpu_reserved'] - memory_before['gpu_reserved']
                print(f"\n📊 Memory Changes:")
                print(f"  GPU Allocated: {gpu_alloc_change:+.2f} GB")
                print(f"  GPU Reserved:  {gpu_reserved_change:+.2f} GB")
            
            cpu_change = memory_after['cpu_rss'] - memory_before['cpu_rss']
            print(f"  CPU RSS: {cpu_change:+.2f} GB")
            print(f"{'='*50}\n")
        
        return result, {
            'before': memory_before,
            'after': memory_after
        }
        
    except Exception as e:
        if verbose:
            print(f"❌ Error during execution: {e}")
            print_gpu_memory_usage("After error:")
            cleanup_gpu_memory(verbose=False)
        raise e

def setup_memory_logging(log_file: Optional[str] = None, log_level: int = logging.INFO):
    """Setup memory usage logging to file."""
    logger = logging.getLogger('memory_monitor')
    logger.setLevel(log_level)
    
    if log_file:
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def log_memory_usage(logger: logging.Logger, prefix: str = "", device: int = 0):
    """Log current memory usage to the logger."""
    if not torch.cuda.is_available():
        logger.info(f"{prefix} CUDA not available")
        return
    
    try:
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        
        process = psutil.Process(os.getpid())
        cpu_rss = process.memory_info().rss / 1024**3
        
        logger.info(f"{prefix} GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved | CPU: {cpu_rss:.2f}GB RSS")
        
    except Exception as e:
        logger.error(f"{prefix} Error logging memory usage: {e}")

def emergency_cleanup():
    """Emergency cleanup function for OOM situations."""
    print("🚨 Emergency memory cleanup initiated!")
    
    try:
        # Clear all caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ Emergency cleanup completed")
        print_gpu_memory_usage("After emergency cleanup:")
        
    except Exception as e:
        print(f"❌ Error during emergency cleanup: {e}")

# Context manager for automatic memory cleanup
class MemoryManager:
    """Context manager for automatic memory cleanup."""
    
    def __init__(self, cleanup_on_exit: bool = True, verbose: bool = True):
        self.cleanup_on_exit = cleanup_on_exit
        self.verbose = verbose
        self.memory_before = {}
        
    def __enter__(self):
        if self.verbose:
            print("🔍 Memory Manager: Entering context")
            print_gpu_memory_usage("Initial state:")
        
        # Record initial memory state
        if torch.cuda.is_available():
            self.memory_before['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3
            self.memory_before['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3
        
        process = psutil.Process(os.getpid())
        self.memory_before['cpu_rss'] = process.memory_info().rss / 1024**3
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print("🔍 Memory Manager: Exiting context")
            if exc_type is not None:
                print(f"❌ Exception occurred: {exc_type.__name__}: {exc_val}")
        
        if self.cleanup_on_exit:
            cleanup_gpu_memory(verbose=self.verbose)
        
        # Report memory changes
        if self.verbose and torch.cuda.is_available():
            current_allocated = torch.cuda.memory_allocated() / 1024**3
            current_reserved = torch.cuda.memory_reserved() / 1024**3
            
            alloc_change = current_allocated - self.memory_before['gpu_allocated']
            reserved_change = current_reserved - self.memory_before['gpu_reserved']
            
            print(f"📊 Memory Changes During Context:")
            print(f"  GPU Allocated: {alloc_change:+.2f} GB")
            print(f"  GPU Reserved:  {reserved_change:+.2f} GB")
