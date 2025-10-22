import os
from data_converstion_mds import retry_failed_tars, DEFAULT_CONFIG

def main():
    # Get the failed tars file path - using the actual path where the file was saved
    failed_tars_file = '/data0/teja_works/diffusion_training/dataset_preparation/micro_diffusion/FlowModelTraining/data_gen/cc12m_dataset/mds/failed_tars.txt'
    
    if not os.path.exists(failed_tars_file):
        print(f"Error: Failed tars file not found at {failed_tars_file}")
        return
        
    # Retry the failed tars
    success = retry_failed_tars(DEFAULT_CONFIG, failed_tars_file)
    
    if success:
        print("\nAll previously failed tars were processed successfully!")
        # Clean up the failed tars file
        os.remove(failed_tars_file)
    else:
        print("\nSome tars still failed. Check retry_failed_tars.txt for details.")

if __name__ == '__main__':
    main()
