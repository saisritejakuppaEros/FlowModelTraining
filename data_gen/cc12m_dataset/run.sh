accelerate launch --num_processes=8 latents_gen.py \
    --input_mds ./mds \
    --output_dir ./flux_latents_mds \
    --resolution 256


    # /data0/teja_works/diffusion_training/dataset_preparation/micro_diffusion/FlowModelTraining/train/output/tensorboard_logs/flux_training_run