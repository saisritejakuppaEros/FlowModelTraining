accelerate launch --num_processes=4 latents_gen.py \
    --input_mds ./mds \
    --output_dir ./flux_latents_mds \
    --resolution 256