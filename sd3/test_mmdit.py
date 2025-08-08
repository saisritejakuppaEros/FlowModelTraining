import torch
from mmdit import MMDiT  # Ensure this import is correct


import math
num_patches = 16

# Define the parameters
pos_embed_max_size = round(math.sqrt(num_patches))
patch_size = 4
depth = 6
adm_in_channels = 16
context_shape = (16, 16)
context_embedder_config = None
device = "cpu"
dtype = torch.float32

# Create a list of image sizes to test
image_sizes = [16, 32]
# Iterate through image sizes and run the model
for input_size in image_sizes:
    print(f"Testing with input size: {input_size}x{input_size}")

    # Initialize the model with the given input size
    diffusion_model = MMDiT(
        input_size=None,  # Will set input_size dynamically
        pos_embed_scaling_factor=None,
        pos_embed_offset=None,
        pos_embed_max_size=pos_embed_max_size,
        patch_size=patch_size,
        in_channels=4,  # Set as per your latent space channels
        depth=depth,
        num_patches=num_patches,
        adm_in_channels=adm_in_channels,
        context_embedder_config=context_embedder_config,
        device=device,
        dtype=dtype,
    )

    # print(diffusion_model)  # Print model details

    # Dummy input tensor with the current image size
    x = torch.randn(1, 4, input_size, input_size)  # Batch size = 1, Channels = 16, H = W = input_size
    t = torch.randint(0, 1000, (1,))  # Random timestep tensor
    y = None  # No class labels for this test
    context = None  # No additional context for this test

    # Forward pass
    with torch.no_grad():
        out = diffusion_model(x, t, y, context)

    # Output results
    print(f"Output shape for input size {input_size}x{input_size}: {out.shape}")
    print("-" * 50)
