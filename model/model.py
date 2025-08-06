import torch
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration
from diffusers import (
    QwenImagePipeline,
    QwenImageTransformer2DModel,
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler
)

# Set the base model path (Hugging Face or local)
BASE_MODEL = "Qwen/Qwen-Image"

# Choose precision and device
dtype = torch.float16 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VAE
vae = AutoencoderKLQwenImage.from_pretrained(BASE_MODEL, subfolder="vae", torch_dtype=dtype)
vae = vae.to(device)
print("✅ VAE loaded")

# Create dummy image tensor (BCHW: 1x3x512x512)
dummy_image = torch.randn(1, 3, 1, 512, 512).to(device, dtype=dtype)

# Encode and decode with VAE
with torch.no_grad():
    latents = vae.encode(dummy_image).latent_dist.sample()
    decoded = vae.decode(latents).sample
print("✅ VAE encode/decode success")

# Load Text Encoder and Tokenizer
tokenizer = Qwen2Tokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE_MODEL, subfolder="text_encoder", torch_dtype=dtype)
text_encoder = text_encoder.to(device)
print("✅ Text encoder loaded")

# Encode dummy prompt
dummy_prompt = "a beautiful scenic mountain with sunrise"
inputs = tokenizer(dummy_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    output = text_encoder(**inputs, output_hidden_states=True)
    text_hidden_states = output.hidden_states[-1]
print("✅ Text prompt encoded, shape:", text_hidden_states.shape)

# Load Transformer (DiT)
transformer = QwenImageTransformer2DModel.from_pretrained(BASE_MODEL, subfolder="transformer", torch_dtype=dtype)
transformer = transformer.to(device)
print("✅ DiT transformer loaded")

# Prepare dummy latent input for transformer
batch_size = 1
latent_dim = latents.shape[1]
H, W = latents.shape[-2], latents.shape[-1]
dummy_latents = torch.randn(batch_size, latent_dim, 1, H, W).to(device, dtype=dtype)

# Flatten and pack
dummy_latents = dummy_latents.permute(0, 2, 1, 3, 4)  # BxFxdxHxW -> BxFxdxHxW (for time)
packed_latents = QwenImagePipeline._pack_latents(dummy_latents, batch_size, latent_dim, H, W)

print("the shape of packed latents", packed_latents.shape)

# Dummy timestep and shape info
timestep = torch.tensor([500.0]).to(device) / 1000.0
txt_lens = [text_hidden_states.shape[1]]
img_shapes = [(1, H // 2, W // 2)]

# Run through DiT
with torch.no_grad():
    transformer_output = transformer(
        hidden_states=packed_latents,
        encoder_hidden_states=text_hidden_states,
        encoder_hidden_states_mask=torch.ones_like(inputs["input_ids"]),
        timestep=timestep,
        txt_seq_lens=txt_lens,
        img_shapes=img_shapes,
        return_dict=False,
    )[0]
print("✅ Transformer output shape:", transformer_output.shape)
