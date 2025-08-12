from logzero import logger
from utils.config import create_component
import torch

class ModelTrainer:
    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    
    def load_vae(self, config):
        logger.debug(f"Loading VAE from {config}")
        vae = create_component(config["module"], config["params"], fsdp_spec=config.get("fsdp", None))
        vae = vae.to(self.device, dtype=torch.bfloat16)
        for param in vae.parameters():
            param.requires_grad = False
        self.vae = vae
        logger.info(f"Loaded VAE")
        
    
    def load_text_encoder(self, config):
        logger.debug(f"Loading Text Encoder from {config}")
        text_encoder = create_component(config["module"], config["params"], fsdp_spec=config.get("fsdp", None))
        text_encoder = text_encoder.to(self.device, dtype=torch.bfloat16)
        for param in text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder = text_encoder
        logger.info(f"Loaded Text Encoder")
    

    def load_clip_encoder(self, config):
        logger.debug(f"Loading CLIP Encoder from {config}")
        clip_encoder = create_component(config["module"], config["params"], fsdp_spec=config.get("fsdp", None))
        clip_encoder = clip_encoder.to(self.device, dtype=torch.bfloat16)
        for param in clip_encoder.parameters():
            param.requires_grad = False
        self.clip_encoder = clip_encoder
        logger.info(f"Loaded CLIP Encoder")
        
    def load_patchifier(self, config):
        logger.debug(f"Loading Patchifier from {config}")
        patchifier = create_component(config["module"], config["params"], fsdp_spec=config.get("fsdp", None))
        self.patchifier = patchifier
        logger.info(f"Loaded Patchifier")
        
    def load_denoiser(self, config):
        logger.debug(f"Loading Denoiser from {config}")
        denoiser = create_component(config["module"], config["params"], fsdp_spec=config.get("fsdp", None))
        denoiser.init_weights()
        for param in denoiser.parameters():
            param.requires_grad = True
        self.denoiser = denoiser
        logger.info(f"Loaded Denoiser")
        
        
    def load_timesampler(self, config):
        time_sampler_config = config
        time_sampler = create_component(time_sampler_config["module"], time_sampler_config["params"])
        self.time_sampler = time_sampler
        
    def load_timewarper(self, config):
        time_warper_config = config
        time_warper = create_component(time_warper_config["module"], time_warper_config["params"])
        self.time_warper = time_warper
        
    def load_flownoiser(self, config):
        flow_noiser_config = config
        flow_noiser = create_component(flow_noiser_config["module"], flow_noiser_config["params"])
        self.flow_noiser = flow_noiser
        
