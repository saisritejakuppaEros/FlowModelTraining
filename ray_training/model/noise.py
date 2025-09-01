# ---- in model/noise.py ----
import torch

class NoiseScheduler(torch.nn.Module):
    def __init__(self, num_training_timesteps: int = 1_000, base_shift: float = 0.5, shift: float = 3,
                 num_inference_timesteps: int = 50, inference: bool = False):
        super().__init__()
        self.base_shift = base_shift
        self.shift = shift

        timesteps = torch.linspace(1.0, num_training_timesteps, int(num_training_timesteps))
        sigmas = timesteps / num_training_timesteps
        sigmas = sigmas * shift / (1 + (shift - 1) * sigmas)

        # keep these as buffers so we can easily .to(device)
        self.register_buffer("sigmas", sigmas, persistent=False)
        self.register_buffer("timesteps", sigmas * num_training_timesteps, persistent=False)

        self.num_training_timesteps = num_training_timesteps
        self.num_inference_timesteps = num_inference_timesteps

        self.sigma_min = float(self.sigmas[-1])
        self.sigma_max = float(self.sigmas[0])

        if inference:
            max_timestep = self.sigma_to_timestep(self.sigma_max)
            min_timestep = self.sigma_to_timestep(self.sigma_min)

            ts = torch.linspace(max_timestep, min_timestep, num_inference_timesteps)
            s = (ts / num_training_timesteps)
            s = s * shift / (1 + (shift - 1) * s)

            # buffers for inference schedule
            self.register_buffer("sigmas", torch.cat([s.flip(0), torch.zeros(1, device=s.device)]), persistent=False)
            self.register_buffer("timesteps", (s * num_training_timesteps).flip(0), persistent=False)

        self.step_index = 0

    def sigma_to_timestep(self, sigma):
        return sigma * self.num_training_timesteps

    def check_timestep(self, timestep):
        """Accepts scalar int or LongTensor[B]."""
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 0:
                raise ValueError("Empty timestep tensor.")
            max_t = int(timestep.max().item())
        else:
            max_t = int(timestep)
        if max_t >= self.num_training_timesteps:
            raise ValueError("Timestep >= num_training_timesteps.")

    def sample_logit_timestep(self, batch_size: int = 1, device: torch.device | str = "cpu") -> torch.LongTensor:
        """Return vector of indices in [0, T-1], shape [B]."""
        u = torch.rand(batch_size, device=device)
        z = torch.logit(u, eps=1e-8) * self.shift + self.base_shift
        sampled_t = torch.sigmoid(z)
        idx = (sampled_t * (self.num_training_timesteps - 1)).long()
        return idx  # [B]

    def add_noise(self, image: torch.FloatTensor, timestep: int | torch.Tensor | None = None):
        """
        Forward process: x_t = (1 - sigma) * x0 + sigma * eps
        Accepts per-batch timesteps.
        """
        B = image.size(0)
        device = image.device

        if timestep is None:
            timestep = self.sample_logit_timestep(B, device=device)  # [B]
        elif not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep] * B, device=device, dtype=torch.long)
        else:
            timestep = timestep.to(device).long()
            if timestep.ndim == 0:
                timestep = timestep.expand(B)
            elif timestep.shape[0] != B and timestep.numel() == 1:
                timestep = timestep.expand(B)
            elif timestep.shape[0] != B:
                raise ValueError(f"timestep shape {timestep.shape} incompatible with batch {B}")

        self.check_timestep(timestep)

        noise = torch.randn_like(image)

        # vectorized gather of sigmas/timesteps
        sigmas = self.sigmas.to(device)[timestep]                           # [B]
        sigmas = sigmas.view(B, *([1] * (image.ndim - 1)))                  # [B,1,1,1]
        ts_cont = self.timesteps.to(device)[timestep]                        # [B] (continuous t)

        noised = (1.0 - sigmas) * image + noise * sigmas
        return noised, noise, ts_cont

    @torch.no_grad()
    def reverse_flow(self, current_sample: torch.Tensor, model_output: torch.FloatTensor, stochasticity: bool):
        current_sample = current_sample.to(torch.float32)
        current_sigma = self.sigmas[self.step_index]
        next_sigma = self.sigmas[self.step_index + 1]
        dt = next_sigma - current_sigma
        if stochasticity:
            noise = torch.randn_like(current_sample)
            x_prev = current_sample - current_sigma * model_output
            prev_sample = (1 - next_sigma) * x_prev + noise * next_sigma
        else:
            prev_sample = current_sample + dt * model_output
        self.step_index += 1
        return prev_sample
