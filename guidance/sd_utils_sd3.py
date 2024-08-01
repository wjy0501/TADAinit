# This code is based on RFDS https://github.com/yangxiaofeng/rectified_flow_prior/blob/main/threestudio/models/guidance/RFDS_Rev_sd3.py
from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers import StableDiffusion3Pipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils.import_utils import is_xformers_available

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

class StableDiffusion(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        sd_version="sd3",
        hf_key=None,
        t_range=[0.02, 0.98],
    ):
        super().__init__()
        self.device = device
        self.sd_version = sd_version

        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
        elif self.sd_version == "sd3":
            model_key = "stable-diffusion-3-medium-diffusers"
        else:
            raise ValueError(
                f"Only support sd3, Stable-diffusion version {self.sd_version} not supported."
            )
        
        self.weights_dtype = torch.float16 if fp16 else torch.float32

        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_key,
            torch_dtype=torch.float16,
        )

        pipe = pipe.to(self.device)

        # if not is_xformers_available():
        #     warnings.warn(
        #         "xformers is not available, memory efficient attention is not enabled."
        #     )
        # else:
        #     pipe.enable_xformers_memory_efficient_attention()

        self.pipe = pipe
        self.vae = pipe.vae
        self.transformer = pipe.transformer

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.transformer.parameters():
            p.requires_grad_(False)

        self.scheduler = self.pipe.scheduler
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        
        self.embeddings = {}
        self.pooled_embeddings = {}

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds, pooled_pos_embeds = self.encode_text(prompts)
        neg_embeds, pooled_neg_embeds = self.encode_text(negative_prompts)

        self.embeddings['pos'] = pos_embeds
        self.pooled_embeddings['pos'] = pooled_pos_embeds
        self.embeddings['neg'] = neg_embeds
        self.pooled_embeddings['neg'] = pooled_neg_embeds

         # directional embeddings
        for d in ['front', 'side', 'back']:
            embeds, pooled_emebds = self.encode_text([f'{p}, {d} view' for p in prompts])
            self.embeddings[d] = embeds
            self.pooled_embeddings[d] = pooled_emebds

        del self.pipe.tokenizer
        del self.pipe.tokenizer_2
        del self.pipe.tokenizer_3
        del self.pipe.text_encoder
        del self.pipe.text_encoder_2
        del self.pipe.text_encoder_3


    def encode_text(self, prompt):
        (
            prompt_embeds,
            _,
            pooled_prompt_embeds,
            _,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            do_classifier_free_guidance=False,
            device=self.device
        )
        return prompt_embeds, pooled_prompt_embeds
    
    @torch.no_grad()
    def forward_transformer(
        self,
        transformer,
        latents,
        t,
        prompt_embeds,
        pooled_prompt_embeds
    ):
        input_dtype = latents.dtype
        return transformer(
            hidden_states=latents.to(self.weights_dtype),
            timestep=t.to(self.weights_dtype),
            encoder_hidden_states=prompt_embeds.to(self.weights_dtype),
            pooled_projections = pooled_prompt_embeds.to(self.weights_dtype),
            joint_attention_kwargs=None,
            return_dict=False,
        )[0].to(input_dtype)

    @torch.no_grad()
    def encode_imgs(
        self, imgs
    ):
        input_dtype = imgs.dtype
        imgs = torch.clamp(imgs, min=0, max=1)
        imgs = self.pipe.image_processor.preprocess(imgs)
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample()
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.no_grad()
    def decode_latents(
        self,
        latents,
        latent_height: int = 64,
        latent_width: int = 64,
    ):
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)
    
    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float16):
        sigmas = self.scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def train_step(
        self,
        pred_rgb,
        step_ratio=None,
        guidance_scale=50,
        as_latent=False, resolution=(512,512),
        vers=None, hors=None,
    ):
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.weights_dtype)

        if as_latent:
            latents = F.interpolate(pred_rgb, (resolution[0]//8, resolution[1]//8), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to resolution (512x512) to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, resolution, mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        with torch.no_grad():
            if step_ratio is not None:
                # dreamtime-like
                # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
                indices = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
                indices = torch.full((batch_size,), indices, dtype=torch.long)
                timesteps = self.scheduler.timesteps[indices].to(device=self.device)
            else:
                indices = torch.randint(self.min_step, self.max_step + 1, (batch_size,))
                timesteps = self.scheduler.timesteps[indices].to(device=self.device)

            noise = torch.randn_like(latents)
            sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
            latents_noisy = sigmas * noise + (1.0 - sigmas) * latents

            if hors is None:
                text_embeddings = self.embeddings['pos'].expand(batch_size, -1, -1)
                embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])
                text_embeddings_pooled = self.pooled_embeddings['pos'].expand(batch_size, -1)
                pooled_embeddings = torch.cat([self.pooled_embeddings['pos'].expand(batch_size, -1), self.pooled_embeddings['neg'].expand(batch_size, -1)])
            else:
                def _get_dir_ind(h):
                    if abs(h) < 60: return 'front'
                    elif abs(h) < 120: return 'side'
                    else: return 'back'
                text_embeddings = torch.cat([self.embeddings[_get_dir_ind(h)] for h in hors], dim=0)
                embeddings = torch.cat([self.embeddings[_get_dir_ind(h)] for h in hors] + [self.embeddings['neg'].expand(batch_size, -1, -1)])
                text_embeddings_pooled = torch.cat([self.pooled_embeddings[_get_dir_ind(h)] for h in hors], dim=0)
                pooled_embeddings = torch.cat([self.pooled_embeddings[_get_dir_ind(h)] for h in hors] + [self.pooled_embeddings['neg'].expand(batch_size, -1)])

            # iRFDS
            noise_pred = self.forward_transformer(
                self.transformer,
                latents_noisy,
                timesteps,
                text_embeddings,
                text_embeddings_pooled,
            )
            # https://github.com/huggingface/diffusers/blob/614d0c64e96b37740e14bb5c2eca8f8a2ecdf23e/examples/dreambooth/train_dreambooth_lora_sd3.py#L1481
            new_latent = noise_pred * (-sigmas) + latents_noisy # the new latents
            noise = noise_pred + new_latent

            latents_noisy = sigmas * noise + (1.0 - sigmas) * latents            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_transformer(
                self.transformer,
                latent_model_input,
                torch.cat([timesteps] * 2),
                embeddings,
                pooled_embeddings,
            )

            (
                noise_pred_pretrain_text,
                noise_pred_pretrain_null,
            ) = noise_pred.chunk(2)
            u = torch.normal(mean=0, std=1, size=(batch_size,), device=self.device)
            # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
            model_pred = noise_pred_pretrain_null + guidance_scale * (noise_pred_pretrain_text - noise_pred_pretrain_null)

            grad = torch.nan_to_num(model_pred)

        target = (grad).detach()
        loss_rfds = 0.5 * F.mse_loss(noise - latents, target, reduction="mean") / batch_size

        return loss_rfds