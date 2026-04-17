from contextlib import nullcontext
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline


DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_GUIDANCE_SCALE = 7.5

MODEL_NAME_MAP = {
    "sd14": "CompVis/stable-diffusion-v1-4",
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sd21": "stabilityai/stable-diffusion-2-1-base",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
}


class DiffusionPatchGenerator(nn.Module):
    def __init__(
        self,
        *,
        patch_size: int,
        model_name: str,
        prompt: str,
        device: torch.device,
        init_mode: str = "random",
        model_path: str = "",
        base_image_path: str = "",
        source_steps: int = 50,
        reverse_steps: int = 25,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        latent_eps: float = 0.5,
        null_inner_steps: int = 15,
        null_epsilon: float = 1e-5,
        seed: int = 42,
        diffusion_dtype: str = "auto",
    ):
        super().__init__()

        if model_name not in MODEL_NAME_MAP:
            raise ValueError(f"Unsupported diffusion model: {model_name}")
        if init_mode not in {"random", "image"}:
            raise ValueError(f"Unsupported diffusion init mode: {init_mode}")
        if reverse_steps <= 0 or source_steps <= 0:
            raise ValueError("Diffusion steps must be positive")
        if reverse_steps > source_steps:
            raise ValueError("reverse_steps must be <= source_steps")
        if init_mode == "image" and not base_image_path:
            raise ValueError("Image init mode requires --diffusion_base_image")
        if init_mode == "image" and not prompt:
            raise ValueError("Image init mode requires --diffusion_prompt for null-text optimization")

        self.patch_size = patch_size
        self.model_name = model_name
        self.prompt = prompt or ""
        self.device = torch.device(device)
        self.init_mode = init_mode
        self.base_image_path = base_image_path
        self.source_steps = int(source_steps)
        self.reverse_steps = int(reverse_steps)
        self.guidance_scale = float(guidance_scale)
        self.latent_eps = float(latent_eps)
        self.null_inner_steps = int(null_inner_steps)
        self.null_epsilon = float(null_epsilon)
        self.seed = int(seed)
        self.height = DEFAULT_HEIGHT
        self.width = DEFAULT_WIDTH
        self.weight_dtype = self._resolve_weight_dtype(diffusion_dtype)
        self.latent_dtype = torch.float32
        self._adam_t = 0

        self.pipeline, self.is_sdxl = self._load_pipeline(model_name, model_path)
        self._freeze_pipeline()
        self.pipeline.scheduler.set_timesteps(self.source_steps)

        if self.is_sdxl:
            (
                self._sdxl_all_added_cond_kwargs,
                self._sdxl_added_cond_kwargs,
                self._sdxl_added_uncond_kwargs,
                self._sdxl_context,
            ) = self._encode_text_sdxl_with_negative(self.prompt)
        else:
            self._base_context = self._init_prompt(self.prompt)
            self._base_uncond_embeddings, self._base_cond_embeddings = self._base_context.chunk(2)

        base_latent, uncond_embeddings = self._prepare_initial_latent()
        self.register_buffer("base_latent", base_latent.to(self.device, dtype=self.latent_dtype))
        self.delta = nn.Parameter(torch.zeros_like(self.base_latent))
        self.register_buffer("_adam_m", torch.zeros_like(self.base_latent))
        self.register_buffer("_adam_v", torch.zeros_like(self.base_latent))

        if uncond_embeddings is None:
            self.uncond_embeddings = None
        else:
            self.uncond_embeddings = [emb.to(self.device) for emb in uncond_embeddings]

    def _resolve_weight_dtype(self, diffusion_dtype: str) -> torch.dtype:
        if self.device.type != "cuda":
            return torch.float32
        if diffusion_dtype == "fp32":
            return torch.float32
        if diffusion_dtype in {"auto", "fp16"}:
            return torch.float16
        raise ValueError(f"Unsupported diffusion dtype: {diffusion_dtype}")

    def _autocast(self):
        if self.device.type != "cuda" or self.weight_dtype == torch.float32:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.weight_dtype)

    def _load_pipeline(self, model_name: str, model_path: str):
        model_ref = model_path or MODEL_NAME_MAP[model_name]
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        is_sdxl = model_name == "sdxl"
        pipeline_cls = StableDiffusionXLPipeline if is_sdxl else StableDiffusionPipeline

        kwargs = {
            "torch_dtype": self.weight_dtype,
            "scheduler": scheduler,
        }
        if not is_sdxl:
            kwargs["safety_checker"] = None
            kwargs["requires_safety_checker"] = False

        try:
            pipeline = pipeline_cls.from_pretrained(model_ref, **kwargs)
        except TypeError:
            kwargs.pop("safety_checker", None)
            kwargs.pop("requires_safety_checker", None)
            pipeline = pipeline_cls.from_pretrained(model_ref, **kwargs)

        pipeline = pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        return pipeline, is_sdxl

    def _freeze_pipeline(self):
        self.pipeline.unet.eval().requires_grad_(False)
        self.pipeline.vae.eval().requires_grad_(False)
        self.pipeline.text_encoder.eval().requires_grad_(False)
        if self.is_sdxl:
            self.pipeline.text_encoder_2.eval().requires_grad_(False)

    def _get_text_embeddings(self, prompt, tokenizer, text_encoder):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        with torch.no_grad():
            prompt_embeds = text_encoder(
                text_input_ids.to(self.device),
                output_hidden_states=True,
            )

        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        if prompt == "":
            return torch.zeros_like(prompt_embeds), torch.zeros_like(pooled_prompt_embeds)
        return prompt_embeds, pooled_prompt_embeds

    def _encode_text_sdxl(self, prompt: str):
        prompt_embeds, _ = self._get_text_embeddings(
            prompt, self.pipeline.tokenizer, self.pipeline.text_encoder
        )
        prompt_embeds_2, pooled_prompt_embeds_2 = self._get_text_embeddings(
            prompt, self.pipeline.tokenizer_2, self.pipeline.text_encoder_2
        )
        prompt_embeds = torch.cat((prompt_embeds, prompt_embeds_2), dim=-1)
        text_encoder_projection_dim = self.pipeline.text_encoder_2.config.projection_dim
        add_time_ids = self.pipeline._get_add_time_ids(
            (1024, 1024),
            (0, 0),
            (1024, 1024),
            self.weight_dtype,
            text_encoder_projection_dim,
        ).to(self.device)
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds_2.to(self.device),
            "time_ids": add_time_ids,
        }
        return added_cond_kwargs, prompt_embeds.to(self.device)

    def _encode_text_sdxl_with_negative(self, prompt: str):
        added_cond_kwargs, prompt_embeds = self._encode_text_sdxl(prompt)
        added_cond_kwargs_uncond, prompt_embeds_uncond = self._encode_text_sdxl("")
        context = torch.cat((prompt_embeds_uncond, prompt_embeds), dim=0)
        all_added_cond_kwargs = {
            "text_embeds": torch.cat(
                (added_cond_kwargs_uncond["text_embeds"], added_cond_kwargs["text_embeds"]),
                dim=0,
            ),
            "time_ids": torch.cat(
                (added_cond_kwargs_uncond["time_ids"], added_cond_kwargs["time_ids"]),
                dim=0,
            ),
        }
        return (
            all_added_cond_kwargs,
            added_cond_kwargs,
            added_cond_kwargs_uncond,
            context,
        )

    @torch.no_grad()
    def _init_prompt(self, prompt: str):
        uncond_input = self.pipeline.tokenizer(
            [""],
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.pipeline.text_encoder(
            uncond_input.input_ids.to(self.device)
        )[0]
        text_input = self.pipeline.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.pipeline.text_encoder(text_input.input_ids.to(self.device))[0]
        return torch.cat([uncond_embeddings, text_embeddings], dim=0)

    def _load_base_image(self) -> np.ndarray:
        image = Image.open(self.base_image_path)
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
        rgb = Image.alpha_composite(background, rgba).convert("RGB")

        w, h = rgb.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        rgb = rgb.crop((left, top, left + side, top + side))
        rgb = rgb.resize((self.width, self.height), Image.Resampling.BICUBIC)
        return np.array(rgb)

    @torch.no_grad()
    def _image_to_latent(self, image: np.ndarray) -> torch.Tensor:
        image_tensor = torch.from_numpy(image).float() / 127.5 - 1.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        vae = self.pipeline.vae
        original_dtype = vae.dtype
        needs_upcast = (
            self.is_sdxl
            and original_dtype == torch.float16
            and getattr(vae.config, "force_upcast", False)
        )
        if needs_upcast:
            vae.to(dtype=torch.float32)
            image_tensor = image_tensor.to(dtype=torch.float32)
        else:
            image_tensor = image_tensor.to(dtype=vae.dtype)

        autocast_ctx = nullcontext() if needs_upcast else self._autocast()
        with autocast_ctx:
            latent = vae.encode(image_tensor).latent_dist.mean
        latent = latent * vae.config.scaling_factor

        if needs_upcast:
            vae.to(dtype=original_dtype)

        return latent.to(self.device, dtype=self.latent_dtype)

    def _decode_latents(self, latents: torch.Tensor, *, output_size: Optional[int] = None) -> torch.Tensor:
        vae = self.pipeline.vae
        original_dtype = vae.dtype
        decode_latents = latents.to(self.device)
        needs_upcast = (
            self.is_sdxl
            and original_dtype == torch.float16
            and getattr(vae.config, "force_upcast", False)
        )
        if needs_upcast:
            vae.to(dtype=torch.float32)
            decode_latents = decode_latents.to(dtype=torch.float32)
        else:
            decode_latents = decode_latents.to(dtype=vae.dtype)

        autocast_ctx = nullcontext() if needs_upcast else self._autocast()
        with autocast_ctx:
            vae_input = (decode_latents / vae.config.scaling_factor).to(
                self.device, dtype=decode_latents.dtype
            )
            image = vae.decode(
                vae_input, return_dict=False
            )[0]
        image = image.div(2).add(0.5).clamp(0.0, 1.0)

        if output_size is not None and image.shape[-1] != output_size:
            image = F.interpolate(
                image,
                size=(output_size, output_size),
                mode="bilinear",
                align_corners=False,
            )

        if needs_upcast:
            vae.to(dtype=original_dtype)

        return image

    def _next_step(self, model_output, timestep: int, sample: torch.Tensor):
        timestep_value = int(timestep)
        timestep_value = min(
            timestep_value
            - self.pipeline.scheduler.config.num_train_timesteps
            // self.pipeline.scheduler.num_inference_steps,
            999,
        )
        next_timestep = int(timestep)
        alpha_prod_t = (
            self.pipeline.scheduler.alphas_cumprod[timestep_value]
            if timestep_value >= 0
            else self.pipeline.scheduler.final_alpha_cumprod
        )
        alpha_prod_t_next = self.pipeline.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        return alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction

    def _prev_step(self, model_output, timestep: int, sample: torch.Tensor):
        timestep_value = int(timestep)
        prev_timestep = (
            timestep_value
            - self.pipeline.scheduler.config.num_train_timesteps
            // self.pipeline.scheduler.num_inference_steps
        )
        alpha_prod_t = self.pipeline.scheduler.alphas_cumprod[timestep_value]
        alpha_prod_t_prev = (
            self.pipeline.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.pipeline.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        return alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction

    def _get_noise_pred_single(
        self,
        latent: torch.Tensor,
        t,
        context: torch.Tensor,
        added_cond_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        latent_input = self.pipeline.scheduler.scale_model_input(latent, t)
        if self.is_sdxl:
            return self.pipeline.unet(
                latent_input.to(self.device, dtype=self.weight_dtype),
                t,
                encoder_hidden_states=context.to(self.device, dtype=self.weight_dtype),
                added_cond_kwargs={
                    key: value.to(self.device, dtype=self.weight_dtype)
                    for key, value in added_cond_kwargs.items()
                },
                return_dict=False,
            )[0]
        return self.pipeline.unet(
            latent_input.to(self.device, dtype=self.weight_dtype),
            t,
            encoder_hidden_states=context.to(self.device, dtype=self.weight_dtype),
            return_dict=False,
        )[0]

    def _guided_diffusion_step(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        t,
        added_cond_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        latents_input = torch.cat([latent] * 2, dim=0)
        latents_input = self.pipeline.scheduler.scale_model_input(latents_input, t)
        if self.is_sdxl:
            noise_pred = self.pipeline.unet(
                latents_input.to(self.device, dtype=self.weight_dtype),
                t,
                encoder_hidden_states=context.to(self.device, dtype=self.weight_dtype),
                added_cond_kwargs={
                    key: value.to(self.device, dtype=self.weight_dtype)
                    for key, value in added_cond_kwargs.items()
                },
                return_dict=False,
            )[0]
        else:
            noise_pred = self.pipeline.unet(
                latents_input.to(self.device, dtype=self.weight_dtype),
                t,
                encoder_hidden_states=context.to(self.device, dtype=self.weight_dtype),
                return_dict=False,
            )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        return self.pipeline.scheduler.step(
            noise_pred,
            t,
            latent.to(self.device, dtype=self.weight_dtype),
            return_dict=False,
        )[0].to(self.latent_dtype)

    @torch.no_grad()
    def _ddim_loop(self, latent: torch.Tensor):
        all_latents = [latent]
        if self.is_sdxl:
            cond_context = self._sdxl_context.chunk(2)[1]
            cond_kwargs = self._sdxl_added_cond_kwargs
        else:
            cond_context = self._base_cond_embeddings
            cond_kwargs = None

        latent_cur = latent.clone().detach()
        timesteps = self.pipeline.scheduler.timesteps
        for i in range(self.source_steps):
            t = timesteps[len(timesteps) - i - 1]
            with self._autocast():
                noise_pred_cond = self._get_noise_pred_single(
                    latent_cur, t, cond_context, cond_kwargs
                )
            latent_cur = self._next_step(noise_pred_cond.to(self.latent_dtype), t, latent_cur)
            all_latents.append(latent_cur)
        return all_latents

    def _null_optimization(
        self,
        latents: List[torch.Tensor],
        num_inner_steps: int,
        epsilon: float,
    ):
        if self.is_sdxl:
            uncond_embeddings, cond_embeddings = self._sdxl_context.chunk(2)
            _ = uncond_embeddings
            added_cond_kwargs = self._sdxl_added_cond_kwargs
            added_uncond_kwargs = self._sdxl_added_uncond_kwargs
            pooled_uncond_embeds = added_uncond_kwargs["text_embeds"].to(
                self.device, dtype=torch.float32
            )
            latent_cur = latents[-1].to(self.device, dtype=self.latent_dtype)
            start_t_idx = self.source_steps - (len(latents) - 1)
            uncond_embeddings_list = []

            for i in range(len(latents) - 1):
                optim_uncond = pooled_uncond_embeds.clone().detach().requires_grad_(True)
                optimizer = torch.optim.Adam(
                    [optim_uncond], lr=9e-2 * (1.0 - i / max(1, self.source_steps))
                )
                latent_prev = latents[len(latents) - i - 2].to(self.device, dtype=self.latent_dtype)
                t = self.pipeline.scheduler.timesteps[start_t_idx + i]

                with torch.no_grad():
                    with self._autocast():
                        noise_pred_cond = self._get_noise_pred_single(
                            latent_cur,
                            t,
                            cond_embeddings,
                            added_cond_kwargs,
                        )

                for j in range(num_inner_steps):
                    optimizer.zero_grad()
                    optim_kwargs = {
                        "text_embeds": optim_uncond.to(self.device, dtype=self.weight_dtype),
                        "time_ids": added_uncond_kwargs["time_ids"].to(
                            self.device, dtype=self.weight_dtype
                        ),
                    }
                    with self._autocast():
                        noise_pred_uncond = self._get_noise_pred_single(
                            latent_cur,
                            t,
                            self._sdxl_context.chunk(2)[0],
                            optim_kwargs,
                        )
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_cond - noise_pred_uncond
                        )
                        latents_prev_rec = self._prev_step(
                            noise_pred.to(self.latent_dtype), t, latent_cur
                        )
                        loss = F.mse_loss(latents_prev_rec, latent_prev)
                    loss.backward()
                    optimizer.step()
                    if float(loss.item()) < epsilon + i * 2e-5:
                        break

                optim_uncond_final = optim_uncond.detach().to(
                    added_cond_kwargs["text_embeds"].dtype
                )
                uncond_embeddings_list.append(optim_uncond_final)

                with torch.no_grad():
                    final_kwargs = {
                        "text_embeds": torch.cat(
                            (
                                optim_uncond_final.to(self.device),
                                added_cond_kwargs["text_embeds"].to(self.device),
                            ),
                            dim=0,
                        ),
                        "time_ids": self._sdxl_all_added_cond_kwargs["time_ids"].to(
                            self.device
                        ),
                    }
                    latent_cur = self._guided_diffusion_step(
                        latent_cur,
                        self._sdxl_context,
                        t,
                        final_kwargs,
                    )
            return uncond_embeddings_list

        uncond_embeddings, cond_embeddings = self._base_context.chunk(2)
        optim_base = uncond_embeddings.to(self.device, dtype=torch.float32)
        latent_cur = latents[-1].to(self.device, dtype=self.latent_dtype)
        start_t_idx = self.source_steps - (len(latents) - 1)
        uncond_embeddings_list = []

        for i in range(len(latents) - 1):
            optim_uncond = optim_base.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam(
                [optim_uncond], lr=1e-2 * (1.0 - i / max(1, self.source_steps))
            )
            latent_prev = latents[len(latents) - i - 2].to(self.device, dtype=self.latent_dtype)
            t = self.pipeline.scheduler.timesteps[start_t_idx + i]

            with torch.no_grad():
                with self._autocast():
                    noise_pred_cond = self._get_noise_pred_single(latent_cur, t, cond_embeddings)

            for j in range(num_inner_steps):
                optimizer.zero_grad()
                with self._autocast():
                    noise_pred_uncond = self._get_noise_pred_single(
                        latent_cur,
                        t,
                        optim_uncond.to(self.device, dtype=self.weight_dtype),
                    )
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                    latents_prev_rec = self._prev_step(
                        noise_pred.to(self.latent_dtype), t, latent_cur
                    )
                    loss = F.mse_loss(latents_prev_rec, latent_prev)
                loss.backward()
                optimizer.step()
                if float(loss.item()) < epsilon + i * 2e-5:
                    break

            optim_uncond_final = optim_uncond[:1].detach().to(cond_embeddings.dtype)
            uncond_embeddings_list.append(optim_uncond_final)

            with torch.no_grad():
                context = torch.cat(
                    (
                        optim_uncond_final.to(self.device),
                        cond_embeddings.to(self.device),
                    ),
                    dim=0,
                )
                latent_cur = self._guided_diffusion_step(latent_cur, context, t)
        return uncond_embeddings_list

    def _prepare_initial_latent(self) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        if self.init_mode == "random":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed)
            latent = torch.randn(
                (
                    1,
                    self.pipeline.unet.config.in_channels,
                    self.height // 8,
                    self.width // 8,
                ),
                generator=generator,
                device="cpu",
                dtype=self.latent_dtype,
            )
            return latent.to(self.device), None

        image = self._load_base_image()
        latent = self._image_to_latent(image)
        ddim_latents = self._ddim_loop(latent)
        ddim_latents = ddim_latents[: self.reverse_steps + 1]
        uncond_embeddings = self._null_optimization(
            ddim_latents,
            num_inner_steps=self.null_inner_steps,
            epsilon=self.null_epsilon,
        )
        return ddim_latents[-1], uncond_embeddings

    def _current_start_latent(self) -> torch.Tensor:
        delta = self.delta
        if self.latent_eps > 0:
            delta = delta.clamp(-self.latent_eps, self.latent_eps)
        return (self.base_latent + delta).to(self.device, dtype=self.latent_dtype)

    @torch.no_grad()
    def _denoise_latents(self, latents: torch.Tensor) -> torch.Tensor:
        self.pipeline.scheduler.set_timesteps(self.source_steps)
        start_t_idx = self.source_steps - self.reverse_steps
        timesteps = self.pipeline.scheduler.timesteps[start_t_idx:]

        latents_cur = latents.to(self.device, dtype=self.latent_dtype)
        if self.is_sdxl:
            for i, t in enumerate(timesteps):
                if self.uncond_embeddings is not None:
                    uncond = self.uncond_embeddings[i]
                else:
                    uncond = self._sdxl_added_uncond_kwargs["text_embeds"]
                added_cond_kwargs = {
                    "text_embeds": torch.cat(
                        (
                            uncond.to(
                                self.device,
                                dtype=self._sdxl_added_cond_kwargs["text_embeds"].dtype,
                            ),
                            self._sdxl_added_cond_kwargs["text_embeds"].to(self.device),
                        ),
                        dim=0,
                    ),
                    "time_ids": self._sdxl_all_added_cond_kwargs["time_ids"].to(self.device),
                }
                with self._autocast():
                    latents_cur = self._guided_diffusion_step(
                        latents_cur,
                        self._sdxl_context,
                        t,
                        added_cond_kwargs,
                    )
            return latents_cur

        for i, t in enumerate(timesteps):
            if self.uncond_embeddings is not None:
                context = torch.cat(
                    (
                        self.uncond_embeddings[i].to(self.device),
                        self._base_cond_embeddings.to(self.device),
                    ),
                    dim=0,
                )
            else:
                context = self._base_context.to(self.device)
            with self._autocast():
                latents_cur = self._guided_diffusion_step(latents_cur, context, t)
        return latents_cur

    def render_patch_for_optimization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            final_latent = self._denoise_latents(self._current_start_latent())
            patch = self._decode_latents(final_latent, output_size=self.patch_size)
        patch = patch.detach().to(self.device, dtype=torch.float32).requires_grad_(True)
        return patch, final_latent.detach()

    @torch.no_grad()
    def render_patch(self) -> torch.Tensor:
        final_latent = self._denoise_latents(self._current_start_latent())
        return self._decode_latents(final_latent, output_size=self.patch_size).to(
            self.device, dtype=torch.float32
        )

    def latent_grad_from_patch_grad(
        self,
        final_latent: torch.Tensor,
        patch_grad: torch.Tensor,
    ) -> torch.Tensor:
        with torch.enable_grad():
            latent = final_latent.detach().clone().to(self.device, dtype=self.latent_dtype)
            latent.requires_grad_(True)

            vae = self.pipeline.vae
            original_dtype = vae.dtype
            decode_latents = latent
            needs_upcast = (
                self.is_sdxl
                and original_dtype == torch.float16
                and getattr(vae.config, "force_upcast", False)
            )

            try:
                if needs_upcast:
                    vae.to(dtype=torch.float32)
                    decode_latents = decode_latents.to(dtype=torch.float32)
                else:
                    decode_latents = decode_latents.to(dtype=vae.dtype)

                autocast_ctx = nullcontext() if needs_upcast else self._autocast()
                with autocast_ctx:
                    vae_input = (decode_latents / vae.config.scaling_factor).to(
                        self.device, dtype=decode_latents.dtype
                    )
                    decoded_patch = vae.decode(vae_input, return_dict=False)[0]

                decoded_patch = decoded_patch.div(2).add(0.5).clamp(0.0, 1.0)
                if decoded_patch.shape[-1] != self.patch_size:
                    decoded_patch = F.interpolate(
                        decoded_patch,
                        size=(self.patch_size, self.patch_size),
                        mode="bilinear",
                        align_corners=False,
                    )

                grad_target = patch_grad.detach().to(self.device, dtype=decoded_patch.dtype)
                loss = torch.sum(decoded_patch * grad_target)
                latent_grad = torch.autograd.grad(
                    loss, latent, retain_graph=False, create_graph=False
                )[0]
            finally:
                if needs_upcast:
                    vae.to(dtype=original_dtype)
        return latent_grad.detach().to(self.device, dtype=self.latent_dtype)

    @torch.no_grad()
    def step(
        self,
        final_latent: torch.Tensor,
        patch_grad: torch.Tensor,
        *,
        optimizer_name: str,
        lr: float,
        max_delta: float,
    ):
        latent_grad = self.latent_grad_from_patch_grad(final_latent, patch_grad)

        if optimizer_name == "adam":
            self._adam_t += 1
            self._adam_m.mul_(0.9).add_(latent_grad, alpha=0.1)
            self._adam_v.mul_(0.999).addcmul_(latent_grad, latent_grad, value=0.001)
            m_hat = self._adam_m / (1.0 - 0.9**self._adam_t)
            v_hat = self._adam_v / (1.0 - 0.999**self._adam_t)
            update = m_hat / (torch.sqrt(v_hat) + 1e-8)
            self.delta.data.sub_(lr * update)
        elif optimizer_name == "sgd":
            self.delta.data.sub_(lr * latent_grad)
        elif optimizer_name == "ifgsm":
            self.delta.data.sub_(lr * latent_grad.sign())
        elif optimizer_name == "clipped-pgd":
            clipped_grad = latent_grad.clamp(min=-max_delta, max=max_delta)
            self.delta.data.sub_(lr * clipped_grad)
        else:
            raise ValueError(
                "Diffusion patch supports only adam, sgd, ifgsm, and clipped-pgd optimizers"
            )

        if self.latent_eps > 0:
            self.delta.data.clamp_(-self.latent_eps, self.latent_eps)


def build_diffusion_patch_generator(args, device) -> Optional[DiffusionPatchGenerator]:
    if getattr(args, "patch_parametrization", "pixel") != "diffusion":
        return None

    return DiffusionPatchGenerator(
        patch_size=args.patch_size,
        model_name=args.diffusion_model,
        model_path=args.diffusion_model_path,
        prompt=args.diffusion_prompt,
        device=device,
        init_mode=args.diffusion_init_mode,
        base_image_path=args.diffusion_base_image,
        source_steps=args.diffusion_source_steps,
        reverse_steps=args.diffusion_reverse_steps,
        guidance_scale=args.diffusion_guidance_scale,
        latent_eps=args.diffusion_latent_eps,
        null_inner_steps=args.diffusion_null_inner_steps,
        null_epsilon=args.diffusion_null_epsilon,
        seed=args.diffusion_seed,
        diffusion_dtype=args.diffusion_dtype,
    )
