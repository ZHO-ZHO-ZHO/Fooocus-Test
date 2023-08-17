import os
import random
import einops
import torch
import numpy as np

import comfy.model_management
import comfy.utils

from comfy.sd import load_checkpoint_guess_config
from nodes import VAEDecode, EmptyLatentImage, CLIPTextEncode, ControlNetApply
from comfy.sample import prepare_mask, broadcast_cond, load_additional_models, cleanup_additional_models
from modules.samplers_advanced import KSampler, KSamplerWithRefiner
from modules.patch import patch_all


patch_all()
opCLIPTextEncode = CLIPTextEncode()
opEmptyLatentImage = EmptyLatentImage()
opVAEDecode = VAEDecode()
opControlNetApply = ControlNetApply()


class StableDiffusionModel:
    def __init__(self, unet, vae, clip, clip_vision):
        self.unet = unet
        self.vae = vae
        self.clip = clip
        self.clip_vision = clip_vision

    def to_meta(self):
        if self.unet is not None:
            self.unet.model.to('meta')
        if self.clip is not None:
            self.clip.cond_stage_model.to('meta')
        if self.vae is not None:
            self.vae.first_stage_model.to('meta')

class ControlNet(ControlBase):
    def __init__(self, control_model, global_average_pooling=False, device=None):
        super().__init__(device)
        self.control_model = control_model
        self.global_average_pooling = global_average_pooling

    def get_control(self, x_noisy, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return {}

        output_dtype = x_noisy.dtype
        if self.cond_hint is None or x_noisy.shape[2] * 8 != self.cond_hint.shape[2] or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            self.cond_hint = utils.common_upscale(self.cond_hint_original, x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, 'nearest-exact', "center").to(self.control_model.dtype).to(self.device)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)

        if self.control_model.dtype == torch.float16:
            precision_scope = torch.autocast
        else:
            precision_scope = contextlib.nullcontext

        with precision_scope(model_management.get_autocast_device(self.device)):
            self.control_model = model_management.load_if_low_vram(self.control_model)
            context = torch.cat(cond['c_crossattn'], 1)
            y = cond.get('c_adm', None)
            control = self.control_model(x=x_noisy, hint=self.cond_hint, timesteps=t, context=context, y=y)
            self.control_model = model_management.unload_if_low_vram(self.control_model)
        out = {'middle':[], 'output': []}
        autocast_enabled = torch.is_autocast_enabled()

        for i in range(len(control)):
            if i == (len(control) - 1):
                key = 'middle'
                index = 0
            else:
                key = 'output'
                index = i
            x = control[i]
            if self.global_average_pooling:
                x = torch.mean(x, dim=(2, 3), keepdim=True).repeat(1, 1, x.shape[2], x.shape[3])

            x *= self.strength
            if x.dtype != output_dtype and not autocast_enabled:
                x = x.to(output_dtype)

            if control_prev is not None and key in control_prev:
                prev = control_prev[key][index]
                if prev is not None:
                    x += prev
            out[key].append(x)
        if control_prev is not None and 'input' in control_prev:
            out['input'] = control_prev['input']
        return out

    def copy(self):
        c = ControlNet(self.control_model, global_average_pooling=self.global_average_pooling)
        self.copy_to(c)
        return c

    def get_models(self):
        out = super().get_models()
        out.append(self.control_model)
        return out


def load_controlnet(ckpt_path, model=None):
    controlnet_data = utils.load_torch_file(ckpt_path, safe_load=True)

    controlnet_config = None
    if "controlnet_cond_embedding.conv_in.weight" in controlnet_data: #diffusers format
        use_fp16 = model_management.should_use_fp16()
        controlnet_config = model_detection.model_config_from_diffusers_unet(controlnet_data, use_fp16).unet_config
        diffusers_keys = utils.unet_to_diffusers(controlnet_config)
        diffusers_keys["controlnet_mid_block.weight"] = "middle_block_out.0.weight"
        diffusers_keys["controlnet_mid_block.bias"] = "middle_block_out.0.bias"

        count = 0
        loop = True
        while loop:
            suffix = [".weight", ".bias"]
            for s in suffix:
                k_in = "controlnet_down_blocks.{}{}".format(count, s)
                k_out = "zero_convs.{}.0{}".format(count, s)
                if k_in not in controlnet_data:
                    loop = False
                    break
                diffusers_keys[k_in] = k_out
            count += 1

        count = 0
        loop = True
        while loop:
            suffix = [".weight", ".bias"]
            for s in suffix:
                if count == 0:
                    k_in = "controlnet_cond_embedding.conv_in{}".format(s)
                else:
                    k_in = "controlnet_cond_embedding.blocks.{}{}".format(count - 1, s)
                k_out = "input_hint_block.{}{}".format(count * 2, s)
                if k_in not in controlnet_data:
                    k_in = "controlnet_cond_embedding.conv_out{}".format(s)
                    loop = False
                diffusers_keys[k_in] = k_out
            count += 1

        new_sd = {}
        for k in diffusers_keys:
            if k in controlnet_data:
                new_sd[diffusers_keys[k]] = controlnet_data.pop(k)

        controlnet_data = new_sd

    pth_key = 'control_model.zero_convs.0.0.weight'
    pth = False
    key = 'zero_convs.0.0.weight'
    if pth_key in controlnet_data:
        pth = True
        key = pth_key
        prefix = "control_model."
    elif key in controlnet_data:
        prefix = ""
    else:
        net = load_t2i_adapter(controlnet_data)
        if net is None:
            print("error checkpoint does not contain controlnet or t2i adapter data", ckpt_path)
        return net

    if controlnet_config is None:
        use_fp16 = model_management.should_use_fp16()
        controlnet_config = model_detection.model_config_from_unet(controlnet_data, prefix, use_fp16).unet_config
    controlnet_config.pop("out_channels")
    controlnet_config["hint_channels"] = 3
    control_model = cldm.ControlNet(**controlnet_config)

    if pth:
        if 'difference' in controlnet_data:
            if model is not None:
                m = model.patch_model()
                model_sd = m.state_dict()
                for x in controlnet_data:
                    c_m = "control_model."
                    if x.startswith(c_m):
                        sd_key = "diffusion_model.{}".format(x[len(c_m):])
                        if sd_key in model_sd:
                            cd = controlnet_data[x]
                            cd += model_sd[sd_key].type(cd.dtype).to(cd.device)
                model.unpatch_model()
            else:
                print("WARNING: Loaded a diff controlnet without a model. It will very likely not work.")

        class WeightsLoader(torch.nn.Module):
            pass
        w = WeightsLoader()
        w.control_model = control_model
        missing, unexpected = w.load_state_dict(controlnet_data, strict=False)
    else:
        missing, unexpected = control_model.load_state_dict(controlnet_data, strict=False)
    print(missing, unexpected)

    if use_fp16:
        control_model = control_model.half()

    global_average_pooling = False
    if ckpt_path.endswith("_shuffle.pth") or ckpt_path.endswith("_shuffle.safetensors") or ckpt_path.endswith("_shuffle_fp16.safetensors"): #TODO: smarter way of enabling global_average_pooling
        global_average_pooling = True

    control = ControlNet(control_model, global_average_pooling=global_average_pooling)
    return control


@torch.no_grad()
def load_model(ckpt_filename):
    unet, clip, vae, clip_vision = load_checkpoint_guess_config(ckpt_filename)
    return StableDiffusionModel(unet=unet, clip=clip, vae=vae, clip_vision=clip_vision)


@torch.no_grad()
def load_lora(model, lora_filename, strength_model=1.0, strength_clip=1.0):
    if strength_model == 0 and strength_clip == 0:
        return model

    lora = comfy.utils.load_torch_file(lora_filename, safe_load=True)
    unet, clip = comfy.sd.load_lora_for_models(model.unet, model.clip, lora, strength_model, strength_clip)
    return StableDiffusionModel(unet=unet, clip=clip, vae=model.vae, clip_vision=model.clip_vision)

def load_controlnet(controlnet_filename):

    controlnet = comfy.sd.load_controlnet(controlnet_filename)
    return (controlnet,)

@torch.no_grad()
def encode_prompt_condition(clip, prompt):
    return opCLIPTextEncode.encode(clip=clip, text=prompt)[0]


@torch.no_grad()
def generate_empty_latent(width=1024, height=1024, batch_size=1):
    return opEmptyLatentImage.generate(width=width, height=height, batch_size=batch_size)[0]

@torch.no_grad()
def CN_condition(conditioning, control_net, image, strength):
    return opControlNetApply.apply_controlnet(conditioning=conditioning, control_net=control_net, image=image, strength=strength)[0]

conditioning
control_net
image
strength

@torch.no_grad()
def decode_vae(vae, latent_image):
    return opVAEDecode.decode(samples=latent_image, vae=vae)[0]


def get_previewer(device, latent_format):
    from latent_preview import TAESD, TAESDPreviewerImpl
    taesd_decoder_path = os.path.abspath(os.path.realpath(os.path.join("models", "vae_approx",
                                                                       latent_format.taesd_decoder_name)))

    if not os.path.exists(taesd_decoder_path):
        print(f"Warning: TAESD previews enabled, but could not find {taesd_decoder_path}")
        return None

    taesd = TAESD(None, taesd_decoder_path).to(device)

    def preview_function(x0, step, total_steps):
        global cv2_is_top
        with torch.no_grad():
            x_sample = taesd.decoder(torch.nn.functional.avg_pool2d(x0, kernel_size=(2, 2))).detach() * 255.0
            x_sample = einops.rearrange(x_sample, 'b c h w -> b h w c')
            x_sample = x_sample.cpu().numpy().clip(0, 255).astype(np.uint8)
            return x_sample[0]

    taesd.preview = preview_function

    return taesd


@torch.no_grad()
def ksampler(model, positive, negative, latent, seed=None, steps=30, cfg=7.0, sampler_name='dpmpp_2m_sde_gpu',
             scheduler='karras', denoise=1.0, disable_noise=False, start_step=None, last_step=None,
             force_full_denoise=False, callback_function=None):
    # SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
    # SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
    #             "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
    #             "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddim", "uni_pc", "uni_pc_bh2"]

    seed = seed if isinstance(seed, int) else random.randint(1, 2 ** 64)

    device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    previewer = get_previewer(device, model.model.latent_format)

    pbar = comfy.utils.ProgressBar(steps)

    def callback(step, x0, x, total_steps):
        y = None
        if previewer and step % 3 == 0:
            y = previewer.preview(x0, step, total_steps)
        if callback_function is not None:
            callback_function(step, x0, x, total_steps, y)
        pbar.update_absolute(step + 1, total_steps, None)

    sigmas = None
    disable_pbar = False

    if noise_mask is not None:
        noise_mask = prepare_mask(noise_mask, noise.shape, device)

    comfy.model_management.load_model_gpu(model)
    real_model = model.model

    noise = noise.to(device)
    latent_image = latent_image.to(device)

    positive_copy = broadcast_cond(positive, noise.shape[0], device)
    negative_copy = broadcast_cond(negative, noise.shape[0], device)

    models = load_additional_models(positive, negative, model.model_dtype())

    sampler = KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler,
                       denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image,
                             start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise,
                             denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar,
                             seed=seed)

    samples = samples.cpu()

    cleanup_additional_models(models)

    out = latent.copy()
    out["samples"] = samples

    return out


@torch.no_grad()
def ksampler_with_refiner(model, positive, negative, refiner, refiner_positive, refiner_negative, latent,
                          seed=None, steps=30, refiner_switch_step=20, cfg=7.0, sampler_name='dpmpp_2m_sde_gpu',
                          scheduler='karras', denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                          force_full_denoise=False, callback_function=None):
    # SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
    # SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
    #             "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
    #             "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddim", "uni_pc", "uni_pc_bh2"]

    seed = seed if isinstance(seed, int) else random.randint(1, 2 ** 64)

    device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    previewer = get_previewer(device, model.model.latent_format)

    pbar = comfy.utils.ProgressBar(steps)

    def callback(step, x0, x, total_steps):
        y = None
        if previewer and step % 3 == 0:
            y = previewer.preview(x0, step, total_steps)
        if callback_function is not None:
            callback_function(step, x0, x, total_steps, y)
        pbar.update_absolute(step + 1, total_steps, None)

    sigmas = None
    disable_pbar = False

    if noise_mask is not None:
        noise_mask = prepare_mask(noise_mask, noise.shape, device)

    comfy.model_management.load_model_gpu(model)

    noise = noise.to(device)
    latent_image = latent_image.to(device)

    positive_copy = broadcast_cond(positive, noise.shape[0], device)
    negative_copy = broadcast_cond(negative, noise.shape[0], device)

    refiner_positive_copy = broadcast_cond(refiner_positive, noise.shape[0], device)
    refiner_negative_copy = broadcast_cond(refiner_negative, noise.shape[0], device)

    models = load_additional_models(positive, negative, model.model_dtype())

    sampler = KSamplerWithRefiner(model=model, refiner_model=refiner, steps=steps, device=device,
                                  sampler=sampler_name, scheduler=scheduler,
                                  denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(noise, positive_copy, negative_copy, refiner_positive=refiner_positive_copy,
                             refiner_negative=refiner_negative_copy, refiner_switch_step=refiner_switch_step,
                             cfg=cfg, latent_image=latent_image,
                             start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise,
                             denoise_mask=noise_mask, sigmas=sigmas, callback_function=callback, disable_pbar=disable_pbar,
                             seed=seed)

    samples = samples.cpu()

    cleanup_additional_models(models)

    out = latent.copy()
    out["samples"] = samples

    return out


@torch.no_grad()
def image_to_numpy(x):
    return [np.clip(255. * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]
