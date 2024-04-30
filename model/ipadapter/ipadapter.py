# Code adapted from https://github.com/tencent-ailab/IP-Adapter
#
# Original code is licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import List
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from insightface.app import FaceAnalysis
from insightface.utils import face_align

from model.encoder_unet import get_adapter_layers
from .resampler import PerceiverAttention, Resampler, FeedForward

class FacePerceiverResampler(torch.nn.Module):
    def __init__(
        self,
        *,
        dim=768,
        depth=4,
        dim_head=64,
        heads=16,
        embedding_dim=1280,
        output_dim=768,
        ff_mult=4,
    ):
        super().__init__()
        
        self.proj_in = torch.nn.Linear(embedding_dim, dim)
        self.proj_out = torch.nn.Linear(dim, output_dim)
        self.norm_out = torch.nn.LayerNorm(output_dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, latents, x):
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)

class ProjPlusModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, clip_embeddings_dim=1280, num_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=cross_attention_dim // 64,
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )
        
    def forward(self, id_embeds, clip_embeds, shortcut=False, scale=1.0):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        out = self.perceiver_resampler(x, clip_embeds)
        if shortcut:
            out = x + scale * out
        return out

class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens
    
class IPAdapter(torch.nn.Module):
    def __init__(self, encoder_unet, image_encoder_path, ip_ckpt, device, num_tokens=4, decoder_unet=None, lcm_unet=None, lora_scale=1.0):
        super(IPAdapter, self).__init__()

        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.cross_attention_dim = 2048
        self.encoder_unet = encoder_unet
        self.dtype = torch.float16
        
        self.decoder_unet = decoder_unet
        self.lcm_unet = lcm_unet
        self.lora_scale = lora_scale

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=self.dtype
        )
        self.clip_image_processor = CLIPImageProcessor()

        # image proj model
        self.image_proj_model = self.init_proj()

        if self.ip_ckpt is not None:
            self.load_ip_adapter()


    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device)
        return image_proj_model

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")

        adapter_keys = list(state_dict["ip_adapter"].keys())
        for key in adapter_keys:
            if "_ip" in key: 

                new_key = key.replace("_ip", "")
                idx, layer, weights = new_key.split(".")
                new_idx = str((int(idx) - 1) // 2)
                new_key = f"{new_idx}.{layer}.{weights}"

                state_dict["ip_adapter"][new_key] = state_dict["ip_adapter"][key]
                del state_dict["ip_adapter"][key]

        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList([layer for layer in get_adapter_layers(self.encoder_unet)])
        missing, unexpected = ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

        filtered_missing = [key for key in missing if "source_attention" not in key]
        
        if len(filtered_missing) > 0 or len(unexpected) > 0:
            raise RuntimeError(f"Error(s) in loading state_dict for ModuleList:\n Missing keys: {filtered_missing}\n Unexpected keys: {unexpected}")

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            pil_image = (pil_image + 1) // 2
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    def forward(self, pil_image, **kwargs):
        return self.get_image_embeds(pil_image, **kwargs)
    
    def get_trainable_params(self):
        return list(self.image_proj_model.parameters())

def numpy_to_pil(images):
    """
    Convert a NumPy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image[:, :, :3]) for image in images]

    return pil_images

class IPAdapterPlusXL(IPAdapter):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.cross_attention_dim,
            ff_mult=4,
        ).to(self.device)
        return image_proj_model

    def get_image_embeds(self, pil_image, skip_uncond=False):

        with torch.no_grad():
            pil_image = pil_image.to(dtype=torch.float32)
            pil_image = ((pil_image + 1) / 2).clamp(0, 1)

            pil_image = numpy_to_pil(pil_image.cpu().permute(0, 2, 3, 1).float().numpy())        
            
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values

            clip_image = clip_image.to(self.device, dtype=self.dtype)
            clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
            
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)

        if skip_uncond:
            return image_prompt_embeds
        
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds
        
class IPAdapterFaceIDPlusXL(IPAdapter):
    def __init__(self, encoder_unet, image_encoder_path, ip_ckpt, device, num_tokens=4, decoder_unet=None, lcm_unet=None, lora_scale=1.0):
        super().__init__(encoder_unet, image_encoder_path, ip_ckpt, device, num_tokens, decoder_unet, lcm_unet, lora_scale)
        self.face_analysis = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
        self.face_embedding_shape = 512
    
    def init_proj(self):
        image_proj_model = ProjPlusModel(
            cross_attention_dim=self.cross_attention_dim,
            id_embeddings_dim=512,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=self.dtype)
        return image_proj_model

    def get_image_embeds(self, torch_images):
        with torch.no_grad():
            torch_images = torch_images.to(dtype=torch.float32)
            torch_images = ((torch_images + 1) / 2).clamp(0, 1)
            numpy_images = torch_images.cpu().permute(0, 2, 3, 1).float().numpy()

            face_ids = []
            cropped_images = []
            for image in numpy_images:
                import cv2
                faces = self.face_analysis.get((image * 255).astype(np.uint8))
                if len(faces) == 0:
                    faceid_embeds = torch.zeros(self.face_embedding_shape).unsqueeze(0)
                    cropped_images.append(cv2.resize(image, (224, 224)))
                else:
                    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
                    cropped_images.append(face_align.norm_crop(image, landmark=faces[0].kps, image_size=224))
                faceid_embeds.to(self.device, dtype=self.dtype)
                face_ids.append(faceid_embeds)
            faceid_embeds = torch.cat(face_ids, dim=0).to(self.device, dtype=self.dtype)
            
            pil_images = numpy_to_pil(np.stack(cropped_images))

            clip_image = self.clip_image_processor(images=pil_images, return_tensors="pt").pixel_values
            clip_image = clip_image.to(self.device, dtype=self.dtype)
            clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
            uncond_clip_image_embeds = self.image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[-2]

        image_prompt_embeds = self.image_proj_model(faceid_embeds, clip_image_embeds, shortcut=True, scale=1.0)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(faceid_embeds), uncond_clip_image_embeds, shortcut=True, scale=1.0)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        
        # Fuse LoRA params, then init KV extraction lastly override kv extraction with ip_adapter
        fused_params = self.fuse_lora_params(state_dict["ip_adapter"], self.encoder_unet)
        self.encoder_unet.init_kv_extraction()
        
        adapter_keys = list(state_dict["ip_adapter"].keys())        
        for key in adapter_keys:
            if "_ip" in key: 

                new_key = key.replace("_ip", "")
                idx, layer, weights = new_key.split(".")
                new_idx = str((int(idx) - 1) // 2)
                new_key = f"{new_idx}.{layer}.{weights}"

                state_dict["ip_adapter"][new_key] = state_dict["ip_adapter"][key]
                del state_dict["ip_adapter"][key]

        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList([layer for layer in get_adapter_layers(self.encoder_unet)])
        missing, unexpected = ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)
                
        filtered_missing = [key for key in missing if "source_attention" not in key]
        unexpected_not_lora = [key for key in unexpected if "_lora" not in key]
        
        if len(filtered_missing) > 0 or len(unexpected_not_lora) > 0:
            raise RuntimeError(f"Error(s) in loading state_dict for ModuleList:\n Missing keys: {filtered_missing}\n Unexpected keys: {unexpected}")

        if self.decoder_unet is not None:
            fused_params = self.fuse_lora_params(state_dict["ip_adapter"], self.decoder_unet)
            assert set(fused_params) == set(unexpected), f"Didn't fuse all lora keys, fused {len(fused_params)} out of {len(unexpected)} keys"
        
        if self.lcm_unet is not None:
            fused_params = self.fuse_lora_params(state_dict["ip_adapter"], self.lcm_unet)
            assert set(fused_params) == set(unexpected), f"Didn't fuse all lora keys, fused {len(fused_params)} out of {len(unexpected)} keys"

    def fuse_lora_params(self, all_params: dict, unet: torch.nn.Module):
        adapter_layers = unet.get_attention_layers(include_self_attention_blocks=True)
        fused_params = []
        for i, layer in tqdm(enumerate(adapter_layers), desc="Fusing IPAdapterFaceIDPlusXL LORA params"):
            for attention_key in ["k", "q", "v"]:
                up_key = f"{i}.to_{attention_key}_lora.up.weight"
                down_key = f"{i}.to_{attention_key}_lora.down.weight"
                lora_weights = self.lora_scale * all_params[up_key].to(dtype=torch.float32) @ all_params[down_key].to(dtype=torch.float32)
                layer_attention = getattr(layer, f"to_{attention_key}") 
                layer_attention.weight.data = layer_attention.weight.data + lora_weights.to(device=layer_attention.weight.device, dtype=layer_attention.weight.dtype)
                fused_params.append(up_key)
                fused_params.append(down_key)
            # handle out
            up_key = f"{i}.to_out_lora.up.weight"
            down_key = f"{i}.to_out_lora.down.weight"
            lora_weights = self.lora_scale * all_params[up_key].to(dtype=torch.float32) @ all_params[down_key].to(dtype=torch.float32)
            layer.to_out[0].weight.data = layer.to_out[0].weight.data + lora_weights.to(device=layer.to_out[0].weight.device, dtype=layer.to_out[0].weight.dtype)
            fused_params.append(up_key)
            fused_params.append(down_key)
        return fused_params
    
    