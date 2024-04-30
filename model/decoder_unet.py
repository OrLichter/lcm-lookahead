# Modified from minSDXL by Simo Ryu: 
# https://github.com/cloneofsimo/minSDXL ,
# which is in turn modified from the original code of:
# https://github.com/huggingface/diffusers
# So has APACHE 2.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from collections import namedtuple

from model.min_sdxl import (
    Timesteps,
    TimestepEmbedding, 
    ResnetBlock2D,
    Attention, 
    FeedForward,
    Downsample2D,
    Upsample2D,
    DownBlock2D,
    UpBlock2D,
    apply_freeu
)

from model.injection_config import get_injection_modules

def create_custom_forward(module):
    def custom_forward(*inputs):
        return module(*inputs)

    return custom_forward

def maybe_grad_checkpoint(resnet, attn, hidden_states, temb, encoder_hidden_states, external_kvs, do_ckpt=True):

    if do_ckpt:
        hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
        hidden_states = torch.utils.checkpoint.checkpoint(
            create_custom_forward(attn), hidden_states, external_kvs, encoder_hidden_states, use_reentrant=False
        )
    else:
        hidden_states = resnet(hidden_states, temb)
        hidden_states = attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            external_kvs=external_kvs,
        )
    return hidden_states

# Concatenated-KV style attention sharing.
class ExpandedKVAttention(Attention):
    def __init__(
        self, inner_dim, cross_attention_dim=None, num_heads=None, dropout=0.0,
    ):
        super(Attention, self).__init__()
        if num_heads is None:
            self.head_dim = 64
            self.num_heads = inner_dim // self.head_dim
        else:
            self.num_heads = num_heads
            self.head_dim = inner_dim // num_heads

        self.scale = self.head_dim**-0.5
        if cross_attention_dim is None:
            cross_attention_dim = inner_dim
        self.to_q = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, inner_dim), nn.Dropout(dropout, inplace=False)]
        )

    def forward(self, hidden_states, external_kv=None, encoder_hidden_states=None):
        q = self.to_q(hidden_states)
        k = (
            self.to_k(encoder_hidden_states)
            if encoder_hidden_states is not None
            else self.to_k(hidden_states)
        )
        v = (
            self.to_v(encoder_hidden_states)
            if encoder_hidden_states is not None
            else self.to_v(hidden_states)
        )

        if external_kv:
            k = torch.cat([k, external_kv.k], axis=1)
            v = torch.cat([v, external_kv.v], axis=1)

        b, t, c = q.size()

        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, c)

        for layer in self.to_out:
            attn_output = layer(attn_output)

        return attn_output

# IP-Adapter style additive attention layers
class AdditiveKVAttention(Attention):
    def __init__(
        self, inner_dim, cross_attention_dim=None, num_heads=None, dropout=0.0,
    ):
        super(Attention, self).__init__()
        if num_heads is None:
            self.head_dim = 64
            self.num_heads = inner_dim // self.head_dim
        else:
            self.num_heads = num_heads
            self.head_dim = inner_dim // num_heads

        self.scale = self.head_dim**-0.5
        if cross_attention_dim is None:
            cross_attention_dim = inner_dim
        self.to_q = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, inner_dim), nn.Dropout(dropout, inplace=False)]
        )

        self.additive_scale = 1.0

    def forward(self, hidden_states, external_kv=None, encoder_hidden_states=None):
        q = self.to_q(hidden_states)
        k = (
            self.to_k(encoder_hidden_states)
            if encoder_hidden_states is not None
            else self.to_k(hidden_states)
        )
        v = (
            self.to_v(encoder_hidden_states)
            if encoder_hidden_states is not None
            else self.to_v(hidden_states)
        )

        b, t, c = q.size()

        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, c)

        if external_kv:
            k = external_kv.k
            v = external_kv.v

            k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

            external_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale,
            )

            external_output = external_output.transpose(1, 2).contiguous().view(b, t, c)
            
            attn_output = attn_output + self.additive_scale * external_output

        for layer in self.to_out:
            attn_output = layer(attn_output)

        return attn_output
    
class ExpandedKVTransformerBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ExpandedKVTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.attn1 = ExpandedKVAttention(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.attn2 = AdditiveKVAttention(hidden_size, 2048)
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.ff = FeedForward(hidden_size, hidden_size)

    def forward(self, x, external_kv=None, encoder_hidden_states=None):
        residual = x

        x = self.norm1(x)
        x = self.attn1(x, external_kv.self_attention)
        x = x + residual

        residual = x

        x = self.norm2(x)
        if encoder_hidden_states is not None:
            x = self.attn2(x, encoder_hidden_states=encoder_hidden_states, external_kv=external_kv.cross_attention)
        else:
            x = self.attn2(x)
        x = x + residual

        residual = x

        x = self.norm3(x)
        x = self.ff(x)
        x = x + residual
        return x

class ExpandedKVTransformer2DModel(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers):
        super(ExpandedKVTransformer2DModel, self).__init__()
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-06, affine=True)
        self.proj_in = nn.Linear(in_channels, out_channels, bias=True)
        self.transformer_blocks = nn.ModuleList(
            [ExpandedKVTransformerBlock(out_channels) for _ in range(n_layers)]
        )
        self.proj_out = nn.Linear(out_channels, out_channels, bias=True)

    def forward(self, hidden_states, external_kvs, encoder_hidden_states=None):
        batch, _, height, width = hidden_states.shape
        res = hidden_states
        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, height * width, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)
        
        for block in self.transformer_blocks:
            if hasattr(block, "full_name"):
                external_kv = external_kvs[block.full_name]
            else:
                external_kv = None

            hidden_states = block(hidden_states, external_kv, encoder_hidden_states)

        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, width, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return hidden_states + res

class ExpandedKVCrossAttnDownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, has_downsamplers=True):
        super(ExpandedKVCrossAttnDownBlock2D, self).__init__()
        self.attentions = nn.ModuleList(
            [
                ExpandedKVTransformer2DModel(out_channels, out_channels, n_layers),
                ExpandedKVTransformer2DModel(out_channels, out_channels, n_layers),
            ]
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_channels, out_channels),
                ResnetBlock2D(out_channels, out_channels, conv_shortcut=False),
            ]
        )
        self.downsamplers = None
        if has_downsamplers:
            self.downsamplers = nn.ModuleList(
                [Downsample2D(out_channels, out_channels)]
            )

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb, encoder_hidden_states, external_kvs):
        output_states = []
        for resnet, attn in zip(self.resnets, self.attentions):

            hidden_states = maybe_grad_checkpoint(resnet, attn, hidden_states, temb, encoder_hidden_states, external_kvs, do_ckpt=self.training and self.gradient_checkpointing)

            output_states.append(hidden_states)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states

class ExpandedKVCrossAttnUpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, prev_output_channel, n_layers):
        super(ExpandedKVCrossAttnUpBlock2D, self).__init__()
        self.attentions = nn.ModuleList(
            [
                ExpandedKVTransformer2DModel(out_channels, out_channels, n_layers),
                ExpandedKVTransformer2DModel(out_channels, out_channels, n_layers),
                ExpandedKVTransformer2DModel(out_channels, out_channels, n_layers),
            ]
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(prev_output_channel + out_channels, out_channels),
                ResnetBlock2D(2 * out_channels, out_channels),
                ResnetBlock2D(out_channels + in_channels, out_channels),
            ]
        )
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels)])

        self.gradient_checkpointing = False

    def forward(
        self, hidden_states, res_hidden_states_tuple, temb, encoder_hidden_states, external_kvs,
    ):
        
        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
            and getattr(self, "resolution_idx", None)
        )
                
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )
            
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = maybe_grad_checkpoint(resnet, attn, hidden_states, temb, encoder_hidden_states, external_kvs, do_ckpt=self.training and self.gradient_checkpointing)


        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states

class ExpandedKVUNetMidBlock2DCrossAttn(nn.Module):
    def __init__(self, in_features):
        super(ExpandedKVUNetMidBlock2DCrossAttn, self).__init__()
        self.attentions = nn.ModuleList(
            [ExpandedKVTransformer2DModel(in_features, in_features, n_layers=10)]
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_features, in_features, conv_shortcut=False),
                ResnetBlock2D(in_features, in_features, conv_shortcut=False),
            ]
        )

    def forward(self, hidden_states, external_kvs, temb=None, encoder_hidden_states=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                external_kvs=external_kvs,
            )
            hidden_states = resnet(hidden_states, temb)

        return hidden_states
 

# Modified SDXL unet which serves as our decoder. Takes extra attention inputs from an IP-Adapter, and extra keys and values from an "encoder" version of the u-net.
# This class should contain no trainable parameters (we calculate all projections in the encoder u-net to better separate the trained and un-trained classes, and to make checkpoints easier to handle).
class ExpandedKVUNet2DConditionModel(nn.Module):
    def __init__(self):
        super(ExpandedKVUNet2DConditionModel, self).__init__()

        # This is needed to imitate huggingface config behavior
        # has nothing to do with the model itself
        # remove this if you don't use diffuser's pipeline
        self.config = namedtuple(
            "config", "in_channels addition_time_embed_dim sample_size"
        )
        self.config.in_channels = 4
        self.config.addition_time_embed_dim = 256
        self.config.sample_size = 128

        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, stride=1, padding=1)
        self.time_proj = Timesteps()
        self.time_embedding = TimestepEmbedding(in_features=320, out_features=1280)
        self.add_time_proj = Timesteps(256)
        self.add_embedding = TimestepEmbedding(in_features=2816, out_features=1280)
        self.down_blocks = nn.ModuleList(
            [
                DownBlock2D(in_channels=320, out_channels=320),
                ExpandedKVCrossAttnDownBlock2D(in_channels=320, out_channels=640, n_layers=2),
                ExpandedKVCrossAttnDownBlock2D(
                    in_channels=640,
                    out_channels=1280,
                    n_layers=10,
                    has_downsamplers=False,
                ),
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                ExpandedKVCrossAttnUpBlock2D(
                    in_channels=640,
                    out_channels=1280,
                    prev_output_channel=1280,
                    n_layers=10,
                ),
                ExpandedKVCrossAttnUpBlock2D(
                    in_channels=320,
                    out_channels=640,
                    prev_output_channel=1280,
                    n_layers=2,
                ),
                UpBlock2D(in_channels=320, out_channels=320, prev_output_channel=640),
            ]
        )
        self.mid_block = ExpandedKVUNetMidBlock2DCrossAttn(1280)
        self.conv_norm_out = nn.GroupNorm(32, 320, eps=1e-05, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, stride=1, padding=1)

        self.name_submodules()

        self.dtype = None


    def forward(
        self, sample, timesteps, encoder_hidden_states, added_cond_kwargs, external_kvs, controlnet_residuals=None, **kwargs
    ):
        # Implement the forward pass through the model
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        text_embeds = added_cond_kwargs.get("text_embeds")
        time_ids = added_cond_kwargs.get("time_ids")

        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb

        sample = self.conv_in(sample)

        # 3. down
        s0 = sample
        sample, [s1, s2, s3] = self.down_blocks[0](
            sample,
            temb=emb,
        )

        sample, [s4, s5, s6] = self.down_blocks[1](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            external_kvs=external_kvs,
        )

        sample, [s7, s8] = self.down_blocks[2](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            external_kvs=external_kvs,
        )

        if controlnet_residuals:
            s0 = s0 + controlnet_residuals["down"][0]
            s1 = s1 + controlnet_residuals["down"][1]
            s2 = s2 + controlnet_residuals["down"][2]
            s3 = s3 + controlnet_residuals["down"][3]
            s4 = s4 + controlnet_residuals["down"][4]
            s5 = s5 + controlnet_residuals["down"][5]
            s6 = s6 + controlnet_residuals["down"][6]
            s7 = s7 + controlnet_residuals["down"][7]
            s8 = s8 + controlnet_residuals["down"][8]


        # 4. mid
        sample = self.mid_block(
            sample, external_kvs, emb, encoder_hidden_states=encoder_hidden_states,
        )

        if controlnet_residuals:
            sample = sample + controlnet_residuals["mid"]

        # 5. up
        sample = self.up_blocks[0](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s6, s7, s8],
            encoder_hidden_states=encoder_hidden_states,
            external_kvs=external_kvs,
        )

        sample = self.up_blocks[1](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s3, s4, s5],
            encoder_hidden_states=encoder_hidden_states,
            external_kvs=external_kvs,
        )

        sample = self.up_blocks[2](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s0, s1, s2],
        )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return [sample]

    def name_submodules(self):
        injection_modules = get_injection_modules()
        
        for module in injection_modules:
            self.get_submodule(module).full_name = module

    def set_adapter_attention_scale(self, scale):
        for module in self.modules():
            if isinstance(module, AdditiveKVAttention):
                module.additive_scale = scale

    def set_gradient_checkpointing(self, value):
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = value
    
    def get_attention_layers(self, include_self_attention_blocks: bool=False):
        attention_layers = []
        for module in self.modules():
            if isinstance(module, ExpandedKVTransformerBlock):
                if include_self_attention_blocks:
                    attention_layers.append(module.attn1)
                attention_layers.append(module.attn2)
        return attention_layers
    
    def enable_freeu(self, s1, s2, b1, b2):
        r"""Enables the FreeU mechanism from https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stage blocks where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of values that
        are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        for i, upsample_block in enumerate(self.up_blocks):
            setattr(upsample_block, "s1", s1)
            setattr(upsample_block, "s2", s2)
            setattr(upsample_block, "b1", b1)
            setattr(upsample_block, "b2", b2)
            setattr(upsample_block, "resolution_idx", i)

    def disable_freeu(self):
        """Disables the FreeU mechanism."""
        freeu_keys = {"s1", "s2", "b1", "b2", "resolution_idx"}
        for i, upsample_block in enumerate(self.up_blocks):
            for k in freeu_keys:
                if hasattr(upsample_block, k) or getattr(upsample_block, k, None) is not None:
                    setattr(upsample_block, k, None)