# Modified from minSDXL by Simo Ryu: 
# https://github.com/cloneofsimo/minSDXL ,
# which is in turn modified from the original code of:
# https://github.com/huggingface/diffusers
# So has APACHE 2.0 license

import torch
import torch.nn as nn

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
    LoRACompatibleLinear,
    LoRALinearLayer
)

from model.injection_config import get_injection_modules, get_lora_injection_modules, get_kvcopy_lora_injection_modules


def create_custom_forward(module):
    def custom_forward(*inputs):
        return module(*inputs)

    return custom_forward

def get_encoder_trainable_params(encoder):
    trainable_params = []

    for module in encoder.modules():
        if isinstance(module, ExtractKVTransformerBlock):
            # If LORA exists in attn1, train them. Otherwise, attn1 is frozen
            # NOTE: not sure if we want it under a different subset
            if module.attn1.to_k.lora_layer is not None:
                trainable_params.extend(module.attn1.to_k.lora_layer.parameters())
                trainable_params.extend(module.attn1.to_v.lora_layer.parameters())
                trainable_params.extend(module.attn1.to_q.lora_layer.parameters())
                trainable_params.extend(module.attn1.to_out[0].lora_layer.parameters())

            if module.attn2.to_k.lora_layer is not None:
                trainable_params.extend(module.attn2.to_k.lora_layer.parameters())
                trainable_params.extend(module.attn2.to_v.lora_layer.parameters())
                trainable_params.extend(module.attn2.to_q.lora_layer.parameters())
                trainable_params.extend(module.attn2.to_out[0].lora_layer.parameters())

            # If LORAs exist in kvcopy layers, train only them
            if module.extract_kv1.to_k.lora_layer is not None:
                trainable_params.extend(module.extract_kv1.to_k.lora_layer.parameters())
                trainable_params.extend(module.extract_kv1.to_v.lora_layer.parameters())
            else:
                trainable_params.extend(module.extract_kv1.to_k.parameters())
                trainable_params.extend(module.extract_kv1.to_v.parameters())
        
    return trainable_params

def get_adapter_layers(encoder):
    adapter_layers = []
    for module in encoder.modules():
        if isinstance(module, ExtractKVTransformerBlock):
            adapter_layers.append(module.extract_kv2)

    return adapter_layers

def get_adapter_trainable_params(encoder):
    adapter_layers = get_adapter_layers(encoder)
    trainable_params = []
    for layer in adapter_layers:
        trainable_params.extend(layer.to_v.parameters())
        trainable_params.extend(layer.to_k.parameters())

    return trainable_params

def maybe_grad_checkpoint(resnet, attn, hidden_states, temb, encoder_hidden_states, adapter_hidden_states, do_ckpt=True):

    if do_ckpt:
        hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
        hidden_states, extracted_kv = torch.utils.checkpoint.checkpoint(
            create_custom_forward(attn), hidden_states, encoder_hidden_states, adapter_hidden_states, use_reentrant=False
        )
    else:
        hidden_states = resnet(hidden_states, temb)
        hidden_states, extracted_kv = attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            adapter_hidden_states=adapter_hidden_states,
        )
    return hidden_states, extracted_kv


def init_lora_in_attn(attn_module, rank: int = 4, is_kvcopy=False):
    # Set the `lora_layer` attribute of the attention-related matrices.

    attn_module.to_k.set_lora_layer(
        LoRALinearLayer(
            in_features=attn_module.to_k.in_features, out_features=attn_module.to_k.out_features, rank=rank
        )
    )
    attn_module.to_v.set_lora_layer(
        LoRALinearLayer(
            in_features=attn_module.to_v.in_features, out_features=attn_module.to_v.out_features, rank=rank
        )
    )

    if not is_kvcopy:
        attn_module.to_q.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_q.in_features, out_features=attn_module.to_q.out_features, rank=rank
            )
        )

        attn_module.to_out[0].set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                rank=rank,
            )
        )

def drop_kvs(encoder_kvs, drop_chance):
    for layer in encoder_kvs:
        len_tokens = encoder_kvs[layer].self_attention.k.shape[1]
        idx_to_keep = (torch.rand(len_tokens) > drop_chance)

        encoder_kvs[layer].self_attention.k = encoder_kvs[layer].self_attention.k[:, idx_to_keep]
        encoder_kvs[layer].self_attention.v = encoder_kvs[layer].self_attention.v[:, idx_to_keep]

    return encoder_kvs

def clone_kvs(encoder_kvs):
    cloned_kvs = {}
    for layer in encoder_kvs:
        sa_cpy = KVCache(k=encoder_kvs[layer].self_attention.k.clone(), 
                         v=encoder_kvs[layer].self_attention.v.clone())

        ca_cpy = KVCache(k=encoder_kvs[layer].cross_attention.k.clone(),
                         v=encoder_kvs[layer].cross_attention.v.clone())

        cloned_layer_cache = AttentionCache(self_attention=sa_cpy, cross_attention=ca_cpy)
        
        cloned_kvs[layer] = cloned_layer_cache

    return cloned_kvs


class KVCache(object):
    def __init__(self, k, v):
        self.k = k
        self.v = v

class AttentionCache(object):
    def __init__(self, self_attention: KVCache, cross_attention: KVCache):
        self.self_attention = self_attention
        self.cross_attention = cross_attention

class KVCopy(nn.Module):
    def __init__(
        self, inner_dim, source_attention_block, cross_attention_dim=None,
    ):
        super(KVCopy, self).__init__()
        
        self.source_attention_block = source_attention_block

        out_dim = cross_attention_dim or inner_dim

        self.to_k = LoRACompatibleLinear(out_dim, inner_dim, bias=False)
        self.to_v = LoRACompatibleLinear(out_dim, inner_dim, bias=False)

    def forward(self, hidden_states):

        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        return KVCache(k=k, v=v)

    def init_kv_copy(self):
        with torch.no_grad():
            self.to_k.weight.copy_(self.source_attention_block.to_k.weight)
            self.to_v.weight.copy_(self.source_attention_block.to_v.weight)

class ExtractKVTransformerBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ExtractKVTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.attn1 = Attention(hidden_size)
        self.extract_kv1 = KVCopy(hidden_size, self.attn1)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.attn2 = Attention(hidden_size, 2048)
        self.extract_kv2 = KVCopy(hidden_size, self.attn2, 2048)
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.ff = FeedForward(hidden_size, hidden_size)

    def forward(self, x, encoder_hidden_states=None, adapter_hidden_states=None, extract_kv=True):

        residual = x

        x = self.norm1(x)
        if extract_kv:
            kv_out_self = self.extract_kv1(x)
        else:
            kv_out_self = None
            
        x = self.attn1(x)
        x = x + residual

        residual = x

        x = self.norm2(x)
        if extract_kv and adapter_hidden_states is not None:
            kv_out_cross = self.extract_kv2(adapter_hidden_states)
        else:
            kv_out_cross = None
            
        if encoder_hidden_states is not None:
            x = self.attn2(x, encoder_hidden_states)
        else:
            x = self.attn2(x)
        x = x + residual

        residual = x

        x = self.norm3(x)
        x = self.ff(x)
        x = x + residual
        return x, AttentionCache(self_attention=kv_out_self, cross_attention=kv_out_cross)
    
    def init_kv_extraction(self):
        self.extract_kv1.init_kv_copy()
        self.extract_kv2.init_kv_copy()
    
class ExtractKVTransformer2DModel(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers):
        super(ExtractKVTransformer2DModel, self).__init__()
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-06, affine=True)
        self.proj_in = nn.Linear(in_channels, out_channels, bias=True)
        self.transformer_blocks = nn.ModuleList(
            [ExtractKVTransformerBlock(out_channels) for _ in range(n_layers)]
        )
        self.proj_out = nn.Linear(out_channels, out_channels, bias=True)

    def forward(self, hidden_states, encoder_hidden_states=None, adapter_hidden_states=None):
        batch, _, height, width = hidden_states.shape
        res = hidden_states
        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, height * width, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)

        extracted_kvs = {}

        for block in self.transformer_blocks:
            hidden_states, extracted_kv = block(hidden_states, encoder_hidden_states, adapter_hidden_states, extract_kv=hasattr(block, "full_name"))
            
            if extracted_kv:
                extracted_kvs[block.full_name] = extracted_kv

        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, width, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return hidden_states + res, extracted_kvs
    
    def init_kv_extraction(self):
        for block in self.transformer_blocks:
            block.init_kv_extraction()

class ExtractKVCrossAttnDownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, has_downsamplers=True):
        super(ExtractKVCrossAttnDownBlock2D, self).__init__()
        self.attentions = nn.ModuleList(
            [
                ExtractKVTransformer2DModel(out_channels, out_channels, n_layers),
                ExtractKVTransformer2DModel(out_channels, out_channels, n_layers),
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

    def forward(self, hidden_states, temb, encoder_hidden_states, adapter_hidden_states=None):
        output_states = []
        extracted_kvs = {}

        for resnet, attn in zip(self.resnets, self.attentions):

            hidden_states, extracted_kv = maybe_grad_checkpoint(resnet, attn, hidden_states, temb, encoder_hidden_states, adapter_hidden_states, do_ckpt=self.training and self.gradient_checkpointing)

            output_states.append(hidden_states)
            extracted_kvs.update(extracted_kv)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states, extracted_kvs

    def init_kv_extraction(self):
        for block in self.attentions:
            block.init_kv_extraction()

class ExtractKVCrossAttnUpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, prev_output_channel, n_layers):
        super(ExtractKVCrossAttnUpBlock2D, self).__init__()
        self.attentions = nn.ModuleList(
            [
                ExtractKVTransformer2DModel(out_channels, out_channels, n_layers),
                ExtractKVTransformer2DModel(out_channels, out_channels, n_layers),
                ExtractKVTransformer2DModel(out_channels, out_channels, n_layers),
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
        self, hidden_states, res_hidden_states_tuple, temb, encoder_hidden_states, adapter_hidden_states=None
    ):
        extracted_kvs = {}
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states, extracted_kv = maybe_grad_checkpoint(resnet, attn, hidden_states, temb, encoder_hidden_states, adapter_hidden_states, do_ckpt=self.training and self.gradient_checkpointing)

            extracted_kvs.update(extracted_kv)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states, extracted_kvs

    def init_kv_extraction(self):
        for block in self.attentions:
            block.init_kv_extraction()

class ExtractKVUNetMidBlock2DCrossAttn(nn.Module):
    def __init__(self, in_features):
        super(ExtractKVUNetMidBlock2DCrossAttn, self).__init__()
        self.attentions = nn.ModuleList(
            [ExtractKVTransformer2DModel(in_features, in_features, n_layers=10)]
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_features, in_features, conv_shortcut=False),
                ResnetBlock2D(in_features, in_features, conv_shortcut=False),
            ]
        )

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, adapter_hidden_states=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        extracted_kvs = {}
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states, extracted_kv = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                adapter_hidden_states=adapter_hidden_states,
            )
            hidden_states = resnet(hidden_states, temb)

            extracted_kvs.update(extracted_kv)

        return hidden_states, extracted_kvs
    
    def init_kv_extraction(self):
        for block in self.attentions:
            block.init_kv_extraction()


class EmbeddingPredictor(nn.Module):
    def __init__(self, feature_dim, inner_dim, first_encoder_dim, second_encoder_dim):
        super(EmbeddingPredictor, self).__init__()

        self.first_encoder_dim = first_encoder_dim
        self.second_encoder_dim = second_encoder_dim

        self.predictor_layers = nn.Sequential(
            nn.Linear(feature_dim, inner_dim),
            nn.Linear(inner_dim, inner_dim),
            nn.Linear(inner_dim, first_encoder_dim + second_encoder_dim)
        )

    def forward(self, features):
        pooled_features = features.mean(axis=(2,3))
        embeddings = self.predictor_layers(pooled_features)
        return embeddings[:, :self.first_encoder_dim], embeddings[:, self.first_encoder_dim:]


# Modified SDXL unet which serves as our encoder. Outputs extra IP-Adapter attention features, and extra keys and values from running denoising on the input image.
class ExtractKVUNet2DConditionModel(nn.Module):
    def __init__(self, predict_word_embedding=False):
        super(ExtractKVUNet2DConditionModel, self).__init__()

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
                ExtractKVCrossAttnDownBlock2D(in_channels=320, out_channels=640, n_layers=2),
                ExtractKVCrossAttnDownBlock2D(
                    in_channels=640,
                    out_channels=1280,
                    n_layers=10,
                    has_downsamplers=False,
                ),
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                ExtractKVCrossAttnUpBlock2D(
                    in_channels=640,
                    out_channels=1280,
                    prev_output_channel=1280,
                    n_layers=10,
                ),
                ExtractKVCrossAttnUpBlock2D(
                    in_channels=320,
                    out_channels=640,
                    prev_output_channel=1280,
                    n_layers=2,
                ),
                UpBlock2D(in_channels=320, out_channels=320, prev_output_channel=640),
            ]
        )
        self.mid_block = ExtractKVUNetMidBlock2DCrossAttn(1280)
        self.conv_norm_out = nn.GroupNorm(32, 320, eps=1e-05, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, stride=1, padding=1)

        self.predict_word_embedding = predict_word_embedding
        if predict_word_embedding:
            self.embedding_prediction_head = EmbeddingPredictor(1280, 1280, 768, 1280)

        self.name_submodules()

        self.dtype = None

    def forward(
        self, sample, timesteps, encoder_hidden_states, added_cond_kwargs, **kwargs
    ):
        
        encoded_features = {}

        # Implement the forward pass through the model
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        text_embeds = added_cond_kwargs.get("text_embeds")
        time_ids = added_cond_kwargs.get("time_ids")
        adapter_hidden_states = added_cond_kwargs.get("adapter_states")

        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb

        sample = self.conv_in(sample)

        extracted_kvs = {}

        # 3. down
        s0 = sample
        sample, [s1, s2, s3] = self.down_blocks[0](
            sample,
            temb=emb,
        )

        sample, [s4, s5, s6], extracted_kv = self.down_blocks[1](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            adapter_hidden_states=adapter_hidden_states,
        )

        extracted_kvs.update(extracted_kv)

        sample, [s7, s8], extracted_kv = self.down_blocks[2](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            adapter_hidden_states=adapter_hidden_states,
        )

        extracted_kvs.update(extracted_kv)

        # 4. mid
        sample, extracted_kv = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, adapter_hidden_states=adapter_hidden_states,
        )

        if self.predict_word_embedding:
            encoded_features["word_embedding"] = self.embedding_prediction_head(sample)

        extracted_kvs.update(extracted_kv)

        # 5. up
        sample, extracted_kv = self.up_blocks[0](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s6, s7, s8],
            encoder_hidden_states=encoder_hidden_states,
            adapter_hidden_states=adapter_hidden_states,
        )

        extracted_kvs.update(extracted_kv)

        sample, extracted_kv = self.up_blocks[1](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s3, s4, s5],
            encoder_hidden_states=encoder_hidden_states,
            adapter_hidden_states=adapter_hidden_states,
        )

        extracted_kvs.update(extracted_kv)

        encoded_features["kv"] = extracted_kvs
        
        return encoded_features
    
    def init_kv_extraction(self):
        for block in self.down_blocks[1:]:
            block.init_kv_extraction()

        for block in self.up_blocks[:2]:
            block.init_kv_extraction()

        self.mid_block.init_kv_extraction()

    def name_submodules(self):
        injection_modules = get_injection_modules()
        
        for module in injection_modules:
            self.get_submodule(module).full_name = module

    def init_lora_in_encoder(self, rank: int = 4):
        # Init LoRA in encoder
        for module_name in get_lora_injection_modules():
            module = self.get_submodule(module_name)
            init_lora_in_attn(module.attn1, rank=rank)
            init_lora_in_attn(module.attn2, rank=rank)

    def init_lora_in_kvcopy(self, rank: int = 4):
        # Init LoRAs in kvcopy modules
        for module_name in get_kvcopy_lora_injection_modules():
            module = self.get_submodule(module_name)
            init_lora_in_attn(module.extract_kv1, rank=rank, is_kvcopy=True)

    def set_gradient_checkpointing(self, value):
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = value
    
    def get_attention_layers(self, include_self_attention_blocks: bool=False):
        attention_layers = []
        for module in self.modules():
            if isinstance(module, ExtractKVTransformerBlock):
                if include_self_attention_blocks:
                    attention_layers.append(module.attn1)
                attention_layers.append(module.attn2)
        return attention_layers