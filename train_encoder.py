import logging
import os
import shutil
from pathlib import Path
from typing import List
import diffusers
import math
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import wandb
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from collections import namedtuple

from functools import partial

import diffusers
from diffusers import (
    DiffusionPipeline,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    AutoencoderTiny,
    LCMScheduler
)
from model.diffusers_vae.autoencoder_kl import AutoencoderKL

from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from losses.loss_config import LossesConfig
from losses.losses import *

from data.data_config import DataConfig
from data.dataset import ImageDataset, collate_fn
from model.decoder_unet import ExpandedKVUNet2DConditionModel
from model.encoder_unet import ExtractKVUNet2DConditionModel, get_encoder_trainable_params, get_adapter_trainable_params, drop_kvs
from pipelines.sdxl_encoder_pipeline_with_adapter import StableDiffusionXLEncoderPipeline as StableDiffusionXLEncoderPipelineWithAdapter
from schedulers.lcm_single_step_scheduler import LCMSingleStepScheduler
from utils import vis_utils
from utils.parser import parse_args
from utils.text_utils import tokenize_prompt, encode_prompt, add_tokens, patch_embedding_forward
from utils.utils import verify_load

from model.ipadapter.ipadapter import IPAdapterPlusXL, IPAdapterFaceIDPlusXL

logger = get_logger(__name__)

def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def importance_sampling_fn(t, max_t, alpha):
    """Importance Sampling Function f(t)"""
    return 1 / max_t * (1 - alpha * math.cos(math.pi * t / max_t))

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
        cpu=args.run_on_cpu
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )

    if args.predict_word_embedding:
        new_token_indices = add_tokens([tokenizer_one, tokenizer_two], ["<s*>"], [text_encoder_one, text_encoder_two])

    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )

    orig_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    orig_unet_config = orig_unet.config

    encoder_unet = ExtractKVUNet2DConditionModel(args.predict_word_embedding)
    missing_keys, unexpected_keys = encoder_unet.load_state_dict(orig_unet.state_dict(), strict=False)
    verify_load(missing_keys, unexpected_keys)
    encoder_unet.init_kv_extraction()

    if args.encoder_lora_rank > 0:
        encoder_unet.init_lora_in_encoder(rank=args.encoder_lora_rank)
    if args.kvcopy_lora_rank > 0:
        encoder_unet.init_lora_in_kvcopy(rank=args.kvcopy_lora_rank)

    decoder_unet = ExpandedKVUNet2DConditionModel()
    decoder_unet.load_state_dict(orig_unet.state_dict())

    del orig_unet
    
    if args.lcm_every_k_steps > 0:
        pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, variant=args.variant)
        pipe.load_lora_weights(args.pretrained_lcm_lora_path)
        pipe.fuse_lora()
        lcm_unet = ExpandedKVUNet2DConditionModel()
        lcm_unet.load_state_dict(pipe.unet.state_dict())
        lcm_scheduler = LCMSingleStepScheduler.from_config(pipe.scheduler.config)

        vae_for_lcm = AutoencoderTiny.from_pretrained("madebyollin/taesdxl")

        if args.lcm_sample_scale_every_k_steps > 0:
            if args.lcm_sample_scale_every_k_steps % args.lcm_every_k_steps != 0:
                raise ValueError("Sample scale should be a multiple of lcm steps")
            # Move unet to GPU so that changing scale will be faster, comment if OOM
            pipe.unet.to(accelerator.device)
        else:
            del pipe

    adapter = IPAdapterPlusXL(encoder_unet, 
                            args.ip_adapter_feature_extractor_path,
                            args.ip_adapter_model_path,
                            accelerator.device,
                            num_tokens=args.ip_adapter_tokens)
    
    adapter.requires_grad_(False)
    decoder_unet.set_adapter_attention_scale(args.adapter_attention_scale)
    adapter.to(accelerator.device)

    if args.save_only_encoder:
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for model in models:
                    if isinstance(model, type(accelerator.unwrap_model(adapter))):
                        torch.save(model.state_dict(), os.path.join(output_dir, "model_ckpt.pt"))

                    weights.pop()

        def load_model_hook(models, input_dir):
            
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                if isinstance(model, type(accelerator.unwrap_model(adapter))):
                    adapter.load_state_dict(torch.load(os.path.join(input_dir, "model_ckpt.pt")), strict=True)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    decoder_unet.requires_grad_(False)
    encoder_unet.requires_grad_(False)

    decoder_unet.set_gradient_checkpointing(args.gradient_checkpointing)
    encoder_unet.set_gradient_checkpointing(args.gradient_checkpointing)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    encoder_unet.to(accelerator.device)
    decoder_unet.to(accelerator.device, dtype=weight_dtype)

    if args.lcm_every_k_steps > 0:
        lcm_unet.requires_grad_(False)
        vae_for_lcm.requires_grad_(False)

        lcm_unet.to(accelerator.device, dtype=weight_dtype)
        vae_for_lcm.to(accelerator.device, dtype=weight_dtype)
        if args.gradient_checkpointing:
            lcm_unet.set_gradient_checkpointing(args.gradient_checkpointing)
            vae_for_lcm.apply(partial(vae._set_gradient_checkpointing, value=True))

    vae.to(accelerator.device, dtype=weight_dtype)
    vae.decoder.mid_block.to(torch.float32)

    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    
    params_to_optimize = []
    
    if not args.freeze_encoder_unet:
        encoder_params_to_train = get_encoder_trainable_params(encoder_unet)

        for param in encoder_params_to_train:
            param.requires_grad_(True)

        params_to_optimize.append({"params": encoder_params_to_train, "lr": args.learning_rate})
    else:
        encoder_params_to_train = None

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Optimization parameters

    if args.optimize_adapter:
        adapter_params = adapter.get_trainable_params() + get_adapter_trainable_params(encoder_unet)

        for param in adapter_params:
            param.requires_grad_(True)

        adapter_lr = args.adapter_lr if args.adapter_lr is not None else args.learning_rate

        params_to_optimize.append({"params": adapter_params, "lr": adapter_lr})

    # Optimizer creation
    optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        params_to_optimize,
        betas=(0.9, 0.999),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Initialize Losses
    losses_configs: LossesConfig = pyrallis.load(LossesConfig, open(args.losses_config_path, "r"))
    diffusion_losses = list()
    lcm_losses = list()
    for loss_config in losses_configs.diffusion_losses:
        loss = namedtuple("loss", ["loss", "weight"])
        diffusion_losses.append(loss(loss=eval(loss_config.name)(visualize_every_k=loss_config.visualize_every_k, 
                                                       dtype=weight_dtype,
                                                       accelerator=accelerator,
                                                       **loss_config.init_params), weight=loss_config.weight))
    for loss_config in losses_configs.lcm_losses:
        loss = namedtuple("loss", ["loss", "weight"])
        loss_class = eval(loss_config.name)
        lcm_losses.append(loss(loss=loss_class(visualize_every_k=loss_config.visualize_every_k, 
                                                       dtype=weight_dtype,
                                                       accelerator=accelerator,
                                                       **loss_config.init_params), weight=loss_config.weight))

    # Dataset and DataLoaders creation:
    data_config: DataConfig = pyrallis.load(DataConfig, open(args.data_config_path, "r"))
    datasets = []
    train_prompts = set()
    for single_dataset in data_config.datasets:
        image_dataset = ImageDataset(
            instance_data_root=single_dataset.data_root,
            instance_prompt=args.instance_prompt,
            metadata_path=single_dataset.metadata_path,
            prompt_in_filename=single_dataset.prompt_in_filename,
            use_only_vanilla_for_encoder=single_dataset.use_only_vanilla_for_encoder,
            size=args.resolution,
            center_crop=args.center_crop,
            aug_images=single_dataset.aug_images,
            concept_placeholder=single_dataset.concept_placeholder,
            use_only_decoder_prompts=single_dataset.use_only_decoder_prompts,
            crop_head_for_encoder_image=data_config.crop_head_for_encoder_image,
            random_target_prob=data_config.random_target_prob,
            )
        datasets.append(image_dataset)
        train_prompts.update(image_dataset.prompts_set)
    print(f'Train prompts: {train_prompts}')
    if data_config.val_dataset is not None:
        val_dataset = ImageDataset(
            instance_data_root=data_config.val_dataset.data_root,
            instance_prompt=args.instance_prompt,
            metadata_path=data_config.val_dataset.metadata_path,
            prompt_in_filename=data_config.val_dataset.prompt_in_filename,
            use_only_vanilla_for_encoder=data_config.val_dataset.use_only_vanilla_for_encoder,
            size=args.resolution,
            center_crop=args.center_crop,
            use_only_decoder_prompts=data_config.val_dataset.use_only_decoder_prompts,
            crop_head_for_encoder_image=data_config.crop_head_for_encoder_image,
        )

    sampler_train = None
    if len(datasets) == 1:
        train_dataset = datasets[0]
    else:
        # Give equal weights to all datasets
        train_dataset = torch.utils.data.ConcatDataset(datasets)
        if data_config.balance_datasets:
            dataset_weights = []
            for single_dataset in datasets:
                dataset_weights.extend([len(train_dataset) / len(single_dataset)] * len(single_dataset))
            sampler_train = torch.utils.data.WeightedRandomSampler(weights=dataset_weights,
                                                                   num_samples=len(dataset_weights))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=sampler_train,
        shuffle=True if sampler_train is None else False,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers
    )

    # Computes additional embeddings/ids required by the SDXL UNet.
    # regular text embeddings (when `train_text_encoder` is not True)
    # pooled text embeddings
    # time ids

    def compute_time_ids():
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        original_size = (args.resolution, args.resolution)
        target_size = (args.resolution, args.resolution)
        crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    # Handle instance prompt.
    instance_time_ids = compute_time_ids()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.
    add_time_ids = instance_time_ids

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    encoder_unet, decoder_unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler, vae, adapter = accelerator.prepare(
        encoder_unet, decoder_unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler, vae, adapter
    )

    if args.lcm_every_k_steps > 0:
        lcm_unet = accelerator.prepare(lcm_unet)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        wandb_kwargs = {"resume": "allow"}

        config = vars(args)
        losses_configs_dictionary_to_log = {k: [vars(vi) for vi in v] for k, v in vars(losses_configs).items()}
        config.update(losses_configs_dictionary_to_log)

        accelerator.init_trackers("encoder_sdxl", config=config, init_kwargs={"wandb": wandb_kwargs})
        if args.experiment_name is not None:
            accelerator.get_tracker("wandb").run.name = accelerator.get_tracker("wandb").run.name + " | " + args.experiment_name

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    args.resume_from_checkpoint = args.resume_from_checkpoint

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

            accelerator.skip_first_batches(train_dataloader, global_step)

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    trainable_models = [encoder_unet]
    if args.optimize_adapter:
        trainable_models.append(adapter)

    if args.gradient_checkpointing:
        checkpoint_models = [vae]
    else:
        checkpoint_models = []

    # Potential scaling of the LCM loss, will be updated when lora scale changes
    lcm_loss_scale = 1.0

    for epoch in range(first_epoch, args.num_train_epochs):

        for step, batch in enumerate(train_dataloader):
            for model in trainable_models + checkpoint_models:
                model.train()                

            with accelerator.accumulate(*trainable_models):
                loss = torch.tensor(0.0)

                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                encoder_pixel_values = batch["encoder_pixel_values"].to(dtype=vae.dtype)

                prompts = batch["prompts"]
                encoder_prompts = batch["encoder_prompts"]

                if args.text_drop_chance:
                    indices_to_drop = np.random.randint(len(prompts)) < args.text_drop_chance
                    prompts[indices_to_drop] = ""
                    encoder_prompts[indices_to_drop] = ""

                @torch.no_grad()
                def convert_to_latent(pixels):
                    # Convert images to latent space
                    model_input = vae.encode(pixels).latent_dist.sample()
                    model_input = model_input * vae.config.scaling_factor
                    if args.pretrained_vae_model_name_or_path is None:
                        model_input = model_input.to(weight_dtype)
                    return model_input

                model_input = convert_to_latent(pixel_values)
                encoder_model_input = convert_to_latent(encoder_pixel_values)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)

                list_of_candidates = [
                    x for x in range(noise_scheduler.config.num_train_timesteps)
                ]
                prob_dist = [
                    importance_sampling_fn(x,
                                        noise_scheduler.config.num_train_timesteps,
                                        0.5)
                    for x in list_of_candidates
                ]
                prob_sum = 0
                # normalize the prob_list so that sum of prob is 1
                for i in prob_dist:
                    prob_sum += i
                prob_dist = [x / prob_sum for x in prob_dist]

                timesteps = np.random.choice(
                    list_of_candidates,
                    size=bsz,
                    replace=True,
                    p=prob_dist)
                timesteps = torch.tensor(timesteps).cuda()


                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                noisy_encoder_model_input = noise_scheduler.add_noise(encoder_model_input, noise, timesteps)

                encoder_input = noisy_encoder_model_input if args.noisy_encoder_input else encoder_model_input

                # Calculate the elements to repeat depending on the use of prior-preservation and custom captions.
                elems_to_repeat_text_embeds = 1
                elems_to_repeat_time_ids = bsz

                def process_conditions(prompts):
                    unet_added_conditions = {"time_ids": add_time_ids.repeat(elems_to_repeat_time_ids, 1)}
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=None,
                        prompt=None,
                        text_input_ids_list=[tokenize_prompt(tokenizer_one, prompts),
                                             tokenize_prompt(tokenizer_two, prompts)],
                    )
                    unet_added_conditions.update(
                        {"text_embeds": pooled_prompt_embeds.repeat(elems_to_repeat_text_embeds, 1)}
                    )
                    prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                    return prompt_embeds_input, unet_added_conditions

                encoder_prompt_embeds_input, encoder_unet_added_conditions = process_conditions(encoder_prompts)

                adapter_states_cond, adapter_states_uncond = adapter(encoder_pixel_values)
                encoder_unet_added_conditions["adapter_states"] = adapter_states_cond

                if args.adapter_drop_chance > 0:
                    idx_to_replace = torch.rand(len(adapter_states_cond)) < args.adapter_drop_chance
                    encoder_unet_added_conditions["adapter_states"][idx_to_replace] = adapter_states_uncond[idx_to_replace]

                # NOTE: Changed from noisy image to the original latents

                encoder_outputs = encoder_unet(
                    encoder_input, timesteps, encoder_prompt_embeds_input, added_cond_kwargs=encoder_unet_added_conditions
                )

                if "word_embedding" in encoder_outputs:
                    for text_encoder_idx, text_encoder in enumerate([text_encoder_one, text_encoder_two]):
                        patch_embedding_forward(text_encoder.get_input_embeddings(), new_token_indices[f"{text_encoder_idx}_<s*>"], encoder_outputs["word_embedding"][text_encoder_idx])

                prompt_embeds_input, unet_added_conditions = process_conditions(prompts)

                if args.kv_drop_chance > 0.0:
                    encoder_outputs["kv"] = drop_kvs(encoder_outputs["kv"], args.kv_drop_chance)

                model_pred = decoder_unet(
                    noisy_model_input, timesteps, prompt_embeds_input, external_kvs=encoder_outputs["kv"],
                    added_cond_kwargs=unet_added_conditions
                )[0]

                loss_arguments = {
                    "encoder_pixel_values": encoder_pixel_values,
                    "target_noise": noise,
                    "predicted_noise": model_pred,
                    "encoder_extracted_kvs": encoder_outputs["kv"],
                    "encoder_prompt_embeddings_input": prompt_embeds_input,
                    "decoder_prompt_embeddings_input": prompt_embeds_input,
                    "timesteps": timesteps,
                }

                loss_dict = dict()
                for loss_config in diffusion_losses:
                    non_weighted_loss = loss_config.loss(**loss_arguments, accelerator=accelerator)
                    loss = loss + non_weighted_loss * loss_config.weight
                    loss_dict[loss_config.loss.__class__.__name__] = non_weighted_loss.item()

                accelerator.backward(loss)
                if accelerator.sync_gradients and encoder_params_to_train is not None :
                    params_to_clip = (encoder_params_to_train)
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Use LCM 
                if args.lcm_every_k_steps > 0 and step % args.lcm_every_k_steps == 0:
                    if args.lcm_sample_scale_every_k_steps > 0 and step % args.lcm_sample_scale_every_k_steps == 0:
                        # Change the scale
                        pipe.unfuse_lora()
                        pipe.load_lora_weights(args.pretrained_lcm_lora_path)
                        if np.random.rand() < args.lcm_sample_full_lcm_prob:
                            new_lora_scale = 1.0
                        else:
                            new_lora_scale = np.random.uniform(args.lcm_min_scale, 1.0)
                        print(f'Iteration {step}, new lora scale: {new_lora_scale}')
                        pipe.fuse_lora(lora_scale=new_lora_scale)
                        # lcm_unet = ExpandedKVUNet2DConditionModel()
                        # lcm_unet.requires_grad_(False)
                        # lcm_unet.to(accelerator.device, dtype=weight_dtype)
                        # if args.gradient_checkpointing:
                        #     lcm_unet.set_gradient_checkpointing(args.gradient_checkpointing)

                        lcm_unet.load_state_dict(pipe.unet.state_dict())
                        lcm_loss_scale = new_lora_scale

                    loss = 0.0

                    elems_to_repeat_time_ids = args.lcm_batch_size
                    pixel_values = pixel_values[:args.lcm_batch_size]
                    encoder_pixel_values = encoder_pixel_values[:args.lcm_batch_size]
                    prompts = prompts[:args.lcm_batch_size]
                    encoder_prompts = encoder_prompts[:args.lcm_batch_size]
                    noise = noise[:args.lcm_batch_size]
                    model_input = model_input[:args.lcm_batch_size]
                    encoder_model_input = encoder_model_input[:args.lcm_batch_size]

                    # timesteps = torch.randint(0, args.lcm_max_timestep, (args.lcm_batch_size,), device=model_input.device)

                    list_of_candidates = [
                        x for x in range(args.lcm_max_timestep)
                    ]
                    prob_dist = [
                        importance_sampling_fn(x,
                                            args.lcm_max_timestep,
                                            0.2)
                        for x in list_of_candidates
                    ]
                    prob_sum = 0
                    # normalize the prob_list so that sum of prob is 1
                    for i in prob_dist:
                        prob_sum += i
                    prob_dist = [x / prob_sum for x in prob_dist]

                    timesteps = np.random.choice(
                        list_of_candidates,
                        size=args.lcm_batch_size,
                        replace=True,
                        p=prob_dist)
                    timesteps = torch.tensor(timesteps).cuda()

                    timesteps = timesteps.long()

                    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                    noisy_encoder_model_input = noise_scheduler.add_noise(encoder_model_input, noise, timesteps)

                    encoder_prompt_embeds_input, encoder_unet_added_conditions = process_conditions(encoder_prompts)

                    adapter_states_cond, adapter_states_uncond = adapter(encoder_pixel_values)
                    encoder_unet_added_conditions["adapter_states"] = adapter_states_cond

                    encoder_input = noisy_encoder_model_input if args.noisy_encoder_input else encoder_model_input

                    encoder_outputs = encoder_unet(encoder_input, timesteps, encoder_prompt_embeds_input, added_cond_kwargs=encoder_unet_added_conditions)

                    if args.kv_drop_chance > 0.0:
                        encoder_outputs["kv"] = drop_kvs(encoder_outputs["kv"], args.kv_drop_chance)

                    if "word_embedding" in encoder_outputs:
                        for text_encoder_idx, text_encoder in enumerate([text_encoder_one, text_encoder_two]):
                            patch_embedding_forward(text_encoder.get_input_embeddings(), new_token_indices[f"{text_encoder_idx}_<s*>"], encoder_outputs["word_embedding"][text_encoder_idx])
                    
                    prompt_embeds_input, unet_added_conditions = process_conditions(prompts)

                    loss_arguments = {
                        "encoder_pixel_values": encoder_pixel_values,
                        "target_noise": noise,
                        "encoder_extracted_kvs": encoder_outputs["kv"],
                        "encoder_prompt_embeddings_input": prompt_embeds_input,
                        "decoder_prompt_embeddings_input": prompt_embeds_input,
                        "timesteps": timesteps,
                        "encoder_added_conditions": unet_added_conditions,
                        "encoder_unet": encoder_unet,
                        "noise_scheduler": noise_scheduler,
                    }

                    lcm_noise_pred = lcm_unet(
                        noisy_model_input,
                        timesteps,
                        prompt_embeds_input,
                        external_kvs=encoder_outputs["kv"],
                        added_cond_kwargs=unet_added_conditions,
                    )[0]
                    lcm_noise_pred = lcm_noise_pred.to(dtype=weight_dtype)

                    denoised_latents = lcm_scheduler.step(
                        lcm_noise_pred,
                        timesteps,
                        noisy_model_input,
                        return_dict=False
                    )

                    loss_arguments["predicted_latents"] = denoised_latents

                    denoised_latents = denoised_latents.to(dtype=vae_for_lcm.dtype)

                    predicted_pixel_values = vae_for_lcm.decode(
                        denoised_latents / vae_for_lcm.config.scaling_factor,
                        return_dict=False
                    )[0]

                    predicted_pixel_values = predicted_pixel_values.to(dtype=weight_dtype)
                    loss_arguments["predicted_pixel_values"] = predicted_pixel_values
                    for loss_config in lcm_losses:
                        non_weighted_loss = loss_config.loss(**loss_arguments, accelerator=accelerator)
                        loss = loss + non_weighted_loss * loss_config.weight * lcm_loss_scale
                        loss_dict[f"lcm_{loss_config.loss.__class__.__name__}"] = non_weighted_loss.item()

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = []
                        for param_block in params_to_optimize:
                            params_to_clip.extend(
                                param_block["params"]
                            )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                        accelerator.save_state(save_path, safe_serialization=False)

                        logger.info(f"Saved state to {save_path}")                          

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            logs.update(loss_dict)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
            
            if accelerator.is_main_process and accelerator.sync_gradients:
                is_train_vis_iter = (global_step - 1) % args.train_vis_steps == 0
                is_val_vis_iter = (global_step - 1) % args.validation_vis_steps == 0 and data_config.val_dataset is not None
                if is_train_vis_iter or is_val_vis_iter:
                    logger.info(
                        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                        f" {args.validation_prompt}."
                    )

                    for model in trainable_models + checkpoint_models:
                        model.eval()

                    # create pipeline

                    pipeline = StableDiffusionXLEncoderPipelineWithAdapter.from_pretrained(
                        args.pretrained_model_name_or_path,
                        vae=accelerator.unwrap_model(vae),
                        text_encoder=accelerator.unwrap_model(text_encoder_one),
                        text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                        encoder_unet=accelerator.unwrap_model(encoder_unet),
                        decoder_unet=accelerator.unwrap_model(decoder_unet),
                        unet_config=orig_unet_config,
                        adapter=accelerator.unwrap_model(adapter),
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                    )

                    if args.vis_lcm:
                        pipeline_lcm = StableDiffusionXLEncoderPipelineWithAdapter.from_pretrained(
                            args.pretrained_model_name_or_path,
                            vae=accelerator.unwrap_model(vae),
                            text_encoder=accelerator.unwrap_model(text_encoder_one),
                            text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                            encoder_unet=accelerator.unwrap_model(encoder_unet),
                            decoder_unet=accelerator.unwrap_model(lcm_unet),
                            unet_config=orig_unet_config,
                            adapter=accelerator.unwrap_model(adapter),
                            revision=args.revision,
                            variant=args.variant,
                            torch_dtype=weight_dtype,
                        )

                        pipeline_lcm.scheduler = LCMScheduler.from_config(pipeline_lcm.scheduler.config)
                        pipeline_lcm = pipeline_lcm.to(accelerator.device)
                        pipeline_lcm.set_progress_bar_config(disable=True)

                    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
                    scheduler_args = {}

                    if "variance_type" in pipeline.scheduler.config:
                        variance_type = pipeline.scheduler.config.variance_type

                        if variance_type in ["learned", "learned_range"]:
                            variance_type = "fixed_small"

                        scheduler_args["variance_type"] = variance_type

                    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                        pipeline.scheduler.config, **scheduler_args
                    )

                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                    # run inference

                    def run_inference(target_prompts, cond_prompts, cond_images, pipeline, num_steps=50, disable_cfg=False) -> List[Image.Image]:
                        generator = torch.Generator(device=accelerator.device).manual_seed(
                            args.seed) if args.seed else None
                        with torch.cuda.amp.autocast():
                            images = []
                            for val_idx in tqdm(range(len(target_prompts))):
                                pipeline_args = {"prompt": target_prompts[val_idx],
                                                 "encoder_prompt": cond_prompts[val_idx]}
                                if args.predict_word_embedding:
                                    pipeline_args["new_token_indices"] = new_token_indices
                                
                                pipeline_args["noisy_encoder_input"] = args.noisy_encoder_input

                                images.append(
                                    pipeline(**pipeline_args, cond_images=cond_images[val_idx:val_idx + 1], num_inference_steps=num_steps,
                                             generator=generator).images[0]
                                )
                        return images
                    
                    def vis_train(pipeline, num_steps, disable_cfg=False):
                        train_result_figure = []
                        n_vis = args.num_train_vis_images
                        train_images_on_train_prompts = run_inference(target_prompts=prompts[:n_vis],
                                                                      cond_prompts=encoder_prompts[:n_vis],
                                                                      cond_images=encoder_pixel_values[:n_vis],
                                                                      pipeline=pipeline,
                                                                      num_steps=num_steps,
                                                                      disable_cfg=disable_cfg)
                        val_prompts = list(np.random.choice(list(data_config.validation_prompts), n_vis, replace=True))
                        train_images_on_val_prompts = run_inference(target_prompts=val_prompts,
                                                                    cond_prompts=encoder_prompts[:n_vis],
                                                                    cond_images=encoder_pixel_values[:n_vis],
                                                                    pipeline=pipeline,
                                                                    num_steps=num_steps,
                                                                    disable_cfg=disable_cfg)
                        encoder_images = pipeline.image_processor.postprocess(encoder_pixel_values[:n_vis],
                                                                              output_type="pil")
                        target_images = pipeline.image_processor.postprocess(
                            pixel_values[:n_vis], output_type="pil")

                        # Order into plot
                        for img_idx in range(n_vis):
                            titles = ["Encoder Input", "Train Target", "Train Prompt", "Val Prompt"]
                            images = [encoder_images[img_idx], target_images[img_idx],
                                      train_images_on_train_prompts[img_idx], train_images_on_val_prompts[img_idx]]
                            captions = [encoder_prompts[img_idx], prompts[img_idx], prompts[img_idx],
                                        val_prompts[img_idx]]
                            train_result_figure.append(
                                vis_utils.create_table_plot(titles=titles, images=images, captions=captions))

                        for img_idx, image in enumerate(train_result_figure):
                            image.save(os.path.join(args.output_dir, f"output_{img_idx}.jpg"))

                        return train_result_figure
                    
                    def vis_val(pipeline, num_steps, disable_cfg=False):
                        val_result_figure = []
                        n_vis = args.num_validation_images
                        # First indices are fixed, last one is random
                        val_inds = list(range(n_vis - 1)) + [np.random.randint(n_vis, len(val_dataset))]
                        val_pixel_values = torch.stack([val_dataset[i]["encoder_images"] for i in val_inds]).to(pixel_values.device).to(pixel_values.dtype)
                        vanilla_prompts = [f'A photo of {val_dataset.concept_placeholder}'] * n_vis
                        val_images_on_vanilla_prompts = run_inference(target_prompts=vanilla_prompts,
                                                                      cond_prompts=vanilla_prompts,
                                                                      cond_images=val_pixel_values,
                                                                      pipeline=pipeline,
                                                                      num_steps=num_steps,
                                                                      disable_cfg=disable_cfg)

                        random_train_prompts = list(np.random.choice(list(train_prompts), n_vis, replace=True))
                        val_images_on_train_prompts = run_inference(target_prompts=random_train_prompts,
                                                                    cond_prompts=vanilla_prompts,
                                                                    cond_images=val_pixel_values,
                                                                    pipeline=pipeline,
                                                                    num_steps=num_steps,
                                                                    disable_cfg=disable_cfg)

                        val_prompts = list(np.random.choice(list(data_config.validation_prompts), n_vis, replace=True))
                        val_images_on_val_prompts = run_inference(target_prompts=val_prompts,
                                                                  cond_prompts=vanilla_prompts,
                                                                  cond_images=val_pixel_values,
                                                                  pipeline=pipeline,
                                                                  num_steps=num_steps,
                                                                  disable_cfg=disable_cfg)
                        val_images = pipeline.image_processor.postprocess(val_pixel_values, output_type="pil")

                        # Order into plot
                        for img_idx in range(n_vis):
                            titles = ["Encoder Input", "Vanilla Prompt", "Train Prompt", "Val Prompt"]
                            images = [val_images[img_idx], val_images_on_vanilla_prompts[img_idx],
                                      val_images_on_train_prompts[img_idx], val_images_on_val_prompts[img_idx]]
                            captions = [vanilla_prompts[img_idx], vanilla_prompts[img_idx],
                                        random_train_prompts[img_idx],
                                        val_prompts[img_idx]]
                            val_result_figure.append(
                                vis_utils.create_table_plot(titles=titles, images=images, captions=captions))

                        for img_idx, image in enumerate(val_result_figure):
                            image.save(os.path.join(args.output_dir, f"output_val_{img_idx}.jpg"))

                        return val_result_figure

                    if is_train_vis_iter:
                        train_result_figure = vis_train(pipeline, num_steps=50)

                    if is_val_vis_iter:
                        val_result_figure = vis_val(pipeline, num_steps=50)

                    if args.vis_lcm:
                        if is_train_vis_iter:
                            train_lcm_figure = vis_train(pipeline_lcm, num_steps=4, disable_cfg=True)
                        if is_val_vis_iter:
                            val_lcm_figure = vis_val(pipeline_lcm, num_steps=4, disable_cfg=True)

                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = np.stack([np.asarray(img) for img in train_result_figure])
                            tracker.writer.add_images("train", np_images, epoch, dataformats="NHWC")
                            np_images = np.stack([np.asarray(img) for img in val_result_figure])
                            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                        if tracker.name == "wandb":
                            tracker_dict = {}
                            if is_train_vis_iter:
                                tracker_dict["vis/train_images"] = [
                                        wandb.Image(image)
                                        for i, image in enumerate(train_result_figure)
                                    ]
                            if is_val_vis_iter:
                                tracker_dict["vis/val_images"] = [
                                    wandb.Image(image)
                                    for i, image in enumerate(val_result_figure)
                                ]

                            if args.vis_lcm:
                                if is_train_vis_iter:
                                    tracker_dict["vis/train_images_lcm"] = [
                                            wandb.Image(image)
                                            for i, image in enumerate(train_lcm_figure)
                                        ]
                                if is_val_vis_iter:
                                    tracker_dict["vis/val_images_lcm"] = [
                                        wandb.Image(image)
                                        for i, image in enumerate(val_lcm_figure)
                                    ]

                            tracker.log(tracker_dict)

                    del pipeline
                    torch.cuda.empty_cache()

    # Save the accelerator state
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.save_state(save_path, safe_serialization=False)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
