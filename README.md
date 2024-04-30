# LCM-Lookahead for Encoder-based Text-to-Image Personalization

[![arXiv](https://img.shields.io/badge/arXiv-2404.03620-b31b1b.svg)](https://arxiv.org/abs/2404.03620)

[[Project Website](https://lcm-lookahead.github.io/)]

> **LCM-Lookahead for Encoder-based Text-to-Image Personalization**<br>
> Rinon Gal<sup>1,2</sup>, Or Lichter<sup>1</sup>, Elad Richardson<sup>1</sup>, Or Patashnik<sup>1</sup>, Amit H. Bermano<sup>1</sup>, Gal Chechik<sup>2</sup>, Daniel Cohen-Or<sup>1</sup> <br>
> <sup>1</sup>Tel Aviv University, <sup>2</sup>NVIDIA

>**Abstract**: <br>
> Recent advancements in diffusion models have introduced fast sampling methods that can effectively produce high-quality images in just one or a few denoising steps. Interestingly, when these are distilled from existing diffusion models, they often maintain alignment with the original model, retaining similar outputs for similar prompts and seeds. These properties present opportunities to leverage fast sampling methods as a shortcut-mechanism, using them to create a preview of denoised outputs through which we can backpropagate image-space losses. In this work, we explore the potential of using such shortcut-mechanisms to guide the personalization of text-to-image models to specific facial identities. We focus on encoder-based personalization approaches, and demonstrate that by tuning them with a lookahead identity loss, we can achieve higher identity fidelity, without sacrificing layout diversity or prompt alignment. We further explore the use of attention sharing mechanisms and consistent data generation for the task of personalization, and find that encoder training can benefit from both.

## Description
This repo contains the official code, data and models for our LCM-Lookahead paper.

## Updates
**30/04/2024** Code released!

## TODO:
- [x] Release code!
- [ ] Add native Diffusers support.
- [ ] Add option to skip shared-attention features for faster inference.
- [ ] Add Insant-ID support?

## Setup

Our code builds on Simo Ryu's excellent [MinSDXL implementation](https://github.com/cloneofsimo/minSDXL). To set up the environment, please run:

```
conda env create -f environment.yaml
conda activate lcm_lh
```

### Prepare pre-trained models

The following model checkpoints should be downloaded and placed somewhere accessible. If you only want to run inference, you only need to download `IP-Adapter CLIP-extractor`.

| Path | Description
| :--- | :----------
|[IR-SE50 Model](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing) | Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in our ID loss and encoder backbone on human facial domain.
|[IP-Adapter Model](https://huggingface.co/h94/IP-Adapter/tree/main/sdxl_models) | Download the ip-adapter-plus-face_sdxl_vit-h.bin checkpoint. Used to initialize the adapter backbone.
|[IP-Adapter CLIP-extractor](https://huggingface.co/h94/IP-Adapter/tree/main/models/image_encoder) | Download the entire directory. Feature extractor used for IP-adapter models. Note: The plus-face_sdxl model uses the non-SDXL clip extractor. Please download the one from the link, and not from the sdxl_models directory.

## Training

### Update configs

Update your loss config file (```config_files/losses.yaml```) to point ```pretrained_arcface_path``` at the IR-SE50 checkpoint.

### Prepare data

See ```config_files/celeba_example.yaml```, ```config_files/generated_single_set_example.yaml```, ```config_files/generated_multi_set_example.yaml``` for examples on how to set up a data configuration file for one or more sets.

We provide a data generation example script in ```scripts/faces_generation.py```. Example pre-generated data is available [here](https://huggingface.co/datasets/rinong/lcm_lookahead).

### Train the model

An example command for training the model:

```
accelerate launch --num_processes <number_of_gpus> --multi_gpu train_encoder.py 
--pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 
--data_config_path config_files/<data_config.yaml> 
--pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix 
--losses_config_path config_files/<loss_config.yaml> 
--ip_adapter_feature_extractor_path <path_to_ip_adapter/image_encoder directory> 
--ip_adapter_model_path <path_to_ip_adapter/ip-adapter-plus-face_sdxl_vit-h.bin> 
--output_dir <path_to_model_output_directory>
--mixed_precision=fp16 
--instance_prompt "a photo of a face" 
--train_batch_size 4 
--learning_rate 5e-6 
--max_train_steps 5001 
--validation_prompt "a painting in the style of monet" 
--seed 0 
--checkpointing_steps 5000 
--checkpoints_total_limit 2 
--optimize_adapter 
--adapter_lr 1e-5 
--lcm_every_k_steps 1 
--lcm_batch_size 4 
--gradient_checkpointing 
--kv_drop_chance 0.05 
--adapter_drop_chance 0.1 
--adam_weight_decay 0.01 
--noisy_encoder_input 
--encoder_lora_rank 4 
--kvcopy_lora_rank 4 
--lcm_sample_scale_every_k_steps 5 
--lcm_sample_full_lcm_prob 0.5
```

Note: If you're training on a single GPU, remove the `--multi_gpu` argument.

### Inference

Our pretrained models are available [here](https://huggingface.co/rinong/lcm_lookahead).

To perform inference, run:

```
python infer.py 
--checkpoint_path <path_to_ckpt_dir>/model_ckpt.pt 
--ip_adapter_feature_extractor_path <path_to_ip_adapter/image_encoder directory>
--input_path <path_to_input_images (file/directory)>
--prompts <list_of_inference_prompts>
--out_dir <path_to_output_dir>
--num_images_per_prompt 1
--use_freeu True
```
See `infer.py` for additional config options. 

Note: If you want to use the model version that doesn't employ extended attention, use the `--no_kv` flag and optionally discard `--use_freeu`.

## Citation

If you make use of our work, please cite our paper:

```
@misc{gal2024lcmlookahead,
    title={LCM-Lookahead for Encoder-based Text-to-Image Personalization}, 
    author={Rinon Gal and Or Lichter and Elad Richardson and Or Patashnik and Amit H. Bermano and Gal Chechik and Daniel Cohen-Or},
    year={2024},
    eprint={2404.03620},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```