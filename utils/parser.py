import argparse
import os

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Train Consistency Encoder.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    
    # parser.add_argument(
    #     "--instance_data_dir",
    #     type=str,
    #     required=True,
    #     help=("A folder containing the training data. "),
    # )

    parser.add_argument(
        "--data_config_path",
        type=str,
        required=True,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )

    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_train_vis_images",
        type=int,
        default=2,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )

    parser.add_argument(
        "--validation_vis_steps",
        type=int,
        default=500,
        help=(
            "Run dreambooth validation every X steps. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )

    parser.add_argument(
        "--train_vis_steps",
        type=int,
        default=500,
        help=(
            "Run dreambooth validation every X steps. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )

    parser.add_argument(
        "--vis_lcm",
        type=bool,
        default=True,
        help=(
            "Also log results of LCM inference",
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--save_only_encoder", action="store_true", help="Only save the encoder and not the full accelerator state")

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument("--freeze_encoder_unet", action="store_true", help="Don't train encoder unet")
    parser.add_argument("--predict_word_embedding", action="store_true", help="Predict word embeddings in addition to KV features")
    parser.add_argument("--ip_adapter_feature_extractor_path", type=str, help="Path to pre-trained feature extractor for IP-adapter")
    parser.add_argument("--ip_adapter_model_path", type=str, help="Path to pre-trained IP-adapter.")
    parser.add_argument("--ip_adapter_tokens", type=int, default=16, help="Number of tokens to use in IP-adapter cross attention mechanism")
    parser.add_argument("--optimize_adapter", action="store_true", help="Optimize IP-adapter parameters (projector + cross-attention layers)")
    parser.add_argument("--adapter_attention_scale", type=float, default=1.0, help="Relative strength of the adapter cross attention layers")
    parser.add_argument("--adapter_lr", type=float, help="Learning rate for the adapter parameters. Defaults to the global LR if not provided")

    parser.add_argument("--noisy_encoder_input", action="store_true", help="Noise the encoder input to the same step as the decoder?")

    # related to CFG:
    parser.add_argument("--adapter_drop_chance", type=float, default=0.0, help="Chance to drop adapter condition input during training")
    parser.add_argument("--text_drop_chance", type=float, default=0.0, help="Chance to drop text condition during training")
    parser.add_argument("--kv_drop_chance", type=float, default=0.0, help="Chance to drop KV condition during training")

    
    
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )

    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )

    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )

    parser.add_argument("--num_train_epochs", type=int, default=1)

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )

    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help=("Max number of checkpoints to store."),
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    parser.add_argument("--max_timesteps_for_x0_loss", type=int, default=1001)

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )

    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )

    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )

    parser.add_argument(
        "--pretrained_lcm_lora_path",
        type=str,
        default="latent-consistency/lcm-lora-sdxl",
        help=("Path for lcm lora pretrained"),
        )
    
    parser.add_argument(
        "--losses_config_path",
        type=str,
        required=True,
        help=("A yaml file containing losses to use and their weights."),
    )
    
    parser.add_argument(
        "--lcm_every_k_steps",
        type=int,
        default=-1,
        help="How often to run lcm. If -1, lcm is not run."
    )

    parser.add_argument(
        "--lcm_batch_size",
        type=int,
        default=1,
        help="Batch size for lcm."
    )
    parser.add_argument(
        "--lcm_max_timestep",
        type=int,
        default=1000,
        help="Max timestep to use with LCM."
    )

    parser.add_argument(
        "--lcm_sample_scale_every_k_steps",
        type=int,
        default=-1,
        help="How often to change lcm scale. If -1, scale is fixed at 1."
    )

    parser.add_argument(
        "--lcm_min_scale",
        type=float,
        default=0.1,
        help="When sampling lcm scale, the minimum scale to use."
    )

    parser.add_argument(
        "--scale_lcm_by_max_step",
        action="store_true",
        help="scale LCM lora alpha linearly by the maximal timestep sampled that iteration"
    )

    parser.add_argument(
        "--lcm_sample_full_lcm_prob",
        type=float,
        default=0.2,
        help="When sampling lcm scale, the probability of using full lcm (scale of 1)."
    )

    parser.add_argument(
        "--run_on_cpu",
        action="store_true",
        help="whether to run on cpu or not"
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        help=("A short description of the experiment to add to the wand run log. "),
    )
    parser.add_argument("--encoder_lora_rank", type=int, default=0, help="Rank of Lora in unet encoder. 0 means no lora")

    parser.add_argument("--kvcopy_lora_rank", type=int, default=0, help="Rank of lora in the kvcopy modules. 0 means no lora")


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    args.optimizer = "AdamW"

    return args