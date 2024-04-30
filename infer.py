from dataclasses import dataclass, field
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import json
import pyrallis
import torch
import torch.utils.checkpoint
from PIL import Image
from facenet_pytorch import MTCNN
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel, 
    DPMSolverMultistepScheduler, )
from torchvision import transforms
from tqdm import tqdm

from model.decoder_unet import ExpandedKVUNet2DConditionModel
from model.encoder_unet import ExtractKVUNet2DConditionModel
from model.ipadapter.ipadapter import IPAdapterPlusXL
from pipelines.sdxl_encoder_pipeline_with_adapter import StableDiffusionXLEncoderPipeline
from pipelines.sdxl_encoder_pipeline_with_adapter_no_kv import StableDiffusionXLEncoderPipeline as StableDiffusionXLEncoderNoKVPipeline
from train_encoder import import_model_class_from_model_name_or_path
from utils import vis_utils
from utils.utils import extract_faces_and_landmarks



@dataclass
class Config:
    # Checkpoint directory, will take the pytorch_bin_model.bin file
    checkpoint_path: Path = Path("pretrained_models/ours/model_ckpt.pt")
    # Path to directory of images or a path to a single image
    inputs_path: Path = Path("sample_images/")
    # Prompt to use for the encoding unet, if None will use the same as the decoding unet
    encoder_prompt: Optional[str] = "A photo of a face"
    # Prompts to use for the decoding unet
    prompts: List[str] = field(default_factory=lambda: ["A photo of a face", "An oil painting of a face", "A cubism painting of a face"])
    out_dir: Path = Path("infer_outputs/")
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    pretrained_vae_model_name_or_path: Optional[str] = 'madebyollin/sdxl-vae-fp16-fix'
    revision: Optional[str] = None
    variant: Optional[str] = None
    mixed_precision: str = "fp16"
    img_size: int = 1024
    device: str = 'cuda:0'
    num_images_per_prompt: int = 2
    ip_adapter_feature_extractor_path: str = 'pretrained_models/ipadapter/image_encoder'
    ip_adapter_tokens: int = 16
    adapter_attention_scale: float = 1.0
    encoder_lora_rank: int = 4
    kvcopy_lora_rank: int = 4
    noisy_encoder_input: bool = True
    guidance_scale_nokv: float = 3.0
    guidance_scale_full: float = 2.0
    guidance_scale_kv: float = 2.0
    crop_face: bool = False
    seed: int = 0
    num_inference_steps: int = 50
    use_freeu: bool = False
    no_kv: bool = False # Run without attention sharing. Please make sure you use an appropriate model checkpoint.
    guidance_scale: float = 5.0 # Scale used when running without shared attention

def init_pipeline(cfg: Config) -> Tuple[StableDiffusionXLEncoderPipeline, Dict]:
    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        cfg.pretrained_model_name_or_path, cfg.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        cfg.pretrained_model_name_or_path, cfg.revision, subfolder="text_encoder_2"
    )

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="text_encoder", revision=cfg.revision, variant=cfg.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=cfg.revision, variant=cfg.variant
    )

    vae_path = (
        cfg.pretrained_model_name_or_path
        if cfg.pretrained_vae_model_name_or_path is None
        else cfg.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if cfg.pretrained_vae_model_name_or_path is None else None,
        revision=cfg.revision,
        variant=cfg.variant,
    )

    orig_unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="unet", revision=cfg.revision, variant=cfg.variant
    )

    orig_unet_config = orig_unet.config

    encoder_unet = ExtractKVUNet2DConditionModel()
    if cfg.encoder_lora_rank > 0:
        encoder_unet.init_lora_in_encoder(rank=cfg.encoder_lora_rank)
    if cfg.kvcopy_lora_rank > 0:
        encoder_unet.init_lora_in_kvcopy(rank=cfg.kvcopy_lora_rank)

    decoder_unet = ExpandedKVUNet2DConditionModel()
    # Load state_dict
    decoder_unet.load_state_dict(orig_unet.state_dict())
    
    weight_dtype = torch.float32
    if cfg.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif cfg.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    adapter = IPAdapterPlusXL(encoder_unet,
                              cfg.ip_adapter_feature_extractor_path,
                            #   cfg.ip_adapter_model_path,
                              None,
                              cfg.device,
                              num_tokens=cfg.ip_adapter_tokens)

    adapter.requires_grad_(False)

    msg = adapter.load_state_dict(torch.load(cfg.checkpoint_path), strict=True)
    print(msg)

    decoder_unet.set_adapter_attention_scale(cfg.adapter_attention_scale)

    adapter.to(cfg.device, dtype=weight_dtype).eval()

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    encoder_unet.to(cfg.device, dtype=weight_dtype).eval()
    decoder_unet.to(cfg.device, dtype=weight_dtype).eval()
    vae.to(cfg.device, dtype=weight_dtype).eval()

    text_encoder_one.to(cfg.device, dtype=weight_dtype).eval()
    text_encoder_two.to(cfg.device, dtype=weight_dtype).eval()

    pipeline_class = StableDiffusionXLEncoderNoKVPipeline if cfg.no_kv else StableDiffusionXLEncoderPipeline

    pipeline = pipeline_class.from_pretrained(
        cfg.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        encoder_unet=encoder_unet,
        decoder_unet=decoder_unet,
        unet_config=orig_unet_config,
        revision=cfg.revision,
        variant=cfg.variant,
        torch_dtype=weight_dtype,
        adapter=adapter
    )

    if cfg.use_freeu:
        pipeline.decoder_unet.enable_freeu(b1=1.0, b2=1.1, s1=0.9, s2=0.2)

    pipeline = pipeline.to(cfg.device).to(weight_dtype)

    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config, **scheduler_args
    )

    return pipeline


@pyrallis.wrap()
def main(cfg: Config):
    if cfg.inputs_path.is_file():
        paths = [cfg.inputs_path]
    else:
        paths = [path for path in cfg.inputs_path.glob('**/*') if path.suffix.lower() in [".png", ".jpg", ".jpeg"]]

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save cfg as json in output dir
    cfg_dict = cfg.__dict__.copy()
    cfg_json_path = str(cfg.out_dir / "config.json")
    for k, v in cfg_dict.items():
        if isinstance(v, Path):
            cfg_dict[k] = str(v)
    with open(cfg_json_path, "w") as f:
        json.dump(cfg_dict, f, indent=4)

    pipeline = init_pipeline(cfg)

    transform = transforms.Compose(
        [
            transforms.Resize(cfg.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(cfg.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    if cfg.crop_face:
        mtcnn = MTCNN(device=cfg.device)
        mtcnn.forward = mtcnn.detect

    # Create seeds to use for all images based on the number of images per prompt
    torch.manual_seed(cfg.seed)
    seeds = torch.randint(0, 1000000, (cfg.num_images_per_prompt,)).tolist()

    for path in tqdm(paths, desc=" images", position=0):
        image_out_dir = cfg.out_dir / path.stem
        image_out_dir.mkdir(parents=True, exist_ok=True)
        input_image_pil = Image.open(path).convert('RGB').resize((cfg.img_size, cfg.img_size))
        input_image = transform(input_image_pil).to(cfg.device).to(pipeline.vae.dtype)
        if cfg.crop_face:
            input_image = extract_faces_and_landmarks(input_image[None], output_size=cfg.img_size, mtcnn=mtcnn)[0][0]
            
        for prompt in tqdm(cfg.prompts, desc=" prompts", position=1):
            encoder_prompt = cfg.encoder_prompt if cfg.encoder_prompt is not None else prompt
            pipeline_args = {"prompt": prompt,
                             "encoder_prompt": encoder_prompt,
                             "num_inference_steps": cfg.num_inference_steps,
                             "guidance_scale_nokv": cfg.guidance_scale_nokv,
                             "guidance_scale_full": cfg.guidance_scale_full,
                             "guidance_scale_kv": cfg.guidance_scale_kv,
                             "guidance_scale": cfg.guidance_scale,
                             }

            if cfg.noisy_encoder_input:
                pipeline_args["noisy_encoder_input"] = cfg.noisy_encoder_input

            for i in tqdm(range(cfg.num_images_per_prompt), desc=" seeds", position=2):
                seed = seeds[i]
                generator = torch.Generator(device=cfg.device).manual_seed(seeds[i])
                with torch.no_grad():
                    image = pipeline(**pipeline_args, cond_images=input_image.unsqueeze(0), generator=generator).images[0]
                image.save(image_out_dir / f"{path.stem}_{prompt.replace(' ', '_')}_{seed}.jpg")

        print('Done')


if __name__ == "__main__":
    main()