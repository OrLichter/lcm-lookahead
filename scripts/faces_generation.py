from diffusers import StableDiffusionXLPipeline
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import pyrallis
from dataclasses import dataclass
from scripts import ids_bank, prompts_bank
import numpy as np
import json
from tqdm import tqdm


@dataclass
class Config:
    out_dir: Path = Path('datasets/generated/generated_people_v1')
    vanilla_images_per_prompt: int = 40
    stylized_images_per_prompt: int = 8


@pyrallis.wrap()
def main(cfg: Config):
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", use_safetensors=True
    ).to("cuda")

    cfg.out_dir.mkdir(exist_ok=True, parents=True)
    metadata = {}
    for name in tqdm(ids_bank.characters + ids_bank.celebs):

        def gen_and_save(prompt: str, out_dir: Path) -> Path:
            seed = np.random.randint(0, 1000000)
            image = pipeline(prompt=prompt.format(concept=name), num_inference_steps=1, guidance_scale=0.0,
                             generator=[torch.Generator().manual_seed(seed)]).images[0]
            out_path = out_dir / f'{prompt.format(concept="conceptname").replace(" ", "_")}_{seed}.jpg'
            image.save(out_path)
            metadata[str(out_path.relative_to(cfg.out_dir))] = name
            return out_path

        vanilla_out_dir = cfg.out_dir / name / 'vanilla'
        vanilla_out_dir.mkdir(exist_ok=True, parents=True)
        for i in range(cfg.vanilla_images_per_prompt):
            for prompt in prompts_bank.vanilla_prompts:
                gen_and_save(prompt=prompt, out_dir=vanilla_out_dir)

        stylized_out_dir = cfg.out_dir / name / 'stylized'
        stylized_out_dir.mkdir(exist_ok=True, parents=True)
        for i in range(cfg.stylized_images_per_prompt):
            for prompt in prompts_bank.stylized_prompts:
                gen_and_save(prompt=prompt, out_dir=stylized_out_dir)

        with open(cfg.out_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)

    print('Done')


if __name__ == "__main__":
    main()
