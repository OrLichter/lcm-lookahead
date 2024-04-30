from pathlib import Path
from typing import Optional

from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
import json
import random
from facenet_pytorch import MTCNN
import torch

from utils.utils import extract_faces_and_landmarks, REFERNCE_FACIAL_POINTS_RELATIVE

def load_image(image_path: str) -> Image:
    image = Image.open(image_path)
    image = exif_transpose(image)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    return image


class ImageDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
            self,
            instance_data_root,
            instance_prompt,
            metadata_path: Optional[str] = None,
            prompt_in_filename=False,
            use_only_vanilla_for_encoder=False,
            concept_placeholder='a face',
            size=1024,
            center_crop=False,
            aug_images=False,
            use_only_decoder_prompts=False,
            crop_head_for_encoder_image=False,
            random_target_prob=0.0,
    ):
        self.mtcnn = MTCNN(device='cuda:0')
        self.mtcnn.forward = self.mtcnn.detect
        resize_factor = 1.3
        self.resized_reference_points = REFERNCE_FACIAL_POINTS_RELATIVE / resize_factor + (resize_factor - 1) / (2 * resize_factor) 
        self.size = size
        self.center_crop = center_crop
        self.concept_placeholder = concept_placeholder
        self.prompt_in_filename = prompt_in_filename
        self.aug_images = aug_images

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.name_to_label = None
        self.crop_head_for_encoder_image = crop_head_for_encoder_image
        self.random_target_prob = random_target_prob

        self.use_only_decoder_prompts = use_only_decoder_prompts

        self.instance_data_root = Path(instance_data_root)

        if not self.instance_data_root.exists():
            raise ValueError(f"Instance images root {self.instance_data_root} doesn't exist.")

        if metadata_path is not None:
            with open(metadata_path, 'r') as f:
                self.name_to_label = json.load(f)  # dict of filename: label
            # Create a reversed mapping
            self.label_to_names = {}
            for name, label in self.name_to_label.items():
                if use_only_vanilla_for_encoder and 'vanilla' not in name:
                    continue
                if label not in self.label_to_names:
                    self.label_to_names[label] = []
                self.label_to_names[label].append(name)
            self.all_paths = [self.instance_data_root / filename for filename in self.name_to_label.keys()]

            # Verify all paths exist
            n_all_paths = len(self.all_paths)
            self.all_paths = [path for path in self.all_paths if path.exists()]
            print(f'Found {len(self.all_paths)} out of {n_all_paths} paths.')
        else:
            self.all_paths = [path for path in list(Path(instance_data_root).glob('**/*')) if
                              path.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            # Sort by name so that order for validation remains the same across runs
            self.all_paths = sorted(self.all_paths, key=lambda x: x.stem)

        self.custom_instance_prompts = None

        self._length = len(self.all_paths)

        self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        if self.prompt_in_filename:
            self.prompts_set = set([self._path_to_prompt(path) for path in self.all_paths])
        else:
            self.prompts_set = set([self.instance_prompt])

        if self.aug_images:
            self.aug_transforms = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5)
                ]
            )

    def __len__(self):
        return self._length

    def _path_to_prompt(self, path):
        # Remove the extension and seed
        split_path = path.stem.split('_')
        while split_path[-1].isnumeric():
            split_path = split_path[:-1]

        prompt = ' '.join(split_path)
        # Replace placeholder in prompt with training placeholder
        prompt = prompt.replace('conceptname', self.concept_placeholder)
        return prompt

    def __getitem__(self, index):
        example = {}
        instance_path = self.all_paths[index]
        instance_image = load_image(instance_path)
        example["instance_images"] = self.image_transforms(instance_image)
        if self.prompt_in_filename:
            example["instance_prompt"] = self._path_to_prompt(instance_path)
        else:
            example["instance_prompt"] = self.instance_prompt

        if self.name_to_label is None:
            # If no labels, simply take the same image but with different augmentation
            example["encoder_images"] = self.aug_transforms(example["instance_images"]) if self.aug_images else example["instance_images"]
            example["encoder_prompt"] = example["instance_prompt"]
        else:
            # Randomly select another image with the same label
            instance_name = str(instance_path.relative_to(self.instance_data_root))
            instance_label = self.name_to_label[instance_name]
            label_set = set(self.label_to_names[instance_label])
            if len(label_set) == 1:
                # We are not supposed to have only one image per label, but just in case
                encoder_image_name = instance_name
                print(f'WARNING: Only one image for label {instance_label}.')
            else:
                encoder_image_name = random.choice(list(label_set - {instance_name}))
            encoder_image = load_image(self.instance_data_root / encoder_image_name)
            example["encoder_images"] = self.image_transforms(encoder_image)

            if self.prompt_in_filename:
                example["encoder_prompt"] = self._path_to_prompt(self.instance_data_root / encoder_image_name)
            else:
                example["encoder_prompt"] = self.instance_prompt
        
        if self.crop_head_for_encoder_image:
            example["encoder_images"] = extract_faces_and_landmarks(example["encoder_images"][None], self.size, self.mtcnn, self.resized_reference_points)[0][0]
        example["encoder_prompt"]  = example["encoder_prompt"].format(placeholder="<ph>")
        example["instance_prompt"] = example["instance_prompt"].format(placeholder="<s*>")

        if random.random() < self.random_target_prob:
            random_path = random.choice(self.all_paths)

            random_image = load_image(random_path)
            example["instance_images"] = self.image_transforms(random_image)
            if self.prompt_in_filename:
                example["instance_prompt"] = self._path_to_prompt(random_path)


        if self.use_only_decoder_prompts:
            example["encoder_prompt"] = example["instance_prompt"]

        return example


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    encoder_pixel_values = [example["encoder_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    encoder_prompts = [example["encoder_prompt"] for example in examples]

    if with_prior_preservation:
        raise NotImplementedError("Prior preservation not implemented.")

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    encoder_pixel_values = torch.stack(encoder_pixel_values)
    encoder_pixel_values = encoder_pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "encoder_pixel_values": encoder_pixel_values,
             "prompts": prompts, "encoder_prompts": encoder_prompts}
    return batch
