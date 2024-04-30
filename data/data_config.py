from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SingleDataConfig:
    data_root: str
    metadata_path: Optional[str] = None
    prompt_in_filename: bool = False
    use_only_vanilla_for_encoder: bool = False
    dataset_weight: float = 1.0 # Not used yet
    aug_images: bool = False
    concept_placeholder: str = "a face"
    use_only_decoder_prompts: bool = False

@dataclass
class DataConfig:
    datasets: List[SingleDataConfig]
    val_dataset: Optional[SingleDataConfig] = None
    validation_prompts: List[str] = field(default_factory=lambda : ["A photo of a face in the style of monet"])
    balance_datasets: bool = False
    crop_head_for_encoder_image: bool = False
    random_target_prob: float = 0.0
