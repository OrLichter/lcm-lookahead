import torch
import wandb
import cv2
import torch.nn.functional as F
import numpy as np
from facenet_pytorch import MTCNN
from torchvision import transforms
from dreamsim import dreamsim
from einops import rearrange
import kornia.augmentation as K
import lpips

from pretrained_models.arcface import Backbone
from utils.vis_utils import add_text_to_image
from utils.utils import extract_faces_and_landmarks
import clip

class Loss:
    """
    General purpose loss class. 
    Mainly handles dtype and visualize_every_k.
    keeps current iteration of loss, mainly for visualization purposes.
    """
    def __init__(self, visualize_every_k=-1, dtype=torch.float32, accelerator=None, **kwargs):
        self.visualize_every_k = visualize_every_k
        self.iteration = -1
        self.dtype=dtype
        self.accelerator = accelerator
        
    def __call__(self, **kwargs):
        self.iteration += 1
        return self.forward(**kwargs)

class ImageL1Loss(Loss):
    """
    Simple L1 loss between predicted_pixel_values and pixel_values
    
    Args:
        predicted_pixel_values (torch.Tensor): The predicted pixel values using 1 step LCM and the VAE decoder.
        encoder_pixel_values (torch.Tesnor): The input image to the encoder
    """
    def forward(
        self, 
        predicted_pixel_values: torch.Tensor,
        encoder_pixel_values: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        print(f"predicted_pixel_values dtype: {predicted_pixel_values.dtype}")
        print(f"pixel_values dtype: {encoder_pixel_values.dtype}")
        return F.l1_loss(predicted_pixel_values, encoder_pixel_values, reduction="mean")


class DreamSIMLoss(Loss):
    """DreamSIM loss between predicted_pixel_values and pixel_values.
    DreamSIM is similar to LPIPS (https://dreamsim-nights.github.io/) but is trained on more human defined similarity dataset
    DreamSIM expects an RGB image of size 224x224 and values between 0 and 1. So we need to normalize the input images to 0-1 range and resize them to 224x224.
    Args:
        predicted_pixel_values (torch.Tensor): The predicted pixel values using 1 step LCM and the VAE decoder.
        encoder_pixel_values (torch.Tesnor): The input image to the encoder
    """
    def __init__(self, device: str='cuda:0', **kwargs):
        super().__init__(**kwargs)
        self.model, _ = dreamsim(pretrained=True, device=device)
        self.model.to(dtype=self.dtype, device=device)
        self.model = self.accelerator.prepare(self.model)
        self.transforms = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1) / 2),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)])

    def forward(
        self,
        predicted_pixel_values: torch.Tensor,
        encoder_pixel_values: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        predicted_pixel_values.to(dtype=self.dtype)
        encoder_pixel_values.to(dtype=self.dtype)
        return self.model(self.transforms(predicted_pixel_values), self.transforms(encoder_pixel_values)).mean()


class LPIPSLoss(Loss):
    """LPIPS loss between predicted_pixel_values and pixel_values.
    Args:
        predicted_pixel_values (torch.Tensor): The predicted pixel values using 1 step LCM and the VAE decoder.
        encoder_pixel_values (torch.Tesnor): The input image to the encoder
    """
    def __init__(self, device: str='cuda:0', **kwargs):
        super().__init__(**kwargs)
        self.model = lpips.LPIPS(net='vgg')
        self.model.to(dtype=self.dtype, device=device)
        self.model = self.accelerator.prepare(self.model)

    def forward(self, predicted_pixel_values, encoder_pixel_values, **kwargs):
        predicted_pixel_values.to(dtype=self.dtype)
        encoder_pixel_values.to(dtype=self.dtype)
        return self.model(predicted_pixel_values, encoder_pixel_values).mean()

class LCMVisualization(Loss):
    """Dummy loss used to visualize the LCM outputs
    Args:
        predicted_pixel_values (torch.Tensor): The predicted pixel values using 1 step LCM and the VAE decoder.
        pixel_values (torch.Tensor): The input image to the decoder
        encoder_pixel_values (torch.Tesnor): The input image to the encoder
    """
    def forward(
        self, 
        predicted_pixel_values: torch.Tensor,
        pixel_values: torch.Tensor,
        encoder_pixel_values: torch.Tensor,
        timesteps: torch.Tensor,
        **kwargs,
    ) -> None:
        if self.visualize_every_k > 0 and self.iteration % self.visualize_every_k == 0:
            predicted_pixel_values = rearrange(predicted_pixel_values, "n c h w -> (n h) w c").detach().cpu().numpy()
            pixel_values = rearrange(pixel_values, "n c h w -> (n h) w c").detach().cpu().numpy()
            encoder_pixel_values = rearrange(encoder_pixel_values, "n c h w -> (n h) w c").detach().cpu().numpy()
            image = np.hstack([encoder_pixel_values, pixel_values, predicted_pixel_values])
            for tracker in self.accelerator.trackers:
                if tracker.name == 'wandb':
                    tracker.log({"TrainVisualization": wandb.Image(image, caption=f"Encoder Input Image, Decoder Input Image, Predicted LCM Image. Timesteps {timesteps.cpu().tolist()}")})
        return torch.tensor(0.0)


class NoiseLoss(Loss):
    """
    Regular diffusion loss between predicted noise and target noise.

    Args:
        predicted_noise (torch.Tensor): noise predicted by the diffusion model
        target_noise (torch.Tensor): actual noise added to the image.
    """
    def forward(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        return F.mse_loss(predicted_noise.float(), target_noise.float(), reduction="mean")

class WeightedNoiseLoss(Loss):
    """
    Weighted diffusion loss between predicted noise and target noise.

    Args:
        predicted_noise (torch.Tensor): noise predicted by the diffusion model
        target_noise (torch.Tensor): actual noise added to the image.
        loss_batch_weights (torch.Tensor): weighting for each batch item. Can be used to e.g. zero-out loss for InstantID training if keypoint extraction fails.
    """
    def forward(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor,
        loss_batch_weights,
        **kwargs
    ) -> torch.Tensor:
        weights = loss_batch_weights.reshape(-1, 1, 1, 1)
        return F.mse_loss(predicted_noise.float() * weights, target_noise.float() * weights, reduction="mean")
           
class IDLoss(Loss):
    """
    Use pretrained facenet model to extract features from the face of the predicted image and target image.
    Facenet expects 112x112 images, so we crop the face using MTCNN and resize it to 112x112.
    Then we use the cosine similarity between the features to calculate the loss. (The cosine similarity is 1 - cosine distance).
    Also notice that the outputs of facenet are normalized so the dot product is the same as cosine distance.
    """
    def __init__(self, pretrained_arcface_path: str, skip_not_found=True, **kwargs):
        super().__init__(**kwargs)
        assert pretrained_arcface_path is not None, "please pass `pretrained_arcface_path` in the losses config. You can download the pretrained model from "\
            "https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing"
        self.mtcnn = MTCNN(device=self.accelerator.device)
        self.mtcnn.forward = self.mtcnn.detect
        self.facenet_input_size = 112  # Has to be 112, can't find weights for 224 size.
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(pretrained_arcface_path))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((self.facenet_input_size, self.facenet_input_size))
        self.facenet.requires_grad_(False)
        self.facenet.eval()
        self.facenet.to(device=self.accelerator.device, dtype=self.dtype)  # not implemented for half precision
        self.face_pool.to(device=self.accelerator.device, dtype=self.dtype)  # not implemented for half precision
        self.visualization_resize = transforms.Resize((self.facenet_input_size, self.facenet_input_size), interpolation=transforms.InterpolationMode.BICUBIC)
        self.reference_facial_points = np.array([[38.29459953, 51.69630051],
                                                 [72.53179932, 51.50139999],
                                                 [56.02519989, 71.73660278],
                                                 [41.54930115, 92.3655014],
                                                 [70.72990036, 92.20410156]
                                                 ])  # Original points are 112 * 96 added 8 to the x axis to make it 112 * 112
        self.facenet, self.face_pool, self.mtcnn = self.accelerator.prepare(self.facenet, self.face_pool, self.mtcnn)

        self.skip_not_found = skip_not_found
    
    def extract_feats(self, x: torch.Tensor):
        """
        Extract features from the face of the image using facenet model.
        """
        x = self.face_pool(x)
        x_feats = self.facenet(x)

        return x_feats

    def forward(
        self, 
        predicted_pixel_values: torch.Tensor,
        encoder_pixel_values: torch.Tensor,
        timesteps: torch.Tensor,
        **kwargs
    ):
        encoder_pixel_values = encoder_pixel_values.to(dtype=self.dtype)
        predicted_pixel_values = predicted_pixel_values.to(dtype=self.dtype)

        predicted_pixel_values_face, predicted_invalid_indices = extract_faces_and_landmarks(predicted_pixel_values, mtcnn=self.mtcnn)
        with torch.no_grad():
            encoder_pixel_values_face, source_invalid_indices = extract_faces_and_landmarks(encoder_pixel_values, mtcnn=self.mtcnn)
        
        if self.skip_not_found:
            valid_indices = []
            for i in range(predicted_pixel_values.shape[0]):
                if i not in predicted_invalid_indices and i not in source_invalid_indices:
                    valid_indices.append(i)
        else:
            valid_indices = list(range(predicted_pixel_values))
            
        valid_indices = torch.tensor(valid_indices).to(device=predicted_pixel_values.device)

        if len(valid_indices) == 0:
            loss =  (predicted_pixel_values_face * 0.0).mean()  # It's done this way so the `backwards` will delete the computation graph of the predicted_pixel_values.
            if self.visualize_every_k > 0 and self.iteration % self.visualize_every_k == 0:
                self.visualize(predicted_pixel_values, encoder_pixel_values, predicted_pixel_values_face, encoder_pixel_values_face, timesteps, valid_indices, loss)
            return loss

        with torch.no_grad():
            pixel_values_feats = self.extract_feats(encoder_pixel_values_face[valid_indices])
            
        predicted_pixel_values_feats = self.extract_feats(predicted_pixel_values_face[valid_indices])
        loss = 1 - torch.einsum("bi,bi->b", pixel_values_feats, predicted_pixel_values_feats)

        if self.visualize_every_k > 0 and self.iteration % self.visualize_every_k == 0:
            self.visualize(predicted_pixel_values, encoder_pixel_values, predicted_pixel_values_face, encoder_pixel_values_face, timesteps, valid_indices, loss)
        return loss.mean()
    
    def visualize(
        self,
        predicted_pixel_values: torch.Tensor,
        encoder_pixel_values: torch.Tensor,
        predicted_pixel_values_face: torch.Tensor,
        encoder_pixel_values_face: torch.Tensor,
        timesteps: torch.Tensor,
        valid_indices: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        small_predicted_pixel_values = (rearrange(self.visualization_resize(predicted_pixel_values), "n c h w -> (n h) w c").detach().cpu().numpy())
        small_pixle_values = rearrange(self.visualization_resize(encoder_pixel_values), "n c h w -> (n h) w c").detach().cpu().numpy() 
        small_predicted_pixel_values_face = rearrange(self.visualization_resize(predicted_pixel_values_face), "n c h w -> (n h) w c").detach().cpu().numpy()
        small_pixle_values_face = rearrange(self.visualization_resize(encoder_pixel_values_face), "n c h w -> (n h) w c").detach().cpu().numpy()
        
        small_predicted_pixel_values = add_text_to_image(((small_predicted_pixel_values * 0.5 + 0.5) * 255).astype(np.uint8), "Pred Images", add_below=False)
        small_pixle_values = add_text_to_image(((small_pixle_values * 0.5 + 0.5) * 255).astype(np.uint8), "Target Images", add_below=False)
        small_predicted_pixel_values_face = add_text_to_image(((small_predicted_pixel_values_face * 0.5 + 0.5) * 255).astype(np.uint8), "Pred Faces", add_below=False)
        small_pixle_values_face = add_text_to_image(((small_pixle_values_face * 0.5 + 0.5) * 255).astype(np.uint8), "Target Faces", add_below=False)


        final_image = np.hstack([small_predicted_pixel_values, small_pixle_values, small_predicted_pixel_values_face, small_pixle_values_face])
        for tracker in self.accelerator.trackers:
            if tracker.name == 'wandb':
                tracker.log({"IDLoss Visualization": wandb.Image(final_image, caption=f"loss: {loss.cpu().tolist()} timesteps: {timesteps.cpu().tolist()}, valid_indices: {valid_indices.cpu().tolist()}")})


class ImageAugmentations(torch.nn.Module):
    # Standard image augmentations used for CLIP loss to discourage adversarial outputs.
    def __init__(self, output_size, augmentations_number, p=0.7):
        super().__init__()
        self.output_size = output_size
        self.augmentations_number = augmentations_number

        self.augmentations = torch.nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=p, padding_mode="border"),  # type: ignore
            K.RandomPerspective(0.7, p=p),
        )

        self.avg_pool = torch.nn.AdaptiveAvgPool2d((self.output_size, self.output_size))

        self.device = None

    def forward(self, input):
        """Extents the input batch with augmentations
        If the input is consists of images [I1, I2] the extended augmented output
        will be [I1_resized, I2_resized, I1_aug1, I2_aug1, I1_aug2, I2_aug2 ...]
        Args:
            input ([type]): input batch of shape [batch, C, H, W]
        Returns:
            updated batch: of shape [batch * augmentations_number, C, H, W]
        """
        # We want to multiply the number of images in the batch in contrast to regular augmantations
        # that do not change the number of samples in the batch)
        resized_images = self.avg_pool(input)
        resized_images = torch.tile(resized_images, dims=(self.augmentations_number, 1, 1, 1))

        batch_size = input.shape[0]
        # We want at least one non augmented image
        non_augmented_batch = resized_images[:batch_size]
        augmented_batch = self.augmentations(resized_images[batch_size:])
        updated_batch = torch.cat([non_augmented_batch, augmented_batch], dim=0)

        return updated_batch
    
class CLIPLoss(Loss):
    def __init__(self, augmentations_number: int = 4, **kwargs):
        super().__init__(**kwargs)

        self.clip_model, clip_preprocess = clip.load("ViT-B/16", device=self.accelerator.device, jit=False)

        self.clip_model.device = None

        self.clip_model.eval().requires_grad_(False)
        
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (SD output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

        self.clip_size = self.clip_model.visual.input_resolution

        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )

        self.image_augmentations = ImageAugmentations(output_size=self.clip_size,
                                                      augmentations_number=augmentations_number)
        
        self.clip_model, self.image_augmentations = self.accelerator.prepare(self.clip_model, self.image_augmentations)

    def forward(self, decoder_prompts, predicted_pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:

        if not isinstance(decoder_prompts, list):
            decoder_prompts = [decoder_prompts]

        tokens = clip.tokenize(decoder_prompts).to(predicted_pixel_values.device)
        image  = self.preprocess(predicted_pixel_values)

        logits_per_image, _ = self.clip_model(image, tokens)

        logits_per_image = torch.diagonal(logits_per_image)

        return (1. - logits_per_image / 100).mean()