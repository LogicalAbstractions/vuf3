from typing import Optional, Tuple

import torch
from PIL.Image import Image
from torch import nn, Tensor

from torchvision.transforms import functional as F, InterpolationMode


class VideoDefaultPreprocessor(nn.Module):
    def __init__(
            self,
            *,
            crop_size: Tuple[int, int],
            resize_size: Tuple[int, int],
            mean: Tuple[float, ...] = (0.43216, 0.394666, 0.37645),
            std: Tuple[float, ...] = (0.22803, 0.22145, 0.216989),
            interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.crop_size = list(crop_size)
        self.resize_size = list(resize_size)
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation

    def forward(self, vid: Tensor) -> Tensor:
        need_squeeze = False
        if vid.ndim < 5:
            vid = vid.unsqueeze(dim=0)
            need_squeeze = True

        N, T, C, H, W = vid.shape
        vid = vid.view(-1, C, H, W)
        original_image: Image = F.to_pil_image(vid[0])
        original_image.save("original.jpg")
        vid = F.resize(vid, self.resize_size, interpolation=self.interpolation)
        resized_image: Image = F.to_pil_image(vid[0])
        resized_image.save("resized.jpg")
        vid = F.center_crop(vid, self.crop_size)
        cropped_image: Image = F.to_pil_image(vid[0])
        cropped_image.save("cropped.jpg")

        vid = F.convert_image_dtype(vid, torch.float)
        vid = F.normalize(vid, mean=self.mean, std=self.std)
        H, W = self.crop_size
        vid = vid.view(N, T, C, H, W)
        vid = vid.permute(0, 2, 1, 3, 4)  # (N, T, C, H, W) => (N, C, T, H, W)

        if need_squeeze:
            vid = vid.squeeze(dim=0)
        return vid

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts batched ``(B, T, C, H, W)`` and single ``(T, C, H, W)`` video frame ``torch.Tensor`` objects. "
            f"The frames are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
            f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are first rescaled to "
            f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and ``std={self.std}``. Finally the output "
            "dimensions are permuted to ``(..., C, T, H, W)`` tensors."
        )
