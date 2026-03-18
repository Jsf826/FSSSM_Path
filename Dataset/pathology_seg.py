import os
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PathologySegDataset(Dataset):
    """
    通用病理分割数据集。

    目录结构假定为：
        root/
            images/
                xxx.png
            masks/
                xxx.png  （与 images 同名）
    """

    def __init__(
        self,
        root: str,
        indices: Optional[Sequence[int]] = None,
        patch_size: int = 256,
        labeled: bool = True,
    ):
        super().__init__()
        self.root = root
        self.image_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.labeled = labeled

        self.filenames = sorted(
            [
                f
                for f in os.listdir(self.image_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
            ]
        )

        if indices is not None:
            self.filenames = [self.filenames[i] for i in indices]

        self.img_transform = transforms.Compose(
            [
                transforms.Resize((patch_size, patch_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.mask_resize = transforms.Resize(
            (patch_size, patch_size), interpolation=Image.NEAREST
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.filenames[idx]
        img_path = os.path.join(self.image_dir, fname)
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)

        if self.labeled:
            mask_path = os.path.join(self.mask_dir, fname)
            mask = Image.open(mask_path)
            mask = self.mask_resize(mask)
            mask_np = np.array(mask, dtype=np.int64)
            if mask_np.ndim == 3:
                # 如果是 RGB mask，取单通道或转换为灰度再用
                mask_np = mask_np[..., 0]
            mask = torch.from_numpy(mask_np).long()
        else:
            # 无标注样本：返回占位 mask（全 0）
            h, w = img.shape[1], img.shape[2]
            mask = torch.zeros((h, w), dtype=torch.long)

        return img, mask

