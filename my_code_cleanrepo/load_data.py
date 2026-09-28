import glob
import os
import random
from typing import Callable, Optional, Union
import rasterio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import v2

def set_seed(seed: int = 24):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class SegmentationDataset(Dataset):
    def __init__(self, list_img_path: list[str], transform: Callable, data_aug:bool=False):
        """
        The data are images stored in int16 and the binary masks stored in int16 with values 0 and 1.
        """
        self.l_img_path = list_img_path
        self.transform = transform
        self.data_aug = data_aug

    def __len__(self):
        return len(self.l_img_path)

    #     return img, mask
    def __getitem__(self, idx: int, plot: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.l_img_path[idx]
        mask_path = img_path.replace("images", "labels")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        rsz = self.transform.resize_size
        with Image.open(mask_path) as mask_pil:
            mask_tensor_original = torch.from_numpy(np.array(mask_pil)).long() # (h,w) tensor
            mask_tensor = v2.Resize((rsz, rsz), interpolation=v2.InterpolationMode.NEAREST)(mask_tensor_original.unsqueeze(0)).squeeze(0) 
            #needed to add channel to resize. then we remove it with squeeze

        if self.transform:
            try:
                with Image.open(img_path) as img_pil: # sometimes PIL bugs with uint16 multiband rasters?
                    img_tensor = self.transform(img_pil)
            except:
                with rasterio.open(img_path) as img_rasterio:
                    img_pil = img_rasterio.read()
                    img_tensor = self.transform(img_pil)
           

        ### data augmentations (only flips for SAR??)
        if self.data_aug:
            if torch.rand(1) < 0.5: # flip h
                img_tensor = F.hflip(img_tensor)
                mask_tensor = F.hflip(mask_tensor)
            if torch.rand(1) < 0.5: # flip v
                img_tensor = F.vflip(img_tensor)
                mask_tensor = F.vflip(mask_tensor)

            if torch.rand(1) < 0.3: # realistic for SAR?
                    k = torch.randint(1, 4, (1,)).item()  # 1,2,3
                    img_tensor = torch.rot90(img_tensor, k, dims=(-2, -1))
                    mask_tensor = torch.rot90(mask_tensor, k, dims=(-2, -1))

            # if torch.rand(1).item() < 0.5: #TODO does this matters at all
            #     sigma = 0.05
            #     img_tensor = img_tensor * torch.exp(torch.randn_like(img_tensor) * sigma)
            #     img_tensor = img_tensor.clamp(max=1.0) 
            # blackout strips
            if torch.rand(1).item() < 0.05:  # 5% chance
                _, H, W = img_tensor.shape

                band_h = int(H * 0.2)  # 1/5 height
                band_w = int(W * 0.2)  # 1/5 width

                mode = torch.randint(0, 4, (1,)).item()

                if mode == 0:
                    # Top (up-most horizontal band)
                    img_tensor[:, :band_h, :] = 0
                    mask_tensor[:band_h, :] = 0

                elif mode == 1:
                    # Bottom (lower-most horizontal band)
                    img_tensor[:, H - band_h:, :] = 0
                    mask_tensor[H - band_h:, :] = 0

                elif mode == 2:
                    # Left-most vertical band
                    img_tensor[:, :, :band_w] = 0
                    mask_tensor[:, :band_w] = 0

                elif mode == 3:
                    # Right-most vertical band
                    img_tensor[:, :, W - band_w:] = 0
                    mask_tensor[:, W - band_w:] = 0

        return img_tensor, mask_tensor

def get_train_val_dataloaders(
    path_train_dirs: list,
    path_val_dirs:list,
    batch_size: int = 16,
    num_workers: int = 0,
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None,
    data_aug:bool=False,
    subset: Optional[Union[int,float]] = None, 
    return_datasets: bool = False,
    random_seed:int=24
) -> tuple[DataLoader, DataLoader]:
    """
    Code from T.Kerdreux adapted in order to have fully split train/val folders instead of index split (problematic with overlapping tiles)
    subset: prioritize give as frac (float)
    """
    l_train_imgs = []
    l_val_imgs = []

    for path_train_dir in path_train_dirs:
        path_train_imgs = os.path.join(path_train_dir, "images")
        assert os.path.isdir(path_train_imgs)
        assert os.path.isdir(os.path.join(path_train_dir, "labels"))
        train_imgs_folder = glob.glob(path_train_imgs + "/*.tif")
        l_train_imgs.extend(train_imgs_folder)
    for path_val_dir in path_val_dirs:
        path_val_imgs = os.path.join(path_val_dir, "images")
        assert os.path.isdir(path_val_imgs)
        assert os.path.isdir(os.path.join(path_val_dir, "labels"))
        val_imgs_folder = glob.glob(path_val_imgs + "/*.tif")
        l_val_imgs.extend(val_imgs_folder)

    # shuffle img and val lists
    set_seed(random_seed)
    random.shuffle(l_train_imgs)
    random.shuffle(l_val_imgs)

    # train_dataset = SegmentationDataset(
    #     l_train_imgs,
    #     transform=transform_train,
    #     data_aug=data_aug
    # )
    # val_dataset = SegmentationDataset(
    #     l_val_imgs,
    #     transform=transform_val,
    #     data_aug=data_aug
    # )
    if not subset:
        train_dataset = SegmentationDataset(
            l_train_imgs,
            transform=transform_train,
            data_aug=data_aug
        )
        val_dataset = SegmentationDataset(
            l_val_imgs,
            transform=transform_val,
            data_aug=False #NOTE: no aug for val
        )
    else:
        if isinstance(subset,float) and 0 < subset < 1: #subset is given as percentage.
            subset_t = round(subset * len(l_train_imgs))
            subset_v = round(subset * len(l_val_imgs))
        else:
            subset_t = subset_v = subset # given as int
        min_subset = min(subset_t, len(l_train_imgs))

        # load train val datasets
        train_dataset = SegmentationDataset(
            l_train_imgs[:min_subset],
            transform=transform_train,
            data_aug=data_aug
        )
        
        min_subset = min(subset_v, len(l_val_imgs))
        val_dataset = SegmentationDataset(
            l_val_imgs[:min_subset], # use val subset?
            # l_val_imgs, # don't?
            transform=transform_val,
            data_aug=data_aug
        )

    # Very standard dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers else False
    )
    if not return_datasets:
        return train_loader, val_loader
    else:
        return train_dataset, val_dataset


def get_train_val_dataloaders_galeio( ## with random shuffling
    path_data_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
    frac_train: float = 0.9,
    random_shuffle:bool = True,
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None,
    data_aug:bool=False,
    subset: Optional[int] = None,
    return_datasets: bool = False,
    random_seed:int=24
) -> tuple[DataLoader, DataLoader]:
    """
    The structure of {url_data_dir} is the following:
    {url_data_dir}/
        imgs/
            img_0.png
            ...
        masks/
            img_0.png
            ...
    args
        subset (Optional[int]): if specified, only a subset of the images will be used.
    """
    assert os.path.exists(path_data_dir)
    path_folder_imgs = os.path.join(path_data_dir, "images")
    path_folder_masks = os.path.join(path_data_dir, "labels")
    assert os.path.exists(path_folder_imgs)
    assert os.path.exists(path_folder_masks)
    
    set_seed(random_seed)
    l_path_imgs = glob.glob(path_folder_imgs + "/*.tif")

    if random_shuffle:
        random.shuffle(l_path_imgs)

    if not subset:
        train_dataset = SegmentationDataset(
            l_path_imgs[: int(len(l_path_imgs) * frac_train)],
            transform=transform_train,
            data_aug=data_aug
        )
        val_dataset = SegmentationDataset(
            l_path_imgs[int(len(l_path_imgs) * frac_train) :],
            transform=transform_val,
            data_aug=data_aug
        )
    else:
        min_subset = min(subset, len(l_path_imgs))
        train_dataset = SegmentationDataset(
            l_path_imgs[: int(min_subset * frac_train)],
            transform=transform_train,
            data_aug=data_aug
        )
        val_dataset = SegmentationDataset(
            l_path_imgs[int(min_subset * frac_train) :],
            transform=transform_val,
            data_aug=data_aug
        )

    # Very standard dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )
    if not return_datasets:
        return train_loader, val_loader
    else:
        return train_dataset, val_dataset


