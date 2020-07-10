import os
import glob
import rasterio
import numpy as np
import pandas as pd
from typing import Optional
import torch
from torch.utils.data import Dataset
from . import transforms
import cv2

import warnings

warnings.simplefilter("ignore")


class SegmentationDatasetEye(Dataset):

    def __init__(
            self,
            images_dir: str,
            masks_dir: str,
            ids: Optional[list] = None,
            transform_name: Optional[str] = None,
    ):
        super().__init__()
        self.ids = ids if ids is not None else os.listdir(images_dir)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transforms.__dict__[transform_name] if transform_name else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id = self.ids[i]
        image_path = os.path.join(self.images_dir, id)
        mask_path = os.path.join(self.masks_dir, id)

        # read data sample
        sample = dict(
            id=id,
            image=self.read_image(image_path),
            mask=self.read_mask(mask_path),
        )

        # apply augmentations
        if self.transform is not None:
            sample = self.transform(**sample)

        sample["mask"] = sample["mask"][None]  # expand first dim for mask

        return sample

    def read_image(self, path):
        image = cv2.imread(path)
#        image = image.transpose(1, 2, 0)
#        print('img:', image.shape)
        return image

    def read_mask(self, path):
        image = np.load(path[:-4] + '.npy')
#        print('mask:', image.shape)
        image = (image > 0).astype(np.uint8)
        return image

    def read_image_profile(self, id):
        path = os.path.join(self.images_dir, id)
        with rasterio.open(path) as f:
            return f.profile


class SegmentationDatasetIris(Dataset):

    def __init__(
            self,
            images_dir: str,
            masks_dir: str,
            ids: Optional[list] = None,
            transform_name: Optional[str] = None,
    ):
        super().__init__()
        self.ids = ids if ids is not None else os.listdir(images_dir)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transforms.__dict__[transform_name] if transform_name else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id = self.ids[i]
        image_path = os.path.join(self.images_dir, id)
        mask_path = os.path.join(self.masks_dir, id)

        # read data sample
        sample = dict(
            id=id,
            image=self.read_image(image_path),
            mask=self.read_mask(mask_path),
        )

        # apply augmentations
        if self.transform is not None:
            sample = self.transform(**sample)

        sample["mask"] = sample["mask"][None]  # expand first dim for mask

        return sample

    def read_image(self, path):
        image = cv2.imread(path)
#        image = image.transpose(1, 2, 0)
#        print('img:', image.shape)
        return image

    def read_mask(self, path):
        image = np.load(path[:-4] + '.npy')
#        print('mask:', image.shape)
        image = (image > 1).astype(np.uint8)
        return image

    def read_image_profile(self, id):
        path = os.path.join(self.images_dir, id)
        with rasterio.open(path) as f:
            return f.profile


class SegmentationDatasetPupil(Dataset):

    def __init__(
            self,
            images_dir: str,
            masks_dir: str,
            ids: Optional[list] = None,
            transform_name: Optional[str] = None,
    ):
        super().__init__()
        self.ids = ids if ids is not None else os.listdir(images_dir)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transforms.__dict__[transform_name] if transform_name else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id = self.ids[i]
        image_path = os.path.join(self.images_dir, id)
        mask_path = os.path.join(self.masks_dir, id)

        # read data sample
        sample = dict(
            id=id,
            image=self.read_image(image_path),
            mask=self.read_mask(mask_path),
        )

        # apply augmentations
        if self.transform is not None:
            sample = self.transform(**sample)

        sample["mask"] = sample["mask"][None]  # expand first dim for mask

        return sample

    def read_image(self, path):
        image = cv2.imread(path)
#        image = image.transpose(1, 2, 0)
#        print('img:', image.shape)
        return image

    def read_mask(self, path):
        image = np.load(path[:-4] + '.npy')
#        print('mask:', image.shape)
        image = (image > 2).astype(np.uint8)
        return image

    def read_image_profile(self, id):
        path = os.path.join(self.images_dir, id)
        with rasterio.open(path) as f:
            return f.profile


class TestSegmentationDataset(Dataset):

    def __init__(self, images_dir, transform_name=None):
        super().__init__()
        self.ids = os.listdir(images_dir)
        self.images_dir = images_dir
        self.transform = transforms.__dict__[transform_name] if transform_name else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id = self.ids[i]
        path = os.path.join(self.images_dir, id)

        sample = dict(
            id=id,
            image=self.read_image(path),
        )

        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    def read_image(self, path):
        image = cv2.imread(path)
#        image = image.transpose(1, 2, 0)
#        print('img:', image.shape)
        return image

    def read_image_profile(self, id):
        pass
