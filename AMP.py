import torch
import cv2
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import os
import random
import math
import glob
from pathlib import Path
from PIL import Image
from skimage.filters import gaussian
from skimage.restoration import denoise_bilateral
import albumentations as A
from BDD100K import letterbox, RandomBilateralBlur, RandomGaussianBlur, augment_hsv, random_perspective

class AMPDataset(torch.utils.data.Dataset):
    '''
    AMP Dataset class that follows BDD100K structure
    '''
    def __init__(self, hyp, valid=False, transform=None):
        '''
        :param hyp: hyperparameters dictionary
        :param valid: if True, use validation set, otherwise training set
        :param transform: Type of transformation (currently unused)
        '''
        self.transform = transform
        self.degrees = hyp["degrees"]
        self.translate = hyp["translate"]
        self.scale = hyp["scale"]
        self.shear = hyp["shear"]
        self.hgain = hyp["hgain"]
        self.sgain = hyp["sgain"]
        self.vgain = hyp["vgain"]
        self.Random_Crop = A.RandomCrop(width=hyp["width_crop"], height=hyp["height_crop"])

        self.prob_perspective = hyp["prob_perspective"]
        self.prob_flip = hyp["prob_flip"]
        self.prob_hsv = hyp["prob_hsv"]
        self.prob_bilateral = hyp["prob_bilateral"]
        self.prob_gaussian = hyp["prob_gaussian"]
        self.prob_crop = hyp["prob_crop"]
        
        self.Tensor = transforms.ToTensor()
        self.valid = valid
        
        if valid:
            self.root = hyp['val_dataset_path'] + '/images/val'
            self.names = os.listdir(self.root)
        else:
            self.root = hyp['dataset_path'] + '/images/train'
            self.names = os.listdir(self.root)

    def _resolve_mask_path(self, image_name):
        """
        Resolve corresponding drivable-area mask path for a given image path.
        Supports common export naming variants (same stem, no .rf hash stem, and mixed extensions).
        """
        image_path = Path(image_name)
        split = image_path.parent.name  # train / val / test
        mask_dir = image_path.parents[2] / 'drivable_area_annotations' / split

        # Primary candidate: same stem with .png
        candidates = [
            mask_dir / f"{image_path.stem}.png",
            mask_dir / f"{image_path.stem}.jpg",
            mask_dir / f"{image_path.stem}.jpeg",
        ]

        # If filename has Roboflow hash segment, try without it.
        if '.rf.' in image_path.stem:
            stem_no_hash = image_path.stem.split('.rf.')[0]
            candidates.extend([
                mask_dir / f"{stem_no_hash}.png",
                mask_dir / f"{stem_no_hash}.jpg",
                mask_dir / f"{stem_no_hash}.jpeg",
            ])

            # Fallback glob: any hashed version with same prefix
            glob_hits = sorted(glob.glob(str(mask_dir / f"{stem_no_hash}.rf.*.png")))
            if glob_hits:
                return glob_hits[0]

        for p in candidates:
            if p.exists():
                return str(p)

        return None

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        '''
        :param idx: Index of the image file
        :return: returns the image and corresponding label files (drivable area + dummy lane lines)
        '''
        W_ = 640
        H_ = 384
        image_name = os.path.join(self.root, self.names[idx])

        image = cv2.imread(image_name)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_name}")
        
        # Load drivable area mask
        mask_path = self._resolve_mask_path(image_name)
        if mask_path is None:
            raise FileNotFoundError(
                f"No mask found for image: {image_name}. Expected under drivable_area_annotations/{Path(image_name).parent.name}"
            )

        label1 = cv2.imread(mask_path, 0)
        if label1 is None:
            raise FileNotFoundError(f"Mask exists but is unreadable: {mask_path}")
        # Create dummy lane line mask (all zeros)
        label2 = np.zeros_like(label1)
        
        if not self.valid:
            if random.random() < self.prob_perspective:
                combination = (image, label1, label2)
                (image, label1, label2) = random_perspective(
                    combination=combination,
                    degrees=self.degrees,
                    translate=self.translate,
                    scale=self.scale,
                    shear=self.shear
                )
            if random.random() < self.prob_hsv:
                augment_hsv(image, self.hgain, self.sgain, self.vgain)
            if random.random() < self.prob_flip:
                image = np.fliplr(image)
                label1 = np.fliplr(label1)
                label2 = np.fliplr(label2)
            
            if random.random() < self.prob_bilateral:
                image = RandomBilateralBlur(image)
            if random.random() < self.prob_gaussian:
                image = RandomGaussianBlur(image)
            if random.random() < self.prob_crop:
                masks = np.stack([label1, label2], axis=2)
                transformed = self.Random_Crop(image=image, mask=masks)
                image = transformed['image']
                labels = transformed['mask']
                label1 = labels[:, :, 0]
                label2 = labels[:, :, 1]
            image = letterbox(image, (H_, W_))
        else:
            image = letterbox(image, (H_, W_))

        label1 = cv2.resize(label1, (W_, 360))
        label2 = cv2.resize(label2, (W_, 360))
        
        _, seg_b1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY_INV)
        _, seg_b2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY_INV)
        _, seg1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY)
        _, seg2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY)

        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        seg_b1 = self.Tensor(seg_b1)
        seg_b2 = self.Tensor(seg_b2)
        seg_da = torch.stack((seg_b1[0], seg1[0]), 0)
        seg_ll = torch.stack((seg_b2[0], seg2[0]), 0)
        image = np.array(image)
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)

        return image_name, torch.from_numpy(image), (seg_da, seg_ll)