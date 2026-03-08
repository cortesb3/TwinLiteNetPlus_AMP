import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class AMPDataset(Dataset):
    def __init__(self, root, img_size=(640, 360), transform=None, valid=False):
        self.root = root
        self.img_size = img_size
        self.transform = transform
        self.valid = valid
        
        # Get file lists and ensure they match
        img_dir = os.path.join(root, "images")
        label_dir = os.path.join(root, "labels")
        
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            raise FileNotFoundError(f"Required directories not found in {root}")
            
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
        
        # Verify matching files
        if len(self.img_files) != len(self.label_files):
            print(f"Warning: Mismatch - {len(self.img_files)} images, {len(self.label_files)} labels")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load Image
        img_path = os.path.join(self.root, "images", self.img_files[idx])
        
        # Handle missing files gracefully
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # Load Label and Create Mask
        mask = np.zeros((h, w), dtype=np.uint8)
        label_path = os.path.join(self.root, "labels", self.label_files[idx])
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    parts = list(map(float, line.split()))
                    if len(parts) < 7:  # Need at least class + 3 points (6 coords)
                        continue
                    # Ignore class ID (parts[0]) and get normalized coords
                    coords = np.array(parts[1:]).reshape(-1, 2)
                    # Denormalize coordinates to pixel values
                    coords[:, 0] *= w
                    coords[:, 1] *= h
                    # Draw the polygon on the mask
                    cv2.fillPoly(mask, [coords.astype(np.int32)], 1)

        # Resize both to model input size
        img = cv2.resize(img, self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # Apply transforms if provided
        if self.transform and not self.valid:
            # Apply transforms here if needed
            pass

        # Convert to Tensors
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        return img_path, img, mask