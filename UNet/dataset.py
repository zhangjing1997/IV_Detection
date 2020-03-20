import os
from os.path import splitext
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in os.listdir(imgs_dir) if file.endswith('.jpg')]
        logging.info(f'creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH)) # resize

        img_np = np.array(pil_img) # pil image to numpy array
        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=2) # HW to HWC
        img_trans = img_np.transpose((2, 0, 1)) # HWC to CHW
        if img_trans.max() > 1:
            img_trans = img_trans / 255 # normalization
        
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = os.path.join(self.masks_dir, idx + '.jpg')
        img_file = os.path.join(self.imgs_dir, idx + '.jpg')
        # assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        # assert len(img_file) == 1, f'Either no image or multiple images found for the ID {idx}: {img_file}'

        mask = Image.open(mask_file)
        img = Image.open(img_file).convert('L')

        assert img.size == mask.size, f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}