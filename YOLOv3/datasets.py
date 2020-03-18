import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape # 1 x 747 x 504
    dim_diff = np.abs(h - w)

    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2 # (top / left) padding and (bottom / right) padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0) # Determine padding
    img = F.pad(img, pad, "constant", value=pad_value) # Add padding -> 747 x 747

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images

def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = transforms.ToTensor()(Image.open(img_path).convert('L')) # extract image as PyTorch tensor: 1 x H x W
        img, _ = pad_to_square(img, 0) # pad to square resolution

        # # Handle images with less than 3 channels
        # if img.shape[0] == 1:
        #     shape = [3]
        #     shape.extend(list(img.shape[1:]))
        #     img = img.expand(shape) # add two more channels by repeating
        
        img = resize(img, self.img_size) # Resize -> 1 x 1 x 416 x 416

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, dataset_name, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            # eg. 0.jpg -> data/custom/images/{dataset_name}/0.jpg
            lines = [x.strip() for x in file.readlines()]
            self.img_files = [f'data/custom/images/{dataset_name}/' + x for x in lines]
            self.label_files = [f'data/custom/labels/{dataset_name}/' + x[:-4] + '.txt' for x in lines]
        
        # self.label_files = [path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt") for path in self.img_files] # get corresponding label file names
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):
        ### --- get image ---
        img_path = self.img_files[index % len(self.img_files)] 
        img = transforms.ToTensor()(Image.open(img_path).convert('L'))
        # img = transforms.ToTensor()(Image.open(img_path))
        # # Handle images with less than 3 channels
        # if len(img.shape) != 3:
        #     img = img.unsqueeze(0) # insert one dimension in axis 0
        #     img = img.expand((3, img.shape[1:])) # add two more channels by repeating
        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        img, pad = pad_to_square(img, 0) # pad to square resolution
        _, padded_h, padded_w = img.shape

        ### --- get label ---
        label_path = self.label_files[index % len(self.img_files)] 
        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            
            # Adjust for added padding
            x1 += pad[0] # padding_left
            y1 += pad[2] # padding_top
            x2 += pad[1] # padding_right
            y2 += pad[3] # padding_bottom
            
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))

        # remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None] 
        
        # add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        
        # selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        
        # resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)