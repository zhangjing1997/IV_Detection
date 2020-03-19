import os
import sys
import time
import argparse

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from unet_model import UNet
from dataset import BasicDataset
from utils import plot_img_and_mask, Logger

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def segment_img(img_path, net, device, scale_factor=1, out_threshold=0.5):
    """ 
    Segement out the mask (target vein region) of the input image (specified by image path) 
    """
    print(f"\nPerforming segmentation on ---> {img_path} ...")

    img = torch.from_numpy(BasicDataset.preprocess(Image.open(img_path).convert('L'), scale_factor))
    img = img.unsqueeze(0) # add dimension
    img = img.to(device=device, dtype=torch.float32)

    begin_time = time.time()
    with torch.no_grad():
        output = net(img)
    end_time = time.time()
    inference_time = end_time - begin_time
    print(f'inference_time: {inference_time}s')

    # transform to image numpy array
    if net.n_classes > 1:
        probs = F.softmax(output, dim=1)
    else:
        probs = torch.sigmoid(output)
    probs = probs.squeeze(0) #cuda tensor
    
    tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            # transforms.Resize(full_img.size[1]),
            transforms.ToTensor()
        ]
    )
    probs = tf(probs.cpu())
    full_mask = probs.squeeze().cpu().numpy()
    if np.count_nonzero(full_mask) == 0:
        print("No veins segmented out on this image!")

    return full_mask > out_threshold, inference_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict masks from input images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset_name", type=str, default="phantom_20", help="the name of dataset used for segementation")
    parser.add_argument('--weights_path', type=str, default='unet_ckpt_xx.pth', help="path to weights file")
    parser.add_argument('--image_folder', type=str, default='data/imgs', help='parent of image folder storing images to do segmentation')
    parser.add_argument('--output_dir', type=str, default='output', help='parent of path to parent saving segmentation results')

    parser.add_argument('--viz', type=int, default=0, help="Visualize the images as they are processed")
    parser.add_argument('--save_results', type=int, default=0, help="Do not save the output masks")
    parser.add_argument('--mask_threshold', type=float, default=0.5, help="Minimum probability value to consider a mask pixel white")
    parser.add_argument('--scale', type=float, default=1.0, help="downscale factor for the input images")

    args = parser.parse_args()

    ckpt_str = args.weights_path.split('/')[-1][:-4]
    logfile = f'logs/segment/{args.dataset_name}_{ckpt_str}.log'
    sys.stdout = Logger(logfile)
    print(args)

    output_dir = args.output_dir + '/' + args.dataset_name
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = UNet(n_channels=1, n_classes=1).to(device)
    net.load_state_dict(torch.load(args.weights_path)['state_dict'])
    net.eval()

    inference_time_total = 0
    count = 0
    image_folder = args.image_folder + '/' + args.dataset_name
    for i, fn in enumerate(os.listdir(image_folder)):
        if not fn.endswith('.jpg'):
            continue
        count += 1
        # single image prediction
        img_path = os.path.join(image_folder, fn)
        mask, inference_time = segment_img(img_path, net, device, args.scale, args.mask_threshold)
        inference_time_total += inference_time

        if args.save_results:
            output_path = os.path.join(output_dir, fn)
            mask_to_image(mask).save(output_path)
            print(f"saved predicted mask to ---> {output_path}")
                
        if args.viz:
            print("Visualizing results for image {}, close to continue ...".format(fn))
            img = Image.open(fn)
            plot_img_and_mask(img, mask)
    
    print(f'\n----- UNet Performance Summary on {args.dataset_name} -----')
    print(f'Using device: {device}')
    print(f'Using model: {args.weights_path}')
    print(f'Total samples: {count}')
    print(f'Total inference time: {inference_time_total}s \t Average inference time: {inference_time_total / count}s')
    print('-----------------------')