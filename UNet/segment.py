import os
import sys
import argparse

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from unet_model import UNet
from dataset import BasicDataset
from utils import plot_img_and_mask, Logger

# def get_output_filenames(args):
#     in_files = args.input
#     out_files = []

#     if not args.output: #如果没有定义output file名称的话，就主动设置成带有OUT的名称。
#         for f in in_files:
#             pathsplit = os.path.splitext(f)
#             out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
#     elif len(in_files) != len(args.output): #check先自定义的outputfile名称数量是否和inputfile名称数量一致
#         logging.error("Input files and output files are not of the same length")
#         raise SystemExit()
#     else:
#         out_files = args.output

#     return out_files

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def predict_img(net, img_path, device, scale_factor=1, out_threshold=0.5):
    """ do predict the mask of an input image (PIL format) """
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(Image.open(img_path).convert('L'), scale_factor))
    img = img.unsqueeze(0) # add dimension
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img) #feed-forward prediction
        # print(output.size())

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)
        probs = probs.squeeze(0) #cuda tensor
        # print(probs.size())
        
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


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

    ckpt_str = args.weights_path.split('/')[2][:-4]
    logfile = f'logs/segment/{args.dataset_name}_{ckpt_str}.log'
    sys.stdout = Logger(logfile)
    print(args)

    output_dir = args.output_dir + '/' + args.dataset_name
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    net = UNet(n_channels=1, n_classes=1).to(device)
    net.load_state_dict(torch.load(args.weights_path)['state_dict'])
    print(f'Loading model from: {args.weights_path}')

    image_folder = args.image_folder + '/' + args.dataset_name
    for i, fn in enumerate(os.listdir(image_folder)):
        if not fn.endswith('.jpg'):
            continue
        print("\nSegment image {} ...".format(fn))
        mask = predict_img(net=net, img_path=os.path.join(image_folder, fn), scale_factor=args.scale, out_threshold=args.mask_threshold, device=device)

        if args.save_results:
            out_fn = os.path.join(output_dir, fn)
            mask_to_image(mask).save(out_fn)
            print("Mask saved to {}".format(out_fn))
                
        if args.viz:
            print("Visualizing results for image {}, close to continue ...".format(fn))
            img = Image.open(fn)
            plot_img_and_mask(img, mask)
