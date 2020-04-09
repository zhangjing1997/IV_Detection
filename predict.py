"""
This script utilizes the trained model of UNet and YOLOv3 to predict the segmentation and detection results respectively.
"""
import os
import sys
import argparse

import torch

from UNet.unet_model import UNet
from UNet.segment import segment_img, mask_to_image
from YOLOv3.yolo_model import Darknet
from YOLOv3.detect import detect_img

class Logger(object):
    """
    make pring statement simultaneously print onto console and into a logfile.
    """
    def __init__(self, logfile_name):
        self.terminal = sys.stdout
        self.log = open(logfile_name, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility. this handles the flush command by doing nothing. you might want to specify some extra behavior here.
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='performance comparison between unet and yolo', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_name", type=str, default="phantom_20", help="the image dataset name that we want to predict on")
    parser.add_argument("--unet_ckpt", type=str, default="", help="path to unet weights file")
    parser.add_argument("--yolo_ckpt", type=str, default="", help="path to yolo weights file")
    parser.add_argument("--device", type=str, default="cuda", help="specify device: cuda or cpu")
    parser.add_argument("--output_dir", type=str, default="output", help="directory saving prediction results")
    opt = parser.parse_args()

    logfile = 'logs/predict/' + opt.dataset_name + '_' + opt.device + '.log'
    sys.stdout = Logger(logfile)
    print(opt)

    device = torch.device(opt.device)

    output_dir_unet = opt.output_dir + '/unet_segmentation/' + opt.dataset_name
    os.makedirs(output_dir_unet, exist_ok=True)
    unet_path = 'UNet/checkpoints/' + opt.unet_ckpt
    model_unet = UNet(n_channels=1, n_classes=1).to(device=device)
    model_unet.load_state_dict(torch.load(unet_path)['state_dict'])
    model_unet.eval()

    output_dir_yolo = opt.output_dir + '/yolo_detection/' + opt.dataset_name
    os.makedirs(output_dir_yolo, exist_ok=True)
    classes = ['vein']
    yolo_path = 'YOLOv3/checkpoints/' + opt.yolo_ckpt
    model_yolo = Darknet('YOLOv3/config/yolov3-custom.cfg', img_size=416).to(device=device)
    model_yolo.load_state_dict(torch.load(yolo_path))
    model_yolo.eval()

    image_folder = 'DATA/' + opt.dataset_name + '/imgs'
    image_files = [x for x in os.listdir(image_folder) if x.endswith('.jpg')] # only jpg files
    for i, fn in enumerate(image_files):
        img_path = os.path.join(image_folder, fn)

        print('\n----------------------')
        # unet segmentation
        mask, inference_time_segment = segment_img(img_path, model_unet, device, scale_factor=1.0, out_threshold=0.5)
        output_path = os.path.join(output_dir_unet, fn)
        mask_to_image(mask).save(output_path)
        print(f"saved predicted mask to ---> {output_path}")

        # yolo detection
        _, detections_rescaled, inference_time_detect = detect_img(img_path, model_yolo, 
                                                                   device, 416, classes, conf_thres=0.825, 
                                                                   nms_thres=0.4, output_dir=output_dir_yolo, save_plot=True)