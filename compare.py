import os
import sys
import math

import numpy as np
from PIL import Image
import cv2

import torch

# sys.path.insert(0, './UNet/')
# sys.path.insert(0, './YOLOv3/')

# from UNet.unet_model import UNet
# from UNet.segment import segment_img, mask_to_image
# from YOLOv3.detect import detect_img
# from YOLOv3.yolo_model import Darknet

from UNet import UNet, segment_img, mask_to_image
from YOLOv3 import detect_img, Darknet

def getCentroidsFromMaskImage(img_path):
    """
    input: path to a gray image that is binarized or just a mask image.
    output: a list of tuples storing the coordinates in pixel of the white regions' centroids.
    """
    img = cv2.imread(img_path, 0) # 0 -> cv2.IMREAD_GRAYSCALE
    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find contours in the binary image
    centroids = []
    for c in contours:
        M = cv2.moments(c) # calculate moments for each contour
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroids.append((cX, cY))
    
    sorted(centroids, key=lambda x: x[0]) # sort centroids from left to right

    return centroids

def getCentroidsFromDetections(bboxes):
    """
    input: a torch tensor storing bounding boxes coordinates of an image.
    ouput: a list of tuples storing the coordinates in pixel of the boxes' centroids.
    """
    if len(bboxes) == 0:
        return []
    
    centroids = []
    for x1, y1, x2, y2, _, _, _ in bboxes:
        cX = (x1 + x2) / 2
        cY = (y1 + y2) / 2
        centroids.append((cX, cY))
    
    sorted(centroids, key=lambda x: x[0]) # sort centroids from left to right

    return centroids

def euclideanDist(coord1, coord2):
    """
    input: two tuples storing two coordinates in pixels.
    output: euclidean distance between them.
    """
    dist = math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)
    return dist

def computecentroidsDiff(centroids_target, centroids_pred):
    """
    input: coordinates of target and predicted vein' centroids
    ouput: a list of euclidean distance between corresponded target and predict centroids
    """
    dist_list = []
    len_target = len(centroids_target)
    len_pred = len(centroids_pred)

    if (len_pred < len_target):
        print('there exists veins that not detected or segmented!')
    elif (len_pred > len_target):
        print('there exists regions that should not be detected or segmented as veins!')
    else:
        for ct, cp in zip(centroids_target, centroids_pred):
            dist_list.append(euclideanDist(ct, cp))
    
    return dist_list


if __name__ == "__main__":
    image_folder = 'UNet/data/imgs/phantom_20'
    target_folder = 'UNet/data/masks/phantom_20'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ['vein']

    model_unet = UNet(n_channels=1, n_classes=1).to(device=device)
    model_unet.load_state_dict(torch.load('UNet/checkpoints/phantom_20/unet_ckpt_26.pth')['state_dict'])
    model_unet.eval()

    model_yolo = Darknet('YOLOv3/config/yolov3-custom.cfg', img_size=416).to(device=device)
    model_yolo.load_state_dict(torch.load('YOLOv3/checkpoints/phantom_20/yolov3_ckpt_38.pth'))
    model_yolo.eval()

    for i, fn in enumerate(os.listdir(image_folder)):
        if not fn.endswith('.jpg'):
            continue
        # input image and target
        img_path = os.path.join(image_folder, fn)
        target_path = os.path.join(target_folder, fn)
        centroids_target = getCentroidsFromMaskImage(target_path)

        # prediction
        mask, _ = segment_img(img_path, model_unet, device, scale_factor=1.0, out_threshold=0.5)
        _, detections_rescaled, _ = detect_img(img_path, model_yolo, device, 416, classes, conf_thres=0.85, nms_thres=0.35, output_dir='', save_plot=False)

        # get centroids from predictions
        centroids_unet = getCentroidsFromMaskImage(mask_to_image(mask))
        centroids_yolo = getCentroidsFromDetections(detections_rescaled)

        # compute error
        unet_target_dist_list = computecentroidsDiff(centroids_unet, centroids_target)
        yolo_target_dist_list = computecentroidsDiff(centroids_yolo, centroids_target)

        # logs: error and stats
        print(f'centroids_target: {centroids_target}')
        print(f'centroids_unet: {centroids_unet}')
        print(f'centroids_yolo: {centroids_yolo}')
        print(f'unet_target_dist: {unet_target_dist_list}')
        print(f'yolo_target_dist: {yolo_target_dist_list}')