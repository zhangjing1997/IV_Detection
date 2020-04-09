"""
This script tests the performance of two methods: unet and yolo, in terms of the distance between the predicted centroid and ground truth.
Note:
    - when the number of the predicted segmentations or detections are not equal to ground truth, the sample would not be counted into performance statistics.
"""
import os
import sys
import math
import argparse
import statistics

import numpy as np
from PIL import Image
import cv2

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

def findOutliers(data, threshold):
    """
    Identify the outliers of a data given as a list, using the z-score method.
    """
    data_mean, data_std = statistics.mean(data), statistics.stdev(data)
    cut_off = data_std * threshold
    lower, upper = data_mean - cut_off, data_mean + cut_off
    outliers = [(i, x) for i, x in enumerate(data) if x < lower or x > upper]

    return outliers

def getCentroidsFromMaskImage(pil_img, num_target_centroids=0, mode='pred'):
    """
    input: a binarized or just a mask image in PIL format.
    output: a list of tuples storing the coordinates in pixel of the white regions' centroids.
    """
    img = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(img, 127, 255, 0)
    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find contours in the binary image

    if mode == 'pred' and len(contours) != num_target_centroids:
        print('we got more or less segemented regions than target regions')
        return []
    
    centroids = []
    for c in contours:
        M = cv2.moments(c) # calculate moments for each contour
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroid = (cX, cY)
        centroids.append(centroid)
    
    return sorted(centroids, key=lambda x: x[0]) # sort centroids from left to right

def getCentroidsFromDetections(bboxes, num_target_centroids):
    """
    input: a torch tensor storing bounding boxes coordinates of an image.
    ouput: a list of tuples storing the coordinates in pixel of the boxes' centroids.
    """
    if bboxes is None:
        return []
    if bboxes.size()[0] != num_target_centroids:
        print('we got more or less detectde bounding boxes than target regions')
        return []
    
    centroids = []
    for x1, y1, x2, y2, _, _, _ in bboxes:
        cX = (x1 + x2) / 2
        cY = (y1 + y2) / 2
        centroid = ((int)(cX.item()), (int)(cY.item()))
        centroids.append(centroid)
    
    return sorted(centroids, key=lambda x: x[0]) # sort centroids from left to right

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
    if (len(centroids_pred) == 0):
        print('this sample does not count due to false positive or true negative predictions')
    else:
        for ct, cp in zip(centroids_target, centroids_pred):
            dist_list.append(euclideanDist(ct, cp))
    
    return dist_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='performance comparison between unet and yolo', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_name", type=str, default="phantom_20", help="the name of dataset used for training")
    parser.add_argument("--unet_ckpt", type=str, default="", help="path to unet weights file")
    parser.add_argument("--yolo_ckpt", type=str, default="", help="path to yolo weights file")
    parser.add_argument("--device", type=str, default="cuda", help="specify device: cuda or cpu")
    opt = parser.parse_args()

    logfile = 'logs/compare/compare' + '_' + opt.dataset_name + '_' + opt.device + '.log'
    sys.stdout = Logger(logfile)
    print(opt)

    image_folder = 'UNet/data/imgs/' + opt.dataset_name
    target_folder = 'UNet/data/masks/' + opt.dataset_name

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(opt.device)
    classes = ['vein']

    unet_path = 'UNet/checkpoints/' + opt.unet_ckpt
    model_unet = UNet(n_channels=1, n_classes=1).to(device=device)
    model_unet.load_state_dict(torch.load(unet_path)['state_dict'])
    model_unet.eval()

    yolo_path = 'YOLOv3/checkpoints/' + opt.yolo_ckpt
    model_yolo = Darknet('YOLOv3/config/yolov3-custom.cfg', img_size=416).to(device=device)
    model_yolo.load_state_dict(torch.load(yolo_path))
    model_yolo.eval()

    unet_target_errors_list = []
    yolo_target_errors_list = []
    unet_infer_time_list = []
    yolo_infer_time_list = []
    image_files = [x for x in os.listdir(image_folder) if x.endswith('.jpg')] # only jpg files
    for i, fn in enumerate(image_files):
        img_path = os.path.join(image_folder, fn)
        target_path = os.path.join(target_folder, fn)
        target_img = Image.open(target_path).convert('L')
        centroids_target = getCentroidsFromMaskImage(target_img, num_target_centroids=0, mode='target')

        # prediction
        print('\n----------------------')
        mask, inference_time_segment = segment_img(img_path, model_unet, device, scale_factor=1.0, out_threshold=0.5)
        _, detections_rescaled, inference_time_detect = detect_img(img_path, model_yolo, device, 416, classes, conf_thres=0.825, nms_thres=0.4, output_dir='', save_plot=False)
        unet_infer_time_list.append(inference_time_segment)
        yolo_infer_time_list.append(inference_time_detect)

        # get centroids from predictions
        centroids_unet = getCentroidsFromMaskImage(mask_to_image(mask), num_target_centroids=len(centroids_target), mode='pred')
        centroids_yolo = getCentroidsFromDetections(detections_rescaled, len(centroids_target))

        # compute error
        unet_target_errors = computecentroidsDiff(centroids_unet, centroids_target)
        yolo_target_errors = computecentroidsDiff(centroids_yolo, centroids_target)

        # logs: error and stats
        print(f'\ncentroids_target: {centroids_target}')
        print(f'centroids_unet: {centroids_unet}')
        print(f'centroids_yolo: {centroids_yolo}')
        print(f'unet_target_errors: {unet_target_errors}')
        print(f'yolo_target_errors: {yolo_target_errors}')

        if len(unet_target_errors) > 0:
            unet_target_errors_list.extend(unet_target_errors)
        
        if len(yolo_target_errors) > 0:
            yolo_target_errors_list.extend(yolo_target_errors)
    
    outliers_unet = findOutliers(unet_target_errors_list, 2)
    outliers_yolo = findOutliers(yolo_target_errors_list, 2)
    outliers_unet_ratio = len(outliers_unet) / len(unet_target_errors_list)
    outliers_yolo_ratio = len(outliers_yolo) / len(yolo_target_errors_list)

    print('\n======== Summary ==========')
    print(f'Using device: {device}')
    print(f'Dataset - {opt.dataset_name}: {image_folder}')
    print(f'Total samples: {len(image_files)}')
    print(f'Model Weights: ')
    print(f'\t- unet: {unet_path}')
    print(f'\t- yolo: {yolo_path}')
    print('outliers_unet:')
    print(outliers_unet)
    print('outliers_yolo:')
    print(outliers_yolo)
    print(f'Average prediction speed/time (s): ')
    print('\t- unet: {:.4f}'.format(statistics.mean(unet_infer_time_list)))
    print('\t- yolo: {:.4f}'.format(statistics.mean(yolo_infer_time_list)))
    print(f'Average centroid prediction error: ')
    print('\t- unet: {:.5f}'.format(statistics.mean(unet_target_errors_list)))
    print('\t- yolo: {:.5f}'.format(statistics.mean(yolo_target_errors_list)))
    print('Outliers ratio of centroid prediction errors: ')
    print('\t- unet: {}/{} = {:.5f}'.format(len(outliers_unet), len(unet_target_errors_list), outliers_unet_ratio))
    print('\t- yolo: {}/{} = {:.5f}'.format(len(outliers_yolo), len(yolo_target_errors_list), outliers_yolo_ratio))