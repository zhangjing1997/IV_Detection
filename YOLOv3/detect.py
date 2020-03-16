from __future__ import division

from yolo_model import Darknet
from utils import load_classes, rescale_boxes, non_max_suppression
from datasets import ImageFolder

import os
import sys
import time
import datetime
import argparse
import random
random.seed(400)

from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.85, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.35, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok=True)

    #### ----- Set up model and dataloader -----
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path) # load darknet weights
    else:
        model.load_state_dict(torch.load(opt.weights_path)) # load checkpoint weights
    model.eval()  # set in evaluation mode
    dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    classes = load_classes(opt.class_path)  # extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


    #### ----- Inferencing / Prediction -----
    img_paths_list = []  # stores image paths
    img_detections_list = []  # stores detections for each image index
    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        input_imgs = Variable(input_imgs.type(Tensor)) # configure input

        # Model Prediction - get predicted detections
        with torch.no_grad():
            img_detections = model(input_imgs)
            img_detections = non_max_suppression(img_detections, opt.conf_thres, opt.nms_thres)

        # log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time) # from load image to finish inferencing
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # save image and detections
        img_paths_list.extend(img_paths) # 用extend是为了保证list的element是single image，而不是batch images
        img_detections_list.extend(img_detections)

    #### ----- drawing and save -----
    print("\nSaving images:")
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)] # bounding-box colors
    
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(img_paths_list, img_detections_list)):
        # if img_i == 1:
        #     break
        print("(%d) Image: '%s'" % (img_i, path))
        
        # Create plot
        img_input = Image.open(path)
        img = np.array(img_input)
        fig, ax = plt.subplots(1)
        ax.imshow(img, cmap='gray')
        # Draw bounding boxes and labels of detections
        if detections is not None:
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2]) # rescale boxes to original image
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)

            for x1, y1, x2, y2, obj_conf, cls_conf, cls_pred in detections:
                print("\t+ Label: %s, obj_conf: %.5f, cls_conf: %.5f" % (classes[int(cls_pred)], obj_conf.item(), cls_conf.item()))
                x1_draw = max(x1, 1)
                y1_draw = max(y1, 1)
                x2_draw = min(x2, img.shape[1] - 1)
                y2_draw = min(y2, img.shape[0] - 1)

                color = cmap(13)
                # add bbox
                bbox = patches.Rectangle((x1_draw, y1_draw), x2_draw - x1_draw, y2_draw - y1_draw, linewidth=2, edgecolor=color, facecolor="none")
                ax.add_patch(bbox)
                # add label
                if x1 > 0:
                    plt.text(x1, y1, s=classes[int(cls_pred)], color="white", verticalalignment="bottom", horizontalalignment='left', bbox={"color": color, "pad": 0})
                elif x2 < img.shape[1]:
                    plt.text(x2, y2, s=classes[int(cls_pred)], color="white", verticalalignment="top", horizontalalignment='right', bbox={"color": color, "pad": 0})
                # plt.text(x1, y1, s=classes[int(cls_pred)], color="white", verticalalignment="top", bbox={"color": color, "pad": 0}) # add label

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        fig.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)

        # Added: Check saved image size and resize + replace if necessary
        img_output = Image.open(f"output/{filename}.png")
        if img_output.size != img_input.size:
            img_output = img_output.resize(img_input.size)
            img_output.save(f"output/{filename}.png")