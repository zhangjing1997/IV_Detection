from __future__ import division
import os
import sys
import time
import datetime
import argparse
import random
random.seed(400)
import statistics

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from yolo_model import Darknet
from utils import load_classes, rescale_boxes, non_max_suppression, Logger, xywh2xyxy, get_batch_statistics, ap_per_class
from datasets import ImageFolder, ListDataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

def plot_img_and_bbox(img_path, detections, detect_size, class_names, output_dir, save_plot=True):
    img_input = Image.open(img_path) # pil image
    img = np.array(img_input) # numpy array

    cmap = plt.get_cmap("tab20b")
    color = cmap(13)
    fig, ax = plt.subplots(1)
    ax.imshow(img, cmap='gray')

    if detections is None:
        print('No veins detected on this image!')
    else:
        detections = rescale_boxes(detections, detect_size, img.shape[:2]) # rescale bbox into original image size
        print(f"Saving image with bbox in original image size - ({img.shape[:2]}):")
        for x1, y1, x2, y2, obj_conf, cls_conf, cls_pred in detections:
            print("\t+ Label: %s, x1: %.3f, y1: %.3f, x2: %.3f, y2: %.3f, obj_conf: %.5f, cls_conf: %.5f" % 
                (class_names[int(cls_pred)], x1, y1, x2, y2, obj_conf.item(), cls_conf.item()))
            x1_draw = max(x1, 1)
            y1_draw = max(y1, 1)
            x2_draw = min(x2, img.shape[1] - 1)
            y2_draw = min(y2, img.shape[0] - 1)

            # add bbox
            bbox = patches.Rectangle((x1_draw, y1_draw), x2_draw - x1_draw, y2_draw - y1_draw, linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(bbox)
            # add label
            if x1 > 0:
                plt.text(x1, y1, s=classes[int(cls_pred)], color="white", verticalalignment="bottom", horizontalalignment='left', bbox={"color": color, "pad": 0})
            elif x2 < img.shape[1]:
                plt.text(x2, y2, s=classes[int(cls_pred)], color="white", verticalalignment="top", horizontalalignment='right', bbox={"color": color, "pad": 0})
        
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())

        # save image plot with detected bboxes
        if save_plot:
            filename = img_path.split('/')[-1].split('.')[0]
            output_path = f"{output_dir}/{filename}.jpg"
            fig.savefig(output_path, bbox_inches="tight", pad_inches=0.0)

            # added: check saved image size and resize + replace if not consistent
            img_output = Image.open(output_path)
            if img_output.size != img_input.size:
                img_output = img_output.resize(img_input.size)
                img_output.save(output_path)
            print(f'saved image plot with detected bboxes into ---> {output_path}')

        plt.close(fig)


def detect_img(img_path, model, device, detect_size, class_names, conf_thres, nms_thres, output_dir, save_plot=True):
    """
    Using the given trained model to detect veins on the image (specified by img_path).
        - model: yolo loaded with checkpoint weights
        - img_path: absolute path to the image
        - detect_size: input image size of the model
        - class_names: a list of target class names
    """
    print(f'\nPerforming object detection on ---> {img_path} \t')
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    img = ImageFolder(folder_path='', img_size=detect_size).preprocess(img_path)
    img = img.unsqueeze(0).type(FloatTensor)

    begin_time = time.time()
    with torch.no_grad():
        detections = model(img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
    end_time = time.time()
    # print(f'detections: {detections}')

    inference_time = end_time - begin_time
    print(f'inference_time: {inference_time}s')

    if detections[0] is not None:
        # it is a list due to the implementation of non_max_suppression that deal with batch samples
        plot_img_and_bbox(img_path, detections[0].clone(), detect_size, class_names, output_dir, save_plot)
    else:
        print('No veins detected on this image!')
    
    return detections, inference_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="phantom_20", help="the name of dataset used for test")
    parser.add_argument("--output_dir", type=str, default="output", help="parent directory of saving test results")
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--save_plot", type=int, default=1, help="whether save plotted results")

    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.85, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.35, help="iou thresshold for non-maximum suppression")

    opt = parser.parse_args()

    ckpt_str = opt.weights_path.split('/')[-1][:-4]
    sys.stdout = Logger(f'logs/detect/{opt.dataset_name}_{ckpt_str}.log') # logfile to save this script printing
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = opt.output_dir + '/' + opt.dataset_name
    os.makedirs(output_dir, exist_ok=True)

    ### --- Model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path) # load darknet weights
    else:
        model.load_state_dict(torch.load(opt.weights_path)) # load checkpoint weights
    model.eval()  # set in evaluation mode

    count = 0
    inference_time_total = 0.0
    iou_total = 0.0
    confidence_total = 0.0
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred, ious)
    classes = load_classes(opt.class_path)  # extracts class labels from file
    image_files = os.listdir(opt.image_folder)
    for i, fn in enumerate(image_files):
        if not fn.endswith('.jpg'):
            continue
        # input image
        img_path = os.path.join(opt.image_folder, fn)

        # single image prediction -> plot bbox and save
        detections, inference_time = detect_img(img_path, model, device, opt.img_size, classes, opt.conf_thres, opt.nms_thres, output_dir, opt.save_plot)

        # label/targets
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        _, targets = ListDataset(opt.dataset_name, 'data/custom/train_valid/phantom_20/valid.txt').preprocess(img_path, label_path)
        labels += targets[:, 1].tolist()
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= opt.img_size # rescale target

        # prediction speed
        inference_time_total += inference_time

        # validation score computation
        sample_metrics += get_batch_statistics(detections, targets, iou_threshold=opt.iou_thres)
        ious = sample_metrics[-1][-1]
        if detections[0] is not None:
            count += 1
            print(f'Showing image with bbox in detected image size - ({opt.img_size}, {opt.img_size})')
            conf_list = []
            ious_list = []
            for i, ((x1, y1, x2, y2, obj_conf, cls_conf, cls_pred), iou) in enumerate(zip(detections[0], ious)):
                conf_list.append(obj_conf.item())
                ious_list.append(iou.item())
                print("\t+ Label: %s, x1: %.3f, y1: %.3f, x2: %.3f, y2: %.3f, iou: %.5f" % (classes[int(cls_pred)], x1, y1, x2, y2, iou.item()))
            confidence_total += statistics.mean(conf_list)
            iou_total += statistics.mean(ious_list)
    
    # performance statistics
    true_positives, pred_scores, pred_labels, ious = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))] # pred_scores -> obj_conf for detected boxes
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    
    print(f'\n----- YOLOv3 Performance Summary on {opt.dataset_name} -----')
    print(f'Using device: {device}')
    print(f'Using model: {opt.weights_path}')
    print(f'Total samples: {len(image_files)} detected samples: {count}')
    print('Precision: {:.3f} Recall: {:.3f} AP: {:.3f} f1: {:.3f} ap_class: {}'.format(precision.mean(), recall.mean(), AP.mean(), f1.mean(), ap_class))
    print('Total inference time: {:.5f}s'.format(inference_time_total))
    print('Average inference time: {:.5f}s Average prediction confidence: {:.3f} Average iou: {:.3f}'.format(inference_time_total / len(image_files), confidence_total / count, iou_total / count))
    print('-----------------------')

    # dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    # img_paths_list = []  # stores image paths
    # img_detections_list = []  # stores detections for each image index

    # print("\nPerforming object detection:")
    # prev_time = time.time()
    # for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    #     input_imgs = Variable(input_imgs.type(Tensor)) # configure input

    #     # Model Prediction - get predicted detections
    #     with torch.no_grad():
    #         img_detections = model(input_imgs)
    #         img_detections = non_max_suppression(img_detections, opt.conf_thres, opt.nms_thres)

    #     # log progress
    #     current_time = time.time()
    #     inference_time = datetime.timedelta(seconds=current_time - prev_time) # from load image to finish inferencing
    #     prev_time = current_time
    #     print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

    #     # save image and detections
    #     img_paths_list.extend(img_paths) # 用extend是为了保证list的element是single image，而不是batch images
    #     img_detections_list.extend(img_detections)

    #### ----- drawing and save -----
    # print("\nSaving images:")
    # cmap = plt.get_cmap("tab20b")
    # colors = [cmap(i) for i in np.linspace(0, 1, 20)] # bounding-box colors
    
    # # Iterate through images and save plot of detections
    # for img_i, (path, detections) in enumerate(zip(img_paths_list, img_detections_list)):
    #     # if img_i == 1:
    #     #     break
    #     print("(%d) Image: '%s'" % (img_i, path))
        
    #     # Create plot
    #     img_input = Image.open(path)
    #     img = np.array(img_input)
    #     fig, ax = plt.subplots(1)
    #     ax.imshow(img, cmap='gray')
    #     # Draw bounding boxes and labels of detections
    #     if detections is not None:
    #         detections = rescale_boxes(detections, opt.img_size, img.shape[:2]) # rescale boxes to original image
    #         unique_labels = detections[:, -1].cpu().unique()
    #         n_cls_preds = len(unique_labels)
    #         bbox_colors = random.sample(colors, n_cls_preds)

    #         for x1, y1, x2, y2, obj_conf, cls_conf, cls_pred in detections:
    #             print("\t+ Label: %s, obj_conf: %.5f, cls_conf: %.5f" % (classes[int(cls_pred)], obj_conf.item(), cls_conf.item()))
    #             x1_draw = max(x1, 1)
    #             y1_draw = max(y1, 1)
    #             x2_draw = min(x2, img.shape[1] - 1)
    #             y2_draw = min(y2, img.shape[0] - 1)

    #             color = cmap(13)
    #             # add bbox
    #             bbox = patches.Rectangle((x1_draw, y1_draw), x2_draw - x1_draw, y2_draw - y1_draw, linewidth=2, edgecolor=color, facecolor="none")
    #             ax.add_patch(bbox)
    #             # add label
    #             if x1 > 0:
    #                 plt.text(x1, y1, s=classes[int(cls_pred)], color="white", verticalalignment="bottom", horizontalalignment='left', bbox={"color": color, "pad": 0})
    #             elif x2 < img.shape[1]:
    #                 plt.text(x2, y2, s=classes[int(cls_pred)], color="white", verticalalignment="top", horizontalalignment='right', bbox={"color": color, "pad": 0})

    #     # Save generated image with detections
    #     plt.axis("off")
    #     plt.gca().xaxis.set_major_locator(NullLocator())
    #     plt.gca().yaxis.set_major_locator(NullLocator())
    #     filename = path.split("/")[-1].split(".")[0]
    #     fig.savefig(f"{output_dir}/{filename}.jpg", bbox_inches="tight", pad_inches=0.0)
    #     plt.close(fig)

    #     # Added: Check saved image size and resize + replace if necessary
    #     img_output = Image.open(f"{output_dir}/{filename}.jpg")
    #     if img_output.size != img_input.size:
    #         img_output = img_output.resize(img_input.size)
    #         img_output.save(f"{output_dir}/{filename}.jpg")