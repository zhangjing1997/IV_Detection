from __future__ import division
import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from yolo_model import Darknet
from utils import *
# parse_data_config, load_classes, non_max_suppression, get_batch_statistics, ap_per_class, Logger, xywh2xyxy
from datasets import ListDataset


def evaluate(model, dataset_name, list_path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    """ Evaluate with the given model on the dataset specified by the given dataset_path to list of samples """
    model.eval()

    # Get dataloader
    dataset = ListDataset(dataset_name, list_path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (img_paths, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Validation round")):
        print(img_paths)
        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        labels += targets[:, 1].tolist() # extract labels
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size # rescale target

        # model forward
        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        
        # evaluation stats
        print(f'outputs: {outputs}')
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="phantom_20", help="the name of dataset used for test")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")

    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")

    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    # parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.85, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.35, help="iou thresshold for non-maximum suppression")

    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.dataset_name, opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path) # load darknet weights
    else:
        model.load_state_dict(torch.load(opt.weights_path)) # load checkpoint weights

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        dataset_name=opt.dataset_name,
        list_path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
