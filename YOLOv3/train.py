from __future__ import division
import os
import sys
import time
import datetime
import argparse

from terminaltables import AsciiTable

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from yolo_model import Darknet
from utils import load_classes, parse_data_config, weights_init_normal, Logger
from datasets import ListDataset
from test import evaluate

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    # Set and get args
    parser = argparse.ArgumentParser(description='Train YOLO on gray images with vein bboxes as targets', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset_name", type=str, default="phantom_20", help="the name of dataset used for training")

    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="parent directory of saving checkpoint weights")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")

    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")

    opt = parser.parse_args()

    logfile = f'logs/train/{opt.dataset_name}_{opt.epochs}ep_{opt.batch_size}bs.log'
    sys.stdout = Logger(logfile) # logfile to save this script printing
    print(opt)

    ### --- Set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')
    checkpoint_dir = opt.checkpoints_dir + '/' + opt.dataset_name
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights)) # load checkpoint model
        else:
            model.load_darknet_weights(opt.pretrained_weights) # load darknet backbone
        print(f'Pretrained weights loaded from {opt.pretrained_weights}')

    ### --- Set up data
    data_config = parse_data_config(opt.dataset_name, opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    print(f' train_path: {train_path}\n valid_path: {valid_path}\n class_names: {data_config["names"]}')
    dataset = ListDataset(opt.dataset_name, train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, pin_memory=True, collate_fn=dataset.collate_fn)

    ### --- Set up training
    optimizer = torch.optim.Adam(model.parameters()) # learning rate等是按照默认的来了
    metrics = ["grid_size", "loss", "x", "y", "w", "h", "conf", "cls", "cls_acc", "recall50", "recall75", "precision", "conf_obj", "conf_noobj"]

    ### --- Start training
    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()

        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i # 总的batch number
            ## -- Net Forward
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            optimizer.zero_grad()
            loss.backward() # gradient calculation
            if batches_done % opt.gradient_accumulations: # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()
            
            ## -- Log progress
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"

                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                # logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # ETA calculating
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1)) # 用当前batch的running tim来估计ETA
            log_str += f"\n---- ETA {time_left}"

            # print logstr when finishing current epoch
            if batch_i == len(dataloader) - 1:
                print(log_str)

            model.seen += imgs.size(0)
        
        ## -- Evaluate the model on the validation set
        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            precision, recall, AP, f1, ap_class = evaluate(model, dataset_name=opt.dataset_name, list_path=valid_path, iou_thres=0.5, conf_thres=0.5, nms_thres=0.5, img_size=opt.img_size, batch_size=8)
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            # logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
        
        ## -- Save checkpoints
        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"{checkpoint_dir}/yolov3_ckpt_%d.pth" % epoch)