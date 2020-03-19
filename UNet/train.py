import os
import sys
import argparse

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from unet_model import UNet
from dataset import BasicDataset
from test import evaluate
from utils import Logger

if __name__ == '__main__':
    ## --- Set and get args
    parser = argparse.ArgumentParser(description='Train the UNet on gray images with vein region as targets', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset_name", type=str, default="phantom_20", help="the name of dataset used for training")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="parent directory of saving checkpoint weights")
    parser.add_argument('--pretrained_weights', type=str, default=None, help='Load model from a .pth file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--down_scale', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='Percent of the data that is used as validation (0-100)')

    args = parser.parse_args()
    
    # console printing redirection
    logfile = f'logs/train/{args.dataset_name}_{args.epochs}ep_{args.batch_size}bs.log'
    sys.stdout = Logger(logfile)
    print(args)
    # tensorboard writer
    # writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')

    checkpoint_dir = args.checkpoints_dir + '/' + args.dataset_name
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## --- Set up model
    net = UNet(n_channels=1, n_classes=1).to(device) # we are using gray images and only have one labelled class
    if args.pretrained_weights:
        net.load_state_dict(torch.load(args.pretrained_weights))
        print(f'Pretrained weights loaded from {args.load}')
    print(f' Using device {device}\n Network:\n \t{net.n_channels} input channels\n \t{net.n_classes} output channels (classes)\n \t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    ## --- Set up data
    imgs_dir = 'data/imgs/' + args.dataset_name
    masks_dir = 'data/masks/' + args.dataset_name
    print(f'imgs_dir: {imgs_dir} masks_dir: {masks_dir}')
    dataset = BasicDataset(imgs_dir, masks_dir, args.down_scale)
    n_val = int(len(dataset) * args.valid_ratio)
    n_train = len(dataset) - n_val
    print(f'n_val: {n_val} n_train: {n_train}')
    train, val = random_split(dataset, [n_train, n_val]) #split into train dataset and validation dataset
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # logging training overview
    print('-----\n Start training:')
    print(f'epochs: {args.epochs} \t batch size: {args.batch_size} \t learning rate: {args.learning_rate} \t') 
    print(f'training size: {n_train} \t validation size: {n_val} \t checkpoints_dir: {args.checkpoints_dir} \t images downscale: {args.down_scale}')
    print('-----')
    
    ## --- Set up training
    global_step = 0
    optimizer = optim.RMSprop(net.parameters(), lr=args.learning_rate, weight_decay=1e-8)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    ## --- Start training
    epoch_loss_list = []
    val_score_list = []
    num_batches_per_epoch = len(dataset) // args.batch_size
    for epoch in range(args.epochs):
        net.train()

        epoch_loss = 0
        for batch in train_loader:
            imgs = batch['image'] # N x C x W x H
            true_masks = batch['mask']
            assert imgs.shape[1] == net.n_channels, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'
            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            ## -- Net forward
            masks_pred = net(imgs) # network feedforward
            loss = criterion(masks_pred, true_masks) # calculate loss
            epoch_loss += loss.item() # extract batch loss from torch tensor

            ## -- Training update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            # write images to tensoboard every 10 batches
            # if global_step % (len(dataset) // (10 * batch_size)) == 0:
            #     writer.add_images('images', imgs, global_step)
            #     if net.n_classes == 1:
            #         writer.add_images('masks/true', true_masks, global_step)
            #         writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
        
        epoch_loss_avg = epoch_loss / num_batches_per_epoch
        epoch_loss_list.append(epoch_loss_avg)
        val_score = evaluate(net, val_loader, device, n_val) # evaluation
        val_score_list.append(val_score)
        print('epoch: {:2d} \t training loss(cross_entropy): {:5f} \t validation score(dice coeff): {:5f}'.format(epoch, epoch_loss_avg, val_score))
        # writer.add_scalar('train_loss', epoch_loss_avg, epoch)
        # writer.add_scalar('val_score', val_score, epoch)

        # save checkpoints and history train_loss/val_score every 5 epochs
        if epoch % 2 == 0:
            torch.save({'state_dict': net.state_dict(), 'loss_list': epoch_loss_list, 'val_score_list': val_score_list}, checkpoint_dir + '/' + f'unet_ckpt_{epoch}.pth')
            epoch_loss_list = []
            val_score_list = []

    # writer.close()
