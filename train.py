import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from utils.voc_2007_loader import pascalVOCLoader as data_loader
from utils.loss import get_loss_function

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'

voc_data_path = '/workspace/Dataset/VOC/VOCdevkit/VOC2012/'
sbd_path = '/workspace/Dataset/VOC/benchmark_RELEASE/'


def train_net(load_pth,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    # data load for Carvana
    # dataset = BasicDataset(dir_img, dir_mask, img_scale)
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train, val = random_split(dataset, [n_train, n_val])
    # train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    # data load for VOC2007
    t_loader = data_loader(
        voc_data_path,
        sbd_path,
        is_transform=True,
        split='train_aug',
        img_size=('same', 'same')
    )
    v_loader = data_loader(
        voc_data_path,
        sbd_path,
        is_transform=True,
        split='val',
        img_size=('same', 'same')
    )

    trainloader = DataLoader(
        t_loader,
        batch_size=1,
        num_workers=16,
        shuffle=True
    )
    valloader = DataLoader(
        v_loader, batch_size=1, num_workers=16
    )

    # init model
    net = UNet(n_channels=3, n_classes=t_loader.n_classes, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    if load_pth:
        net.load_state_dict(
            torch.load(load_pth, map_location=device)
        )
        logging.info(f'Model loaded from {load_pth}')
    net = net.to(device=device)
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    n_train = len(t_loader)
    n_val = len(v_loader)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.SGD(net.parameters(), lr=1.0e-10, weight_decay=0.0005, momentum=0.99)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    loss_fc = get_loss_function('cross_entropy', size_average=False)

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for images, labels, f_name in trainloader:
                imgs = images
                true_masks = labels

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = loss_fc(input=masks_pred, target=true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, valloader, loss_fc, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    logging.info('Validation cross entropy: {}'.format(val_score))
                    writer.add_scalar('Loss/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(load_pth=args.load,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
