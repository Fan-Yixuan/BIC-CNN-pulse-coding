import argparse
import logging
import os
import random

import numpy as np
import torch
import torchvision
from torch import nn
from tqdm import tqdm

from config import cfg
from model.ann import Encoder, ResNet
from model.snn import SpikingResNet

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--config', help="Path to config file", type=str, default='')
parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

if args.config != '':
    cfg.merge_from_file(args.config)
cfg.merge_from_list(args.opts)

random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
torch.cuda.manual_seed_all(cfg.SEED)

logger = logging.getLogger()
logger.setLevel('DEBUG')
BASIC_FORMAT = "%(asctime)s: %(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
if not os.path.isdir(cfg.DIR.RESULT):
    os.makedirs(cfg.DIR.RESULT)
save_path = os.path.join(cfg.DIR.RESULT, cfg.NAME + '.log')
fhlr = logging.FileHandler(save_path, 'w+')
fhlr.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fhlr)
logger.info('Config\n%s', cfg)

train_dataset = torchvision.datasets.MNIST(root=cfg.DIR.DATA, train=True, download=True, transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAIN.BS, shuffle=True, num_workers=0, drop_last=True)
test_set = torchvision.datasets.MNIST(root=cfg.DIR.DATA, train=False, download=True, transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.TEST.BS, shuffle=False, num_workers=0)

ann = ResNet([2, 2, 2, 2], classes=10, channel=cfg.MODEL.WINDOW).cuda()
snn = SpikingResNet([2, 2, 2, 2], classes=10, channel=1).cuda()
encoder = Encoder(cfg.MODEL.WINDOW).cuda()
criterion = nn.CrossEntropyLoss()  # nn.MSELoss()
optimizer_ann = torch.optim.Adam(ann.parameters(), lr=cfg.TRAIN.LR)
optimizer_snn = torch.optim.Adam(snn.parameters(), lr=cfg.TRAIN.LR)
optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=cfg.TRAIN.LR)
scheduler_ann = torch.optim.lr_scheduler.MultiStepLR(optimizer_ann, milestones=[10, 30, 50], gamma=0.1)
scheduler_snn = torch.optim.lr_scheduler.MultiStepLR(optimizer_snn, milestones=[10, 30, 50], gamma=0.1)
scheduler_encoder = torch.optim.lr_scheduler.MultiStepLR(optimizer_encoder, milestones=[10, 30, 50], gamma=0.1)

logger.info('ANN start training.')
for epoch in range(cfg.TRAIN.EPOCHS):
    running_loss = 0.
    correct = 0.
    total = 0.
    ann.train()
    encoder.train()
    for cnt, (images, labels) in enumerate(tqdm(train_loader)):
        ann.zero_grad()
        optimizer_ann.zero_grad()
        optimizer_encoder.zero_grad()

        if cfg.MODE == 'conv':
            dvs = encoder(images.cuda())
        else:
            BS, CH, W, H = images.shape
            dvs = torch.zeros(BS, cfg.MODEL.WINDOW, CH, W, H).cuda()
            for cnt_window in range(cfg.MODEL.WINDOW):
                dvs[:, cnt_window, :, :, :] = (images.float().cuda() > torch.rand(images.size()).cuda()).float()

        outputs = ann(dvs)
        loss = criterion(outputs, labels.cuda())
        running_loss += loss.item()
        loss.backward()
        optimizer_ann.step()
        optimizer_encoder.step()

        _, predicted = outputs.cpu().max(1)
        total += float(labels.size(0))
        correct += float(predicted.eq(labels).sum().item())

    scheduler_ann.step()
    scheduler_encoder.step()
    logger.info('Epoch %d/%d  Loss: %.5f  Train acc: %.3f', epoch + 1, cfg.TRAIN.EPOCHS, running_loss / len(train_loader),
                100 * correct / total)
    for param in encoder.parameters():
        logger.info(param)

    running_loss = 0.
    correct = 0.
    total = 0.
    ann.eval()
    encoder.eval()
    with torch.no_grad():
        for cnt, (images, labels) in enumerate(test_loader):
            if cfg.MODE == 'conv':
                dvs = encoder(images.cuda())
            else:
                BS, CH, W, H = images.shape
                dvs = torch.zeros(BS, cfg.MODEL.WINDOW, CH, W, H).cuda()
                for cnt_window in range(cfg.MODEL.WINDOW):
                    dvs[:, cnt_window, :, :, :] = (images.float().cuda() > torch.rand(images.size()).cuda()).float()

            outputs = ann(dvs)
            loss = criterion(outputs, labels.cuda())
            running_loss += loss.item()

            _, predicted = outputs.cpu().max(1)
            total += float(labels.size(0))
            correct += float(predicted.eq(labels).sum().item())

    logger.info('Test Acc: %.2f', 100 * correct / total)

torch.cuda.empty_cache()
logger.info('SNN start training.')
for epoch in range(cfg.TRAIN.EPOCHS):
    running_loss = 0.
    correct = 0.
    total = 0.
    snn.train()
    encoder.eval()
    for cnt, (images, labels) in enumerate(tqdm(train_loader)):
        snn.zero_grad()
        optimizer_snn.zero_grad()

        if cfg.MODE == 'conv':
            dvs = encoder(images.cuda())
            dvs = dvs.unsqueeze(dim=2)
            dvs = (dvs > torch.zeros(dvs.size()).cuda()).float()
        else:
            BS, CH, W, H = images.shape
            dvs = torch.zeros(BS, cfg.MODEL.WINDOW, CH, W, H).cuda()
            for cnt_window in range(cfg.MODEL.WINDOW):
                dvs[:, cnt_window, :, :, :] = (images.float().cuda() > torch.rand(images.size()).cuda()).float()

        outputs = snn(dvs)
        loss = criterion(outputs, labels.cuda())
        running_loss += loss.item()
        loss.backward()
        optimizer_snn.step()

        _, predicted = outputs.cpu().max(1)
        total += float(labels.size(0))
        correct += float(predicted.eq(labels).sum().item())

    scheduler_snn.step()
    logger.info('Epoch %d/%d  Loss: %.5f  Train acc: %.3f', epoch + 1, cfg.TRAIN.EPOCHS, running_loss / len(train_loader),
                100 * correct / total)

    running_loss = 0.
    correct = 0.
    total = 0.
    snn.eval()
    encoder.eval()
    with torch.no_grad():
        for cnt, (images, labels) in enumerate(test_loader):
            if cfg.MODE == 'conv':
                dvs = encoder(images.cuda())
                dvs = dvs.unsqueeze(dim=2)
                dvs = (dvs > torch.zeros(dvs.size()).cuda()).float()
            else:
                BS, CH, W, H = images.shape
                dvs = torch.zeros(BS, cfg.MODEL.WINDOW, CH, W, H).cuda()
                for cnt_window in range(cfg.MODEL.WINDOW):
                    dvs[:, cnt_window, :, :, :] = (images.float().cuda() > torch.rand(images.size()).cuda()).float()

            outputs = snn(dvs)
            loss = criterion(outputs, labels.cuda())
            running_loss += loss.item()

            _, predicted = outputs.cpu().max(1)
            total += float(labels.size(0))
            correct += float(predicted.eq(labels).sum().item())

    logger.info('Test Acc: %.2f', 100 * correct / total)
