from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import  transforms
import pickle
import os
import os.path
import datetime
import numpy as np
from data.rotationloader import DataLoader, GenericDataset
from utils.util import AverageMeter, accuracy
from models.resnet import BasicBlock
from tqdm import tqdm
import shutil

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        if is_adapters:
            self.parallel_conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if is_adapters:
            out = F.relu(self.bn1(self.conv1(x)+self.parallel_conv1(x)))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def train(epoch, model, device, dataloader, optimizer, exp_lr_scheduler, criterion, args):
    loss_record = AverageMeter()
    acc_record = AverageMeter()
    exp_lr_scheduler.step()
    model.train()
    for batch_idx, (data, label) in enumerate(tqdm(dataloader(epoch))):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
     
        # measure accuracy and record loss
        acc = accuracy(output, label)
        acc_record.update(acc[0].item(), data.size(0))
        loss_record.update(loss.item(), data.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train Epoch: {} Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(epoch, loss_record.avg, acc_record.avg))

    return loss_record

def test(model, device, dataloader, args):
    acc_record = AverageMeter()
    model.eval()
    for batch_idx, (data, label) in enumerate(tqdm(dataloader())):
        data, label = data.to(device), label.to(device)
        output = model(data)
     
        # measure accuracy and record loss
        acc = accuracy(output, label)
        acc_record.update(acc[0].item(), data.size(0))

    print('Test Acc: {:.4f}'.format(acc_record.avg))
    return acc_record 

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Rot_resNet')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                                    help='disables CUDA training')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=1,
                                    help='random seed (default: 1)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, svhn')
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--model_name', type=str, default='rotnet')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= os.path.join(args.exp_root, runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+'{}.pth'.format(args.model_name) 

    dataset_train = GenericDataset(
        dataset_name=args.dataset_name,
        split='train',
        dataset_root=args.dataset_root
       )
    dataset_test = GenericDataset(
        dataset_name=args.dataset_name,
        split='test',
        dataset_root=args.dataset_root
        )

    dloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True)

    dloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False)

    global is_adapters
    is_adapters = 0
    model = ResNet(BasicBlock, [2,2,2,2], num_classes=4)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)

    criterion = nn.CrossEntropyLoss()

    best_acc = 0 
    for epoch in range(args.epochs +1):
        loss_record = train(epoch, model, device, dloader_train, optimizer, exp_lr_scheduler, criterion, args)
        acc_record = test(model, device, dloader_test, args)
        
        is_best = acc_record.avg > best_acc 
        best_acc = max(acc_record.avg, best_acc)
        if is_best:
            torch.save(model.state_dict(), args.model_dir)

if __name__ == '__main__':
    main()
