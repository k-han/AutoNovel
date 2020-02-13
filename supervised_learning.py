import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import cluster_acc, Identity, AverageMeter
from models.resnet import ResNet, BasicBlock 
from data.cifarloader import CIFAR10Loader, CIFAR100Loader
from data.svhnloader import SVHNLoader
from tqdm import tqdm
import numpy as np
import os

def train(model, train_loader, labeled_eval_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion1 = nn.CrossEntropyLoss() 
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        exp_lr_scheduler.step()
        for batch_idx, (x, label, idx) in enumerate(tqdm(train_loader)):
            x, label = x.to(device), label.to(device)
            output1, _, _ = model(x)
            loss= criterion1(output1, label)
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        print('test on labeled classes')
        args.head = 'head1'
        test(model, labeled_eval_loader, args)

def test(model, test_loader, args):
    model.eval() 
    preds=np.array([])
    targets=np.array([])
    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        output1, output2, _ = model(x)
        if args.head=='head1':
            output = output1
        else:
            output = output2
        _, pred = output.max(1)
        targets=np.append(targets, label.cpu().numpy())
        preds=np.append(preds, pred.cpu().numpy())
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds) 
    print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    return preds 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--step_size', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)
    parser.add_argument('--num_labeled_classes', default=5, type=int)
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--rotnet_dir', type=str, default='./data/experiments/selfsupervised_learning/rotnet_cifar10.pth')
    parser.add_argument('--model_name', type=str, default='resnet_rotnet')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, svhn')
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= os.path.join(args.exp_root, runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+'{}.pth'.format(args.model_name) 

    model = ResNet(BasicBlock, [2,2,2,2], args.num_labeled_classes, args.num_unlabeled_classes).to(device)

    num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    state_dict = torch.load(args.rotnet_dir)
    del state_dict['linear.weight']
    del state_dict['linear.bias']
    model.load_state_dict(state_dict, strict=False)
    for name, param in model.named_parameters(): 
        if 'head' not in name and 'layer4' not in name:
            param.requires_grad = False
 
    if args.dataset_name == 'cifar10':
        labeled_train_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, target_list = range(args.num_labeled_classes))
        labeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))
    elif args.dataset_name == 'cifar100':
        labeled_train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, target_list = range(args.num_labeled_classes))
        labeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))
    elif args.dataset_name == 'svhn':
        labeled_train_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, target_list = range(args.num_labeled_classes))
        labeled_eval_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))

    if args.mode == 'train':
        train(model, labeled_train_loader, labeled_eval_loader, args)
        torch.save(model.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))
    elif args.mode == 'test':
        print("model loaded from {}.".format(args.model_dir))
        model.load_state_dict(torch.load(args.model_dir))
    print('test on labeled classes')
    args.head = 'head1'
    test(model, labeled_eval_loader, args)
