import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from utils.util import BCE, PairEnum, cluster_acc, Identity, AverageMeter, seed_torch
from utils import ramps 
from models.resnet import ResNet, BasicBlock 
from data.cifarloader import CIFAR10Loader, CIFAR10LoaderMix, CIFAR100Loader, CIFAR100LoaderMix
from data.svhnloader import SVHNLoader, SVHNLoaderMix
from tqdm import tqdm
import numpy as np
import os

def train(model, train_loader, labeled_eval_loader, unlabeled_eval_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion1 = nn.CrossEntropyLoss() 
    criterion2 = BCE() 
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        exp_lr_scheduler.step()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length) 
        for batch_idx, ((x, x_bar),  label, idx) in enumerate(tqdm(train_loader)):
            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)
            output1, output2, feat = model(x)
            output1_bar, output2_bar, _ = model(x_bar)
            prob1, prob1_bar, prob2, prob2_bar=F.softmax(output1, dim=1),  F.softmax(output1_bar, dim=1), F.softmax(output2, dim=1), F.softmax(output2_bar, dim=1)

            mask_lb = label<args.num_labeled_classes

            rank_feat = (feat[~mask_lb]).detach()
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2= PairEnum(rank_idx)
            rank_idx1, rank_idx2=rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)

            rank_diff = rank_idx1 - rank_idx2
            rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
            target_ulb = torch.ones_like(rank_diff).float().to(device) 
            target_ulb[rank_diff>0] = -1 

            prob1_ulb, _= PairEnum(prob2[~mask_lb]) 
            _, prob2_ulb = PairEnum(prob2_bar[~mask_lb]) 

            loss_ce = criterion1(output1[mask_lb], label[mask_lb])
            loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
            consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar)
            loss = loss_ce + loss_bce + w * consistency_loss 

            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        print('test on labeled classes')
        args.head = 'head1'
        test(model, labeled_eval_loader, args)
        print('test on unlabeled classes')
        args.head='head2'
        test(model, unlabeled_eval_loader, args)


def train_IL(model, train_loader, labeled_eval_loader, unlabeled_eval_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion1 = nn.CrossEntropyLoss() 
    criterion2 = BCE() 
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        exp_lr_scheduler.step()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)
        for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):
            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)
            output1, output2, feat = model(x)
            output1_bar, output2_bar, _ = model(x_bar)
            prob1, prob1_bar, prob2, prob2_bar = F.softmax(output1, dim=1), F.softmax(output1_bar, dim=1), F.softmax(output2, dim=1), F.softmax(output2_bar, dim=1)
            
            mask_lb = label < args.num_labeled_classes

            rank_feat = (feat[~mask_lb]).detach()

            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = PairEnum(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
            
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)

            rank_diff = rank_idx1 - rank_idx2
            rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
            target_ulb = torch.ones_like(rank_diff).float().to(device)
            target_ulb[rank_diff > 0] = -1

            prob1_ulb, _ = PairEnum(prob2[~mask_lb])
            _, prob2_ulb = PairEnum(prob2_bar[~mask_lb])

            loss_ce = criterion1(output1[mask_lb], label[mask_lb])

            label[~mask_lb] = (output2[~mask_lb]).detach().max(1)[1] + args.num_labeled_classes

            loss_ce_add = w * criterion1(output1[~mask_lb], label[~mask_lb]) / args.rampup_coefficient * args.increment_coefficient
            loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
            consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar)

            loss = loss_ce + loss_bce + loss_ce_add + w * consistency_loss

            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        print('test on labeled classes')
        args.head = 'head1'
        test(model, labeled_eval_loader, args)
        print('test on unlabeled classes')
        args.head='head2'
        test(model, unlabeled_eval_loader, args)

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--rampup_length', default=150, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=50)
    parser.add_argument('--increment_coefficient', type=float, default=0.05)
    parser.add_argument('--step_size', default=170, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)
    parser.add_argument('--num_labeled_classes', default=5, type=int)
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--warmup_model_dir', type=str, default='./data/experiments/pretrain/auto_novel/resnet_rotnet_cifar10.pth')
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--IL', action='store_true', default=False, help='w/ incremental learning')
    parser.add_argument('--model_name', type=str, default='resnet')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, svhn')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)
    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= os.path.join(args.exp_root, runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+'{}.pth'.format(args.model_name) 

    model = ResNet(BasicBlock, [2,2,2,2], args.num_labeled_classes, args.num_unlabeled_classes).to(device)

    num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    if args.mode=='train':
        state_dict = torch.load(args.warmup_model_dir)
        model.load_state_dict(state_dict, strict=False)
        for name, param in model.named_parameters(): 
            if 'head' not in name and 'layer4' not in name:
                param.requires_grad = False
 
    if args.dataset_name == 'cifar10':
        mix_train_loader = CIFAR10LoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, labeled_list=range(args.num_labeled_classes), unlabeled_list=range(args.num_labeled_classes, num_classes))
        labeled_train_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, target_list = range(args.num_labeled_classes))
        unlabeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        unlabeled_eval_loader_test = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        labeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))
        all_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(num_classes))
    elif args.dataset_name == 'cifar100':
        mix_train_loader = CIFAR100LoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, labeled_list=range(args.num_labeled_classes), unlabeled_list=range(args.num_labeled_classes, num_classes))
        labeled_train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, target_list = range(args.num_labeled_classes))
        unlabeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        unlabeled_eval_loader_test = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        labeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))
        all_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(num_classes))
    elif args.dataset_name == 'svhn':
        mix_train_loader = SVHNLoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, labeled_list=range(args.num_labeled_classes), unlabeled_list=range(args.num_labeled_classes, num_classes))
        labeled_train_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, target_list = range(args.num_labeled_classes))
        unlabeled_eval_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        unlabeled_eval_loader_test = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        labeled_eval_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))
        all_eval_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(num_classes))
     
    if args.mode == 'train':
        if args.IL:
            save_weight = model.head1.weight.data.clone()
            save_bias = model.head1.bias.data.clone()
            model.head1 = nn.Linear(512, num_classes).to(device)
            model.head1.weight.data[:args.num_labeled_classes] = save_weight
            model.head1.bias.data[:] = torch.min(save_bias) - 1.
            model.head1.bias.data[:args.num_labeled_classes] = save_bias
            train_IL(model, mix_train_loader, labeled_eval_loader, unlabeled_eval_loader, args)
        else:
            train(model, mix_train_loader, labeled_eval_loader, unlabeled_eval_loader, args)
        torch.save(model.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))
    else:
        print("model loaded from {}.".format(args.model_dir))
        if args.IL:
            model.head1 = nn.Linear(512, num_classes).to(device)
        model.load_state_dict(torch.load(args.model_dir))

    print('Evaluating on Head1')
    args.head = 'head1'
    print('test on labeled classes (test split)')
    test(model, labeled_eval_loader, args)
    if args.IL:
        print('test on unlabeled classes (test split)')
        test(model, unlabeled_eval_loader_test, args)
        print('test on all classes (test split)')
        test(model, all_eval_loader, args)
    print('Evaluating on Head2')
    args.head = 'head2'
    print('test on unlabeled classes (train split)')
    test(model, unlabeled_eval_loader, args)
    print('test on unlabeled classes (test split)')
    test(model, unlabeled_eval_loader_test, args)