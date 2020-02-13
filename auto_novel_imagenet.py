import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import BCE, PairEnum, cluster_acc, Identity, AverageMeter, seed_torch
from utils import ramps 
from torchvision.models.resnet import BasicBlock
from data.imagenetloader import ImageNetLoader30, ImageNetLoader882_30Mix, ImageNetLoader882
from tqdm import tqdm
import numpy as np
import math
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class ResNet(nn.Module):

    def __init__(self, block, layers, num_labeled_classes=10, num_unlabeled_classes=10):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.head1= nn.Linear(512 * block.expansion, num_labeled_classes)
        self.head2= nn.Linear(512 * block.expansion, num_unlabeled_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out1 = self.head1(x)
        out2 = self.head2(x)

        return out1, out2, x 


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

            mask_lb = idx<train_loader.labeled_length

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

            consistency_loss = (F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar))

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

def copy_param(model, pretrain_dir):
    pre_dict = torch.load(pretrain_dir)
    new=list(pre_dict.items())
    dict_len = len(pre_dict.items())
    model_kvpair=model.state_dict()
    count=0
    for key, value in model_kvpair.items():
        if count < dict_len:
            layer_name,weights=new[count]      
            model_kvpair[key]=weights
            count+=1
        else:
            break
    model.load_state_dict(model_kvpair)
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                            help='device ids assignment (e.g 0 1 2 3)')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--rampup_length', default=50, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=10.0)
    parser.add_argument('--step_size', default=30, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--unlabeled_batch_size', default=128, type=int)
    parser.add_argument('--num_labeled_classes', default=882, type=int)
    parser.add_argument('--num_unlabeled_classes', default=30, type=int)
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/ImageNet/')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--warmup_model_dir', type=str, default='./data/experiments/pretrained/resnet18_imagenet_classif_882_ICLR18.pth')
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--model_name', type=str, default='resnet')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--unlabeled_subset', type=str, default='A')
    parser.add_argument('--mode', type=str, default='train')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)
    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= os.path.join(args.exp_root, runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+'{}_{}.pth'.format(args.model_name, args.unlabeled_subset) 

    model = ResNet(BasicBlock, [2,2,2,2], args.num_labeled_classes, args.num_unlabeled_classes)
    model = nn.DataParallel(model, args.device_ids).to(device)
    model = copy_param(model, args.warmup_model_dir)

    for name, param in model.named_parameters(): 
        if 'head' not in name and 'layer4' not in name:
            param.requires_grad = False

    mix_train_loader = ImageNetLoader882_30Mix(args.batch_size, num_workers=8, path=args.dataset_root, unlabeled_subset=args.unlabeled_subset, aug='twice', shuffle=True, subfolder='train', unlabeled_batch_size=args.unlabeled_batch_size)
    labeled_eval_loader = ImageNetLoader882(args.batch_size, num_workers=8, path=args.dataset_root, aug=None, shuffle=False, subfolder='val')
    unlabeled_eval_loader = ImageNetLoader30(args.batch_size, num_workers=8, path=args.dataset_root, subset=args.unlabeled_subset, aug=None, shuffle=False, subfolder='train')

    if args.mode == 'train':
        train(model, mix_train_loader, labeled_eval_loader, unlabeled_eval_loader, args)
        torch.save(model.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))
    else:
        print("model loaded from {}.".format(args.model_dir))
        model.load_state_dict(torch.load(args.model_dir))

    print('test on labeled classes')
    args.head = 'head1'
    test(model, labeled_eval_loader, args)
    print('test on unlabeled classes')
    args.head = 'head2'
    test(model, unlabeled_eval_loader, args)
