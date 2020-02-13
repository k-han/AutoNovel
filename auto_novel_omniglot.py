import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import BCE, PairEnum, cluster_acc, Identity, AverageMeter, seed_torch, str2bool
from utils import ramps 
from data.omniglotloader import OmniglotLoaderMix, alphabetLoader, omniglot_evaluation_alphabets_mapping 
from tqdm import tqdm
import numpy as np
import os

class VGG(nn.Module):

    def __init__(self, num_labeled_classes=5, num_unlabeled_classes=5):
        super(VGG, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64, 128, kernel_size=3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                        )
        self.layer3 = nn.Sequential(
                        nn.Conv2d(128, 256, kernel_size=3, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                        )
        self.layer4 = nn.Sequential(
                        nn.Conv2d(256, 256, kernel_size=3, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                        )
        self.head1 = nn.Sequential(
                        nn.Linear(1024, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(inplace=True),
                        nn.Linear(512, num_labeled_classes)
                        )
        self.head2 = nn.Sequential(
                        nn.Linear(1024, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(inplace=True),
                        nn.Linear(512, num_unlabeled_classes)
                        )
            
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        out1 = self.head1(x)
        out2 = self.head2(x)
        return out1, out2, x 

def train(model, train_loader, unlabeled_eval_loader, args):
    optimizer = Adam(model.parameters(), lr=args.lr)
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

            consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar)

            loss = loss_ce + loss_bce + w * consistency_loss 

            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        print('test on unlabeled classes')
        args.head='head2'
        test(model, unlabeled_eval_loader, args)


def test(model, test_loader, args):
    model.eval()
    acc_record = AverageMeter()
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
    return acc, nmi, ari 

def copy_param(model, pre_dict):
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
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--rampup_length', default=1, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=100.0)
    parser.add_argument('--step_size', default=100, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--num_labeled_classes', default=964, type=int)
    parser.add_argument('--num_unlabeled_classes', default=20, type=int)
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--dataset_root', type=str, default='./data/datasets')
    parser.add_argument('--exp_root', type=str, default='./data/experiments')
    parser.add_argument('--warmup_model_dir', type=str, default='./data/experiments/pretrained/vgg6_omniglot_proto.pth')
    parser.add_argument('--model_name', type=str, default='vgg6')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', type=str, default='train')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    seed_torch(args.seed)
    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= os.path.join(args.exp_root, runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if args.mode == 'train':
        state_dict = torch.load(args.warmup_model_dir) 

    acc = {}
    nmi = {}
    ari = {}

    for _, alphabetStr in omniglot_evaluation_alphabets_mapping.items():
 
        mix_train_loader= OmniglotLoaderMix(alphabet=alphabetStr, batch_size=args.batch_size, aug='twice', shuffle=True, num_workers=2, root=args.dataset_root, unlabeled_batch_size=32)
        unlabeled_eval_loader= alphabetLoader(root=args.dataset_root, batch_size=args.batch_size, alphabet=alphabetStr, subfolder_name='images_evaluation', aug=None, num_workers=2, shuffle=False)
        args.num_unlabeled_classes = unlabeled_eval_loader.num_classes
        args.model_dir = model_dir+'/'+'{}_{}.pth'.format(args.model_name, alphabetStr) 

        model = VGG(num_labeled_classes=args.num_labeled_classes, num_unlabeled_classes=args.num_unlabeled_classes).to(device)
 
        if args.mode == 'train':
            model = copy_param(model, state_dict)
            for name, param in model.named_parameters(): 
                if 'head' not in name and 'layer4' not in name:
                    param.requires_grad = False
            train(model, mix_train_loader, unlabeled_eval_loader, args)
            torch.save(model.state_dict(), args.model_dir)
            print("model saved to {}.".format(args.model_dir))
        elif args.mode == 'test':
            print("model loaded from {}.".format(args.model_dir))
            model.load_state_dict(torch.load(args.model_dir))
        print('test on unlabeled classes')
        args.head = 'head2'
        acc[alphabetStr], nmi[alphabetStr], ari[alphabetStr] = test(model, unlabeled_eval_loader, args)
    print('ACC for all alphabets:',acc)
    print('NMI for all alphabets:',nmi)
    print('ARI for all alphabets:',ari)
    avg_acc, avg_nmi, avg_ari = sum(acc.values())/float(len(acc)), sum(nmi.values())/float(len(nmi)), sum(ari.values())/float(len(ari))
    print('AVG: acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(avg_acc, avg_nmi, avg_ari))

