import random
import time
import warnings
import sys
import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
# import matplotlib.pyplot as plt

sys.path.append('.')
from adaptation.pac_bayes import jointdisagreement
from adaptation.pseudo_labeling import PseudoLabeling
from adaptation,uncertainty import VariationRatio, PredictiveEntropy, MutualInfo
from modules.kernels import JointMultipleKernelMaximumMeanDiscrepancy, GaussianKernel
from modules.classifier import MCdropClassifier
import vision.datasets as datasets
import vision.models as models
from tools.utils import *
from tools.transforms import ResizeImage
from tools.lr_scheduler import StepwiseLR

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Domain Adaptation')
    # data args
    parser.add_argument('--root', default='/home/pku1616/liny/PacBayesDan/data', help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', help='dataset: ' + ' | '.join(dataset_names) +' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 2)')

    # training args
    parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-i', '--iters_per_epoch', default=100, type=int, help='Number of iterations per epoch')
    parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning_rate',  default=0.003, type=float,metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=0.0005, type=float, metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--gamma', default=0.0003, type=float)
    parser.add_argument('--decat_rate', default=0.75, type=float)

    # model args
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=architecture_names)
    parser.add_argument('--bottleneck_dim', default=256, type=int)
    parser.add_argument('--classifier_width', default=256, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--dropout_type', default='Bernoulli', type=str)
    parser.add_argument('--freeze_backbone', default=False, type=bool)

    # loss args
    parser.add_argument('--lambda1', default=1., type=float, help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--lambda2', default=1., type=float, help='the trade-off hyper-parameter for joint loss')
    parser.add_argument('--linear', default=False, action='store_true',  help='whether use the linear version')
    parser.add_argument('--loss_sample_num', default=3, type=int, help='number of models sampled to calcualate loss')

    # pseudo args
    parser.add_argument('--start_epoch', default=5, type=int)
    parser.add_argument('--prob_ema', default=False, type=bool)
    parser.add_argument('--prob_ema_gamma', default=0.1, type=float, help='EMA of pseudo label')
    parser.add_argument('--prob_type', default='epoch_update', type=str, choices=['prediction', 'prediction_avg', 'source_prototype', 'target_prototype'])
    parser.add_argument('--weights_type', default='uncertainty', type=str, choices=['uncertainty', 'entropy', 'threshold', 'max_value', 'time_consistency'])
    parser.add_argument('--threshold', default=0.75, type=float)
    parser.add_argument('--tc_ema_gamma', default=0.9, type=float, help='EMA of time consistency')
    parser.add_argument('--uncertainty_type', default='predictive_entropy', type='str', choices=['predictive_entropy', 'mutual_info', 'variation_ratio'])
    parser.add_argument('--uncertainty_sample_num', default=100, type=int, help='number of models sampled to calcualate uncertainty')

    # other args
    parser.add_argument('-p', '--print_freq', default=20, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default='2', type=str)

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    setup_seed(args.seed)
    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        ResizeImage(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    val_transform = transforms.Compose([
        ResizeImage(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dataset = datasets.__dict__[args.data]
    args.root = os.path.join(args.root, args.data)
    train_source_dataset = dataset(root=args.root, task=args.source, download=False, transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_dataset = dataset(root=args.root, task=args.target, download=False, transform=train_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset = dataset(root=args.root, task=args.target, download=False, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if args.data == 'DomainNet':
        test_dataset = dataset(root=args.root, task=args.target, evaluate=True, download=False, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        test_loader = val_loader

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    #pseudo labels
    pseudo_labels = PseudoLabeling(len(train_target_dataset), train_target_dataset.num_classes,
                                   args.prob_ema_gamma, args.tc_ema_gamma, args.threshold, device)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True)
    num_classes = train_source_dataset.num_classes
    classifier = MCdropClassifier(backbone, num_classes).to(device)

    # define loss function
    jmmd_loss = JointMultipleKernelMaximumMeanDiscrepancy(
        kernels=(
            [GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
            (GaussianKernel(sigma=0.92, track_running_stats=False),)
        ),
        linear=args.linear, thetas=None
    ).to(device)

    # define optimizer
    parameters = classifier.get_parameters()
    optimizer = SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_sheduler = StepwiseLR(optimizer, init_lr=args.lr, gamma=args.gamma, decay_rate=args.decay_rate)

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        pseudo_labels.copy_history()

        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, jmmd_loss, optimizer,
              lr_sheduler, pseudo_labels, epoch, args)

        acc1 = validate(val_loader, classifier, args)

        # remember best acc@1 and save checkpoint
        if acc1 > best_acc1:
            best_model = copy.deepcopy(classifier.state_dict())
        best_acc1 = max(acc1, best_acc1)

        # calculate source prototype and update pseudo labels according to prototype_s
        prototype_s = calculate_source_prototype(train_source_loader, classifier)
        prototype_s = prototype_s.to(device)
        EPOCH_PROTOTYPE_S_update(prototype_s, val_loader, classifier, pseudo_labels, args, epoch)
        #if epoch>=args.start_epoch-1:
        #    EPOCH_PROTOTYPE_S_update(prototype_s, val_loader, classifier, pseudo_labels, args, epoch)

        # calculate target prototype and update pseudo labels according to prototype_t
        #EPOCH_PROTOTYPE_T_update(val_loader, classifier, pseudo_labels, args, epoch)
        #if epoch>=args.start_epoch-1:
        #    EPOCH_PROTOTYPE_T_update(val_loader, classifier, pseudo_labels, args, epoch)

        #PROTOTYPE_MAXST_update(train_source_loader, val_loader, classifier, pseudo_labels, args, epoch)

        #PROTOTYPE_MATRIX_update(val_loader, classifier, pseudo_labels, args, epoch)

        # EPOCH_update_pseudo_label(val_loader, classifier, pseudo_labels, args, epoch)
        #if epoch>=args.start_epoch-1:
        #    EPOCH_update_pseudo_label(val_loader, classifier, pseudo_labels, args, epoch)
        #if epoch>=args.start_epoch:
        #    pseudo_labels.time_consistency_weight()

    print("best_acc1 = {:3.3f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(best_model)
    acc1 = validate(test_loader, classifier, args)
    print("test_acc1 = {:3.3f}".format(acc1))

def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: nn.Module,
          jmmd_loss: JointMultipleKernelMaximumMeanDiscrepancy, optimizer: SGD, lr_sheduler: StepwiseLR,
          pseudo_labels: PseudoLabeling, epoch: int, args: argparse.Namespace):
    losses = AverageMeter('Loss', ':3.2f')
    cls_losses = AverageMeter('Cls Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':5.4f')
    joint_losses = AverageMeter('Joint Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    tgt_accs = AverageMeter('Tgt Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [losses, cls_losses, trans_losses, joint_losses, cls_accs, tgt_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    jmmd_loss.train()

    for i in range(args.iters_per_epoch):
        model.train()
        jmmd_loss.train()

        lr_sheduler.step()

        x_s, labels_s, index_s = next(train_source_iter)
        x_t, labels_t, index_t = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        h = model.backbone_forward(x)
        y, f = model.head_forward(h)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = jmmd_loss(
            (f_s, F.softmax(y_s, dim=1)),
            (f_t, F.softmax(y_t, dim=1))
        )

        loss = cls_loss + transfer_loss * args.lambda1

        if epoch >= args.start_epoch:
            with torch.no_grad():
                pseudo_labels_t = pseudo_labels.get_hard_pseudo_label(index_t)
                pseudo_labels.entropy_weight()
                # pseudo_labels.threshold_weight()
                #pseudo_labels.difference_to_one_weight()
                weights_t = pseudo_labels.get_weight(index_t)
            weights_t = F.softmax(weights_t, dim=0)
            
            ys_mc = []
            for j in range(args.loss_sample_num):
                y_mc, _ = model.head_forward(h)
                ys_mc.append(y_mc)

            joint_loss = 0.0
            for j in range(args.loss_sample_num):
                for k in range(j+1, args.loss_sample_num):
                    y1_s, y1_t = ys_mc[j].chunk(2, dim=0)
                    y2_s, y2_t = ys_mc[k].chunk(2, dim=0)
                    joint_loss += jointdisagreement(y1_s, y1_t, y2_s, y2_t, labels_s, pseudo_labels_t, weights_t)

            joint_loss = joint_loss* 2.0 / ((args.loss_sample_num-1.0)*args.loss_sample_num)
            loss += joint_loss * args.lambda2


        cls_acc = accuracy(y_s, labels_s)[0]
        tgt_acc = accuracy(y_t, labels_t)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        tgt_accs.update(tgt_acc.item(), x_t.size(0))
        cls_losses.update(cls_loss.item(), x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))

        try:
            joint_losses.update(joint_loss.item(), x_s.size(0))
        except:
            joint_losses.update(0.0, x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if epoch>=args.start_epoch-1:
        #    ITERATION_update_pseudo_label(x_t, index_t, model, pseudo_labels, args, epoch)

        if i % args.print_freq == 0:
            progress.display(i)

def validate(val_loader: DataLoader, model: nn.Module, args: argparse.Namespace) -> float:
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [losses, top1, top5],
        prefix='Test: ')

    model.eval()

    with torch.no_grad():
        for i, (images, target, index) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg

def psudo_label_update_and_weight_calculate(train_source_loader, val_loader, model, pseudo_labels, args, epoch):
    if args.prob_type == 'prediction':
        EPOCH_update_pseudo_label(val_loader, model, pseudo_labels, args, epoch)
    elif args.prob_type == 'source_prototype':
        EPOCH_PROTOTYPE_S_update(train_source_loader, val_loader, classifier, pseudo_labels, args, epoch)
    elif args.prob_type == 'target_prototype':
        EPOCH_PROTOTYPE_T_update(val_loader, model, pseudo_labels, args, epoch)
    elif args.prob_type == 'prediction_avg':
        assert args.weights_type == 'uncertainty'
        pass
    else:
        raise ValueError(f'pesudo label unpdate type not found')

                    # pseudo_labels.threshold_weight()
                #pseudo_labels.difference_to_one_weight()
                weights_t = pseudo_labels.get_weight(index_t)

    if args.weights_type == 'uncertainty':

    elif args.weights_type == 'entropy':
        pseudo_labels.entropy_weight()
    elif args.weights_type =='threshold':
        pseudo_labels.threshold_weight()
    elif args.weights_type == 'max_value':
        pseudo_labels.difference_to_one_weight()
    elif args.weights_type == 'time_consistency':
        pseudo_labels.time_consistency_weight():
    else:
        raise ValueError(f'pesudo label weight unpdate type not found')

def uncertainty_update(target_loader: DataLoader, model: nn.Module, pseudo_labels: PseudoLabeling, args, epoch):
    model.eval()
    model.activate_dropout()

    with torch.no_grad():
        for i, (images, target, index) in enumerate(loader):
            images = images.to(device)
            ys = []
            for j in range(args.uncertainty_sample_num):
                y, _ = model(images)
                y = F.softmax(y, dim=1)
                ys.append(y)
            ys = torch.cat(ys, 0)
            y = torch.mean(ys, 0)

            if args.prob_type == 'prediction_avg'
                if args.prob_ema==True:
                    pseudo_labels.EMA_update_p(y, index, epoch)
                else:
                    pseudo_labels.update_p(y, index)
            
            if args.uncertainty_type == 'predictive_entropy':
                uncertainty = PredictiveEntropy(ys)
            elif args.uncertainty_type == 'mutual_info':
                uncertainty = MutualInfo(ys)
            elif args.uncertainty_type == 'variation_ratio':
                uncertainty = VariationRatio(ys, device)
            else:
                raise ValueError(f'uncertainty type not found')
            
            pseudo_labels.update_weight(-1.0*uncertainty, index)
            

def EPOCH_update_pseudo_label(loader: DataLoader, model: nn.Module, pseudo_labels: PseudoLabeling, args, epoch):
    model.eval()
    with torch.no_grad():
        for i, (images, target, index) in enumerate(loader):
            images = images.to(device)
            target = target.to(device)
            y, _ = model(images)
            y = F.softmax(y, dim=1)
            if args.prob_ema==True:
                pseudo_labels.EMA_update_p(y, index, epoch)
            else:
                pseudo_labels.update_p(y, index)

def ITERATION_update_pseudo_label(x_t: torch.Tensor, index_t: torch.Tensor, model: nn.Module, pseudo_labels: PseudoLabeling, args, epoch):
    model.eval()
    with torch.no_grad():
        y_t, _ = model(x_t)
        y = F.softmax(y_t, dim=1)
        if args.prob_ema == True:
            pseudo_labels.EMA_update_p(y, index_t, epoch)
        else:
            pseudo_labels.update_p(y, index_t)

def EPOCH_PROTOTYPE_S_update(source_loader:DataLoader, target_loader:DataLoader, model:nn.Module, pseudo_labels:PseudoLabeling, args, epoch):
    model.eval()
    num_classes = model.num_classes
    features_dim = model.features_dim
    count = [0]*num_classes
    prototype_s = torch.zeros((num_classes, features_dim), device=device)

    with torch.no_grad():
        for i, (images, target, index) in enumerate(source_loader):
            images = images.to(device)
            target = target.to(device)
            y, f = model(images)

            for j in range(images.shape[0]):
                count[int(target[j].item())] += 1
                prototype_s[int(target[j].item())] += f[j]

        for i in range(prototype_s.shape[0]):
            prototype_s[i] = prototype_s[i]/count[i]

        for i, (images, target, index) in enumerate(target_loader):
            images = images.to(device)
            target = target.to(device)
            y, f = model(images)

            prediction = ((f.unsqueeze(1) - prototype_s.unsqueeze(0)) ** 2).sum(2).pow(0.5)
            prediction = F.softmax(prediction, 1)

            if args.prob_ema==True:
                pseudo_labels.EMA_update_p(prediction, index, epoch)
            else:
                pseudo_labels.update_p(prediction, index)

def EPOCH_PROTOTYPE_T_update(target_loader: DataLoader, model: nn.Module, pseudo_labels: PseudoLabeling, args, epoch):
    model.eval()
    num_classes = model.num_classes
    features_dim = model.features_dim
    count = [0] * num_classes
    prototype_t = torch.zeros((num_classes, features_dim), device=device)

    with torch.no_grad():
        for i, (images, target, index) in enumerate(target_loader):
            images = images.to(device)
            target = target.to(device)
            y, f = model(images)
            confidence, index = torch.max(y, dim=1)

            #for j in range(y.shape[0]):
            #    for k in range(y.shape[1]):
            #        prototype_t[k] += y[j,k].item() * f[j]
            #        count[k] += y[j,k].item()

            for j in range(images.shape[0]):
                count[int(index[j].item())] += 1
                prototype_t[int(index[j].item())] += f[j]

        for i in range(prototype_t.shape[0]):
            prototype_t[i] = prototype_t[i]/count[i]

        for i, (images, target, index) in enumerate(target_loader):
            images = images.to(device)
            target = target.to(device)
            y, f = model(images)

            prediction = ((f.unsqueeze(1) - prototype_s.unsqueeze(0)) ** 2).sum(2).pow(0.5)
            prediction = F.softmax(prediction, 1)

            if args.prob_ema==True:
                pseudo_labels.EMA_update_p(prediction, index, epoch)
            else:
                pseudo_labels.update_p(prediction, index)

def PROTOTYPE_MAXST_update(source_loader: DataLoader, val_loader: DataLoader, model: nn.Module,
                           pseudo_labels: PseudoLabeling, args, epoch):
    model.eval()
    num_classes = model.num_classes
    features_dim = model.features_dim
    count_s = [0] * num_classes
    count_t = [0] * num_classes
    prototype_s = torch.zeros((num_classes, features_dim), device=device)
    prototype_t = torch.zeros((num_classes, features_dim), device=device)

    with torch.no_grad():
        for i, (images, target, index) in enumerate(source_loader):
            images = images.to(device)
            target = target.to(device)
            y1, f1, y2, f2 = model(images)
            f = 0.5*f1 + 0.5*f2   #f:(batch_size, features_dim)

            for j in range(images.shape[0]):
                count_s[int(target[j].item())] += 1
                prototype_s[int(target[j].item())] += f[j]

        for i in range(prototype_s.shape[0]):
            prototype_s[i] = prototype_s[i]/count_s[i]

        for i, (images, target, index) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            y1, f1, y2, f2 = model(images)
            f = 0.5 * f1 + 0.5 * f2  # f:(batch_size, features_dim)
            y = 0.5 * y1 + 0.5 * y2  # y:(batch_size, num_classes)
            confidence, index = torch.max(y, dim=1)

            for j in range(images.shape[0]):
                count_t[int(index[j].item())] += 1
                prototype_t[int(index[j].item())] += f[j]

        for i in range(prototype_t.shape[0]):
            prototype_t[i] = prototype_t[i]/count_t[i]

        for i, (images, target, index) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            y1, f1, y2, f2 = model(images)
            f = 0.5*f1 + 0.5*f2
            prediction_s = torch.zeros_like(y1)
            prediction_t = torch.zeros_like(y1)
            for j in range(prediction_s.shape[0]):
                for k in range(prediction_s.shape[1]):
                    prediction_s[j,k] = torch.exp(-torch.norm(f[j]-prototype_s[k], 2))
                prediction_s[j] = prediction_s[j]/torch.sum(prediction_s[j])

            for j in range(prediction_t.shape[0]):
                for k in range(prediction_t.shape[1]):
                    prediction_t[j,k] = torch.exp(-torch.norm(f[j]-prototype_t[k], 2))
                prediction_t[j] = prediction_t[j]/torch.sum(prediction_t[j])

            prediction = torch.zeros_like(prediction_s)
            for j in range(prediction.shape[0]):
                conf_s = torch.max(prediction_s[j]).item()
                conf_t = torch.max(prediction_t[j]).item()
                if conf_s>conf_t:
                    prediction[j] = prediction_s[j]
                else:
                    prediction[j] = prediction_t[j]

            if args.prob_ema==True:
                pseudo_labels.EMA_update_p(prediction, index, epoch)
            else:
                pseudo_labels.update_p(prediction, index)

def PROTOTYPE_MATRIX_update(loader: DataLoader, model: nn.Module, pseudo_labels: PseudoLabeling, args, epoch):
    model.eval()
    num_classes = model.num_classes
    features_dim = model.features_dim
    count = [0] * num_classes
    with torch.no_grad():
        weight1 = model.get_head_weight1()
        weight2 = model.get_head_weight2()
        matrix = 0.5*weight1 + 0.5*weight2   #(num_classes, features_dim)
        for i, (images, target, index) in enumerate(loader):
            images = images.to(device)
            target = target.to(device)
            y1, f1, y2, f2 = model(images)
            f = 0.5*f1 + 0.5*f2     #(batch_size, features_dim)
            prediction = torch.zeros_like(y1)    #(batch_size, num_classes)
            for j in range(prediction.shape[0]):
                for k in range(prediction.shape[1]):
                    a = f[j]/torch.norm(f[j], 2)
                    b = matrix[k,:]/torch.norm(matrix[k,:], 2)
                    dist = 1-torch.sum(a*b)
                    prediction[j,k] = torch.exp(-20.0*dist)
                prediction[j] = prediction[j]/torch.sum(prediction[j])

            if args.prob_ema==True:
                pseudo_labels.EMA_update_p(prediction, index, epoch)
            else:
                pseudo_labels.update_p(prediction, index)


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    args = get_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(args)
