# -*- coding: utf-8 -*-
''' latest train cmd
    python train_LPRNet.py --train_img_dirs   data/test  --test_img_dirs   data/test --pretrained_model weights/origin_Final_LPRNet_model.pth --train_batch_size 256 --learning_rate 0.001 --test_interval 500 --max_epoch 
    train this
    python train_net.py --train_batch_size 256 --learning_rate 0.001 --test_interval 500 --max_epoch 5
'''

from data import  CHARS_DICT, LPRDataLoader,CBLDataLoader,CBLdata2iter
from data.CBLchars import CHARS 

from model.LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import os
from d2l import torch as d2l

def creat_net(args):
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successful to build network!")
    return lprnet
def weights_init(m):#TODO this func should move to moddle py
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)
def init_net_weight(lprnet,args):
    if args.pretrained_model:
        # load pretrained model
        lprnet.load_state_dict(torch.load(args.pretrained_model))
        print("load pretrained model successful!")
    elif False:
        #TODO backbone load_state_dict,container weights_init
        None
    else:
        lprnet.backbone.apply(weights_init)
        lprnet.container.apply(weights_init)
        print("initial net weights successful!")
    return 

def creat_optim(lprnet,args):
    # optimizer = optim.SGD(lprnet.parameters(), lr=args.learning_rate,
    #                       momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.RMSprop(
        lprnet.parameters(),
        lr=args.learning_rate,
        alpha=0.9,
        eps=1e-08,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'
    return optimizer,ctc_loss

def creat_dataset(args):
    train_img_dirs = os.path.expanduser(args.train_img_dirs)
    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    train_dataset = CBLDataLoader(args.CBLtrain, args.img_size, args.lpr_max_len)
    test_dataset = CBLDataLoader(args.CBLval, args.img_size, args.lpr_max_len)
    epoch_size = len(train_dataset) // args.train_batch_size
    max_iter = args.max_epoch * epoch_size
    return train_dataset,test_dataset,epoch_size,max_iter

def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)

def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    """
    Sets the learning rate
    """
    lr = 0
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    if lr == 0:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=5, type=int, help='epoch to train the network')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--train_img_dirs', default="~/workspace/trainMixLPR", help='the train images path')
    parser.add_argument('--test_img_dirs', default="~/workspace/testMixLPR", help='the test images path')
    parser.add_argument('--is_CBL',default=True,type=bool, help="dataset is CBL dataset")
    parser.add_argument('--CBLtrain',default="data/CBLPRD-330k_v1/train.txt", help="CBL train's anno file")
    parser.add_argument('--CBLval',default="data/CBLPRD-330k_v1/val.txt", help="CBL val's anno file")
    parser.add_argument('--dropout_rate', default=0.5, type=float,help='dropout rate.')
    parser.add_argument('--learning_rate', default=0.0001, type=float,help='base value of learning rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--train_batch_size', default=256, type=int,help='training batch size.')
    parser.add_argument('--test_batch_size', default=120, type=int,help='testing batch size.')
    parser.add_argument('--phase_train', default=True, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--save_interval', default=2000, type=int, help='interval for save model state dict')
    parser.add_argument('--test_interval', default=2000, type=int, help='interval for evaluate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    parser.add_argument('--lr_schedule', default=[4, 8, 12, 14, 16], help='schedule for learning rate.')
    parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
    # parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--pretrained_model', default='', help='pretrained base model')
    parser.add_argument('--epoch_p_save', default=3)
    parser.add_argument('--epoch_p_test', default=1)

    args = parser.parse_args()

    return args


def train():
    args = get_parser()
    # get dataset
    train_dataset,test_dataset,epoch_size,max_iter=creat_dataset(args)
    train_iter=CBLdata2iter( train_dataset,
                args.train_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
    test_iter=CBLdata2iter(
        test_dataset,
        args.test_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    # init net
    lprnet=creat_net(args)
    init_net_weight(lprnet,args)
    # define optimizer, loss
    optimizer,ctc_loss=creat_optim(lprnet,args)

    # ready to train
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    T_length = 18 # args.lpr_max_len
    epochs_num = args.max_epoch
    device = torch.device("cuda:0" if args.cuda else "cpu")
    animatior=d2l.Animator(xlabel='epoch', xlim=[1, epochs_num], legend=['train loss', 'train acc', 'test acc'])
    
    for epoch_num in range(epochs_num):
        metric = d2l.Accumulator(3)
        lprnet.train()
        if epoch_num%args.epoch_p_save==0 and epoch_num!=0:
            torch.save(lprnet.state_dict(), args.save_folder + 'LPRNet_' + '_epoch_' + repr(epoch_num) + '.pth')
        if epoch_num%args.epoch_p_test==0 and epoch_num!=0:
            val_acc= Greedy_Decode_Eval(lprnet, test_dataset, args)
            animatior.add(epoch_num,(None,None,val_acc))
        for i,(images, labels, lengths, lp_class)in enumerate(train_iter):
            start_time = time.time()
            # get ctc parameters
            input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
            # update lr
            lr = adjust_learning_rate(optimizer, epoch_num, args.learning_rate, args.lr_schedule)
            if args.cuda:
                # images.to(device)
                # labels.to(device)
                images=images.cuda()
                labels=labels.cuda()
                pass
            # forward
            logits = lprnet(images)
            log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
            # print(labels.shape)
            log_probs = log_probs.log_softmax(2).requires_grad_()
            # backprop
            optimizer.zero_grad()
            loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
            if loss.item() == np.inf:
                continue
            loss.backward()
            optimizer.step()
            end_time = time.time()
            if i % 20 == 0:
                print(f'Epoch: {epoch_num}/{epochs_num} || batch: {i}/{epoch_size} || Loss: {loss.item():.4f} || Batch time: {end_time - start_time:.4f} s || LR: {lr:.8f}')
                animatior.add(epoch_num+i/epoch_size,(loss.cpu().detach().numpy(),None,None))


    # final test
    print("Final test Accuracy:")
    val_acc=Greedy_Decode_Eval(lprnet, test_dataset, args)
    animatior.add(epochs_num,(None,None,val_acc))

    # save final parameters
    torch.save(lprnet.state_dict(), args.save_folder + 'Final_LPRNet_model.pth')

def Greedy_Decode_Eval(Net, datasets, args):
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(CBLdata2iter(
        datasets,
        args.test_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )) 

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    for i in range(epoch_size):
        # load train data
        images, labels, lengths, lp_class = next(batch_iterator)
        
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets],dtype=object)

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        prebs = Net(images)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1

    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print(f"[Info] Test Accuracy: {Acc} [{Tp}:{Tn_1}:{Tn_2}:{(Tp+Tn_1+Tn_2)}]")
    t2 = time.time()
    print(f"[Info] Test Speed: {(t2 - t1)}s /{len(datasets)}]")
    return Acc


if __name__ == "__main__":
    train()
