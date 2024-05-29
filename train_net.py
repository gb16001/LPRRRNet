# -*- coding: utf-8 -*-
''' latest train cmd
    python train_LPRNet.py --train_img_dirs   data/test  --test_img_dirs   data/test --pretrained_model weights/origin_Final_LPRNet_model.pth --train_batch_size 256 --learning_rate 0.001 --test_interval 500 --max_epoch 
    train this
    python train_net.py --train_batch_size 256 --learning_rate 0.001 --test_interval 500 --max_epoch 5

    --pretrained_model weights/CBL-acc.822.pth --learning_rate 0.0001
'''

from data import  CHARS_DICT, LPRDataLoader,CBLDataLoader,CBLdata2iter
from data.CBLchars import CHARS 

from model.LPRNet import build_lprnet
from model import fuckNet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch.utils.tensorboard import SummaryWriter

from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import os
from d2l import torch as d2l
from dynaconf import Dynaconf

def creat_net(args):
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    fucknet=fuckNet(len(CHARS))
    print("Successful to build network!")
    return  lprnet

def weights_init(m):#TODO this func should move to moddle py.
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
def init_net_weight(lprnet:nn.Module,args):#TODO this func should move to moddle py.
    if args.pretrained_model:
        # load pretrained model
        lprnet.load_state_dict(torch.load(args.pretrained_model))
        print("load pretrained model successful!")
    elif False:
        #TODO backbone load_state_dict,container weights_init
        None
    else:
        for idx,m in enumerate(lprnet.modules()):
            m.apply(weights_init)
        # lprnet.backbone.apply(weights_init)
        # if hasattr(lprnet, 'container'):
        #     lprnet.container.apply(weights_init)
        print("initial net weights successful!")
    return 

def creat_optim(lprnet,args):
    # optimizer = optim.SGD(lprnet.parameters(), lr=args.learning_rate,
    # momentum=args.momentum, weight_decay=args.weight_decay)
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
    parser.add_argument('--config_file', default='args.yaml',help='config yaml file')
    args = parser.parse_args()

    return args


def train(conf_file:str):
    # args = get_parser()
    # conf_file:str=get_parser().config_file
    args=Dynaconf(settings_files=[conf_file])
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
    lprnet.to(device)
    # animatior=d2l.Animator(xlabel='epoch', xlim=[0, epochs_num], legend=['train loss', 'train acc', 'test acc'])
    Tboard_writer = SummaryWriter()
    Tboard_writer.add_graph(lprnet, torch.randn(1,3,24,94).cuda())  #模型及模型输入数据

    
    for epoch_num in range(epochs_num):
        # metric = d2l.Accumulator(3)
        lprnet.train()
        if epoch_num%args.epoch_p_save==0 and epoch_num!=0:
            torch.save(lprnet.state_dict(), args.save_folder + 'LPRNet_' + '_epoch_' + repr(epoch_num) + '.pth')
        if epoch_num%args.epoch_p_test==0 and epoch_num!=0:
            val_acc= Greedy_Decode_Eval(lprnet, test_dataset, args)
            # animatior.add(epoch_num,(None,None,val_acc))
            Tboard_writer.add_scalar('train/valAcc', val_acc, epoch_num*epoch_size)
            for name,param in lprnet.named_parameters():
                Tboard_writer.add_histogram(name,param.clone().cpu().data.numpy(),epoch_num)
        for i,(images, labels, lengths, lp_class)in enumerate(train_iter):
            start_time = time.time()
            images=images.to(device)
            labels=labels.to(device)
            # get ctc parameters
            input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
            # update lr
            lr = adjust_learning_rate(optimizer, epoch_num, args.learning_rate, args.lr_schedule)
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
                # animatior.add(epoch_num+i/epoch_size,(loss.item(),None,None))
                Tboard_writer.add_scalar('train/loss', loss.item(), epoch_num*epoch_size+i)


    # final test
    print("Final test Accuracy:")
    val_acc=Greedy_Decode_Eval(lprnet, test_dataset, args)
    # animatior.add(epochs_num,(None,None,val_acc))
    Tboard_writer.add_scalar('train/valAcc', val_acc, epochs_num*epoch_size)

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
    train('args.yaml')
