# -*- coding: utf-8 -*-

from data import  CHARS_DICT, LPRDataLoader,CBLDataLoader,CBLdata2iter
from data.CBLchars import CHARS ,LP_CLASS

from model.LPRNet import build_lprnet
from model import LPRRRNet,T_LENGTH,init_net_weight,myNet,O_fix
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
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    mynet=O_fix(len(CHARS))
    print("Successful to build network!")
    return  mynet


def creat_optim(lprnet,args):
    # optimizer = optim.SGD(lprnet.parameters(), lr=args.learning_rate,
    # momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = (
        optim.RMSprop(
            lprnet.parameters(),
            lr=args.learning_rate,
            alpha=0.9,
            eps=1e-08,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        if args.optim == "RMS"
        else optim.Adam(
            lprnet.parameters(),
            lr=args.learning_rate,
            betas=(0.8, 0.9),
            weight_decay=args.weight_decay,
        )
    )

    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'
    CE_loss=nn.CrossEntropyLoss() if args.lpr_class_predict else None
    return optimizer,ctc_loss,CE_loss

def creat_dataset(args):
    train_dataset = CBLDataLoader(args.CBLtrain, args.img_size, args.lpr_max_len)
    test_dataset = CBLDataLoader(args.CBLval, args.img_size, args.lpr_max_len)
    epoch_size = len(train_dataset) // args.train_batch_size
    return train_dataset,test_dataset,epoch_size

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
    train_dataset,test_dataset,epoch_size=creat_dataset(args)
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
    optimizer,ctc_loss,CE_loss=creat_optim(lprnet,args)

    # ready to train
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    # animatior=d2l.Animator(xlabel='epoch', xlim=[0, epochs_num], legend=['train loss', 'train acc', 'test acc'])
    Tboard_writer = SummaryWriter(args.tb_log_dir)
    Tboard_writer.add_graph(lprnet, torch.randn(1,3,24,94).cuda())  #模型及模型输入数据
    # Continue training for additional epochs
    init_epochs = args.init_epoch
    add_epochs = args.add_epochs
    end_epochs=init_epochs+add_epochs
    global_step = init_epochs * epoch_size
    
    for epoch_num in range(init_epochs,end_epochs):
        # metric = d2l.Accumulator(3)
        lprnet.train()
        if epoch_num%args.epoch_p_save==0 and epoch_num!=0:
            torch.save(lprnet.state_dict(), args.save_folder + 'LPRNet_' + '_epoch_' + repr(epoch_num) + '.pth')
        if epoch_num%args.epoch_p_test==0 and epoch_num!=0:
            val_acc= Greedy_Decode_Eval(lprnet, test_iter, args)
            # animatior.add(epoch_num,(None,None,val_acc))
            Tboard_writer.add_scalar('train/valAcc', val_acc, epoch_num*epoch_size)
            for name,param in lprnet.named_parameters():
                Tboard_writer.add_histogram(name,param.clone().cpu().data.numpy(),epoch_num)
        for i,(images, labels, lengths, lp_classes)in enumerate(train_iter):
            start_time = time.time()
            images=images.to(device)
            labels=labels.to(device)
            lp_classes=torch.tensor(lp_classes,device=device)
            # get ctc parameters
            input_lengths, target_lengths = sparse_tuple_for_ctc(T_LENGTH, lengths)
            # update lr
            lr = adjust_learning_rate(optimizer, epoch_num, args.learning_rate, args.lr_schedule)
            # forward
            if args.lpr_class_predict:
                logits,lp_class_hat=lprnet(images)
            else:
                logits = lprnet(images)
            log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
            # print(labels.shape)
            log_probs = log_probs.log_softmax(2).requires_grad_()
            # backprop
            optimizer.zero_grad()
            loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)+0.2*CE_loss(lp_class_hat, lp_classes) if args.lpr_class_predict else ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
            if loss.item() == np.inf:
                continue
            loss.backward()
            optimizer.step()
            end_time = time.time()
            if i % 20 == 0:
                print(f'Epoch: {epoch_num}/{end_epochs} || batch: {i}/{epoch_size} || Loss: {loss.item():.4f} || Batch time: {end_time - start_time:.4f} s || LR: {lr:.8f}')
                # animatior.add(epoch_num+i/epoch_size,(loss.item(),None,None))
                Tboard_writer.add_scalar('train/loss', loss.item(), epoch_num*epoch_size+i)
    Tboard_writer.close()


    # final test
    print("Final test Accuracy:")
    val_acc=Greedy_Decode_Eval(lprnet, test_iter, args)
    # animatior.add(epochs_num,(None,None,val_acc))
    Tboard_writer.add_scalar('train/valAcc', val_acc, end_epochs*epoch_size)

    # save final parameters
    torch.save(lprnet.state_dict(), args.save_folder + 'Final_LPRNet_model.pth')

def unpack_lables(labels,lengths):
    start = 0
    targets = []
    for length in lengths:
        label = labels[start:start+length]
        targets.append(label)
        start += length
    targets = np.array([el.numpy() for el in targets],dtype=object)
    return targets

def greedy_decode(prebs,):
    
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
    return preb_labels

import numpy as np
def check_lables(preb_labels,targets,lp_class,*Tn):
    Tp,Tn_1,Tn_2=Tn
    for i, label in enumerate(preb_labels):
        if len(label) != len(targets[i]):
            Tn_1 += 1
            # print(f"{lp_class[i]}")
            continue
        if (np.asarray(targets[i]) == np.asarray(label)).all():
            Tp += 1
        else:
            Tn_2 += 1
            # print(f"{label}|{targets[i]}")
    return Tp, Tn_1, Tn_2


def Greedy_Decode_Eval(Net, testIter, args):
    Net.eval()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    Tp,Tn_1,Tn_2=0,0,0
    t1 = time.time()
    for i,(images, labels, lengths, lp_class) in enumerate(testIter):
        images=images.to(device)
        targets =unpack_lables(labels,lengths)
        # forward
        if args.lpr_class_predict:
            prebs,lpClass_hat = Net(images)
        else:
            prebs = Net(images)
        preb_labels=greedy_decode(prebs)
        Tp,Tn_1,Tn_2=check_lables(preb_labels,targets,lp_class,Tp,Tn_1,Tn_2)

    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print(f"[Info] Test Accuracy: {Acc} [{Tp}:{Tn_1}:{Tn_2}:{(Tp+Tn_1+Tn_2)}]")
    t2 = time.time()
    print(f"[Info] Test Speed: {(t2 - t1)}s /{len(testIter)}]")
    Net.train()
    return Acc


if __name__ == "__main__":
    train('args.yaml')
    # evaluate('args.yaml')
