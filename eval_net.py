from data import CHARS_DICT, LPRDataLoader, CBLDataLoader, CBLdata2iter
from data.CBLchars import CHARS, LP_CLASS
import numpy as np
import torch
import time
from dynaconf import Dynaconf
from train_net import creat_dataset, creat_net, init_net_weight
import pandas as pd

def evaluate(conf_file: str):
    args = Dynaconf(settings_files=[conf_file])
    train_dataset, test_dataset, epoch_size = creat_dataset(args)

    # statistic dataset
    series = test_dataset.anno_csv.iloc[:, 2]
    lp_clc_freq = series.value_counts()
    print(lp_clc_freq)

    test_iter = CBLdata2iter(
        test_dataset,
        args.test_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    device = torch.device("cuda:0" if args.cuda else "cpu")
    net = creat_net(args).to(device)
    assert args.pretrained_model != ""
    init_net_weight(net, args)
    evalor = eval_Net(lp_clc_freq)
    evalor.Greedy_Decode_Eval(net, test_iter, args)
    # err analyze
    evalor.print_result()
    return


class eval_Net:

    def __init__(self, lp_clc_freq) -> None:

        self.Tn1_lp_clc = []
        self.Tn2_lp_clc = []
        self.lp_clc_freq = lp_clc_freq
        return

    def unpack_lables(self, labels, lengths):
        start = 0
        targets = []
        for length in lengths:
            label = labels[start : start + length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets], dtype=object)
        return targets

    def greedy_decode(self, prebs):
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
            for c in preb_label:  # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        return preb_labels

    def check_lables(self, preb_labels, targets, lp_class, *Tn):
        Tp, Tn_1, Tn_2 = Tn
        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                # print(f"{lp_class[i]}")
                self.Tn1_lp_clc.append(lp_class[i])
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1
                # print(f"{label}|{targets[i]}")
                self.Tn2_lp_clc.append(lp_class[i])
        return Tp, Tn_1, Tn_2

    def Greedy_Decode_Eval(self, Net, testIter, args):
        Net.eval()
        self.Tn1_lp_clc, self.Tn2_lp_clc = [], []

        device = torch.device("cuda:0" if args.cuda else "cpu")
        Tp, Tn_1, Tn_2 = 0, 0, 0
        t1 = time.time()
        for i, (images, labels, lengths, lp_class) in enumerate(testIter):
            images = images.to(device)
            targets = self.unpack_lables(labels, lengths)
            # forward
            prebs = Net(images)
            preb_labels = self.greedy_decode(prebs)
            Tp, Tn_1, Tn_2 = self.check_lables(
                preb_labels, targets, lp_class, Tp, Tn_1, Tn_2
            )

    def print_result(self):
        Tn1_clc_np, Tn2_clc_np = np.array(self.Tn1_lp_clc), np.array(self.Tn2_lp_clc)
        values, counts = np.unique(Tn1_clc_np, return_counts=True)
        err1_freq = dict(zip(LP_CLASS, counts / self.lp_clc_freq[values]))
        # print(f'len err:{err1_freq}')
        values, counts = np.unique(Tn2_clc_np, return_counts=True)
        err2_freq = dict(zip(LP_CLASS, counts / self.lp_clc_freq[values]))
        # print(f'char err:{err2_freq}')
        df = pd.DataFrame(
            {
                "车牌类型": LP_CLASS,
                "len err": [err1_freq.get(cls, np.nan) for cls in LP_CLASS],
                "char err": [err2_freq.get(cls, np.nan) for cls in LP_CLASS],
            }
        )
        print(df)
        return

    pass


if __name__ == "__main__":

    evaluate("args.yaml")
