# TODO
#
# rewrite __getitem__
# 多线程load data


from torch.utils.data import *
from torchvision import transforms
from PIL import Image
from imutils import paths
import numpy as np
import random
import cv2
import os
import pandas as pd
from CBLchars import CHARS, CHARS_DICT, LP_CLASS_DICT


class LPRDataLoader(Dataset):
    def __init__(
        self, annoFile, imgSize, lpr_max_len, PreprocFun=None, shuffle: bool = False
    ):
        self.anno_csv = pd.read_csv(annoFile, sep=" ")
        # lp class->class num
        lp_class = self.anno_csv.iloc[:, 2]
        lp_class_num = LP_class_map(lp_class)
        self.anno_csv.iloc[:, 2] = lp_class_num
        # shuffle self.anno_csv
        if shuffle:
            self.anno_csv = self.anno_csv.sample(frac=1).reset_index(drop=True)
        self.img_dir = os.path.dirname(annoFile)

        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = transforms.Compose(
                [
                    transforms.Resize(imgSize[::-1]),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化图像
                ]
            )

    def __len__(self):
        return len(self.anno_csv)

    def __getitem__(self, index):
        filename, LicencePlate, Lp_class = self.anno_csv.iloc[index, :]
        filePath = self.img_dir + "/" + filename
        img_int = Image.open(filePath)
        img_tensor = self.PreprocFun(img_int)
        label = [CHARS_DICT[c] for c in LicencePlate]

        # image = cv2.imread(filename)
        # height, width, _ = img_int.shape
        # if height != self.img_size[1] or width != self.img_size[0]:
        #     img_int = cv2.resize(img_int, self.img_size)
        # img_int = self.PreprocFun(img_int)
        # basename = os.path.basename(filename)
        # imgname, suffix = os.path.splitext(basename)
        # imgname = imgname.split("-")[0].split("_")[0]
        # label = list()
        # for c in LicencePlate:
        #     # one_hot_base = np.zeros(len(CHARS))
        #     # one_hot_base[CHARS_DICT[c]] = 1
        #     label.append(CHARS_DICT[c])
        # if len(label) == 8 and not self.check(label):
        #     print(LicencePlate)
        #     assert 0, "Error label ^~^!!!"

        return (
            img_tensor,
            label,
            len(label),
            Lp_class,
        )  # TODO [warn]returned img used to be np, now is tensor. err may occor.

    # def transform(self, img):
    #     img = img.astype('float32')
    #     img -= 127.5
    #     img *= 0.0078125
    #     img = np.transpose(img, (2, 0, 1))
    #     return img

    def check(self, label):
        if (
            label[2] != CHARS_DICT["D"]
            and label[2] != CHARS_DICT["F"]
            and label[-1] != CHARS_DICT["D"]
            and label[-1] != CHARS_DICT["F"]
        ):
            print("Error label, Please check!")
            return False
        else:
            return True


def dummy_trans():
    import pandas as pd

    data = pd.read_csv("data/CBLPRD-330k_v1/val.txt", sep=" ")
    lp_class = data.iloc[:, 2]
    print(lp_class)
    lp_class = pd.get_dummies(lp_class)  # , dummy_na=True. no nan in lp class
    print(type(lp_class.columns[-1]))
    nan_true_count = lp_class[np.nan].sum()
    print(lp_class)
    return


def LP_class_map(
    lp_class, class_dict=LP_CLASS_DICT
):  # 将 lp_class 的字符串替换为分类序号
    lp_class_num = lp_class.map(class_dict)
    return lp_class_num


def shuffle_pd(pd_csv):
    return pd_csv.sample(frac=1).reset_index(drop=True)

def collate_fn(batch):
    import torch
    imgs = []
    labels = []
    lengths = []
    lp_classes=[]
    for _, sample in enumerate(batch):
        img, label, length ,lp_class= sample
        imgs.append(img)
        labels.extend(label)
        lengths.append(length)
        lp_classes.append(lp_class)
    labels = np.asarray(labels).flatten().astype(int)
    
    return torch.stack(imgs, 0), torch.from_numpy(labels), lengths, lp_classes

def test_module():
    dataset = LPRDataLoader("data/CBLPRD-330k_v1/val.txt", [94, 24], 8)
    one_data = dataset[10]
    batch_iterator = iter(DataLoader(dataset, 10, shuffle=True, num_workers=8, collate_fn=collate_fn))
    images, labels, lengths,lp_cla = next(batch_iterator)
    
    import time
    start_time = time.time()  # 记录开始时间
    for i,batch in enumerate(batch_iterator):
        # print(batch)
        pass
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算消耗的时间
    print(f"Elapsed time: {elapsed_time} seconds")
    return

if __name__ == "__main__":
    
    pass