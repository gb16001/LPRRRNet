# TODO CBL chars is diff with old dataset
import torch
from torch.utils.data import *
from torchvision import transforms
from PIL import Image
from imutils import paths
import numpy as np
import random
import cv2
import os
import pandas as pd
from .CBLchars import CHARS, CHARS_DICT, LP_CLASS_DICT


class CBLDataLoader(Dataset):
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

        return (
            img_tensor,
            label,
            len(label),
            Lp_class,
        )

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
    lp_classes = []
    for _, sample in enumerate(batch):
        img, label, length, lp_class = sample
        imgs.append(img)
        labels.extend(label)
        lengths.append(length)
        lp_classes.append(lp_class)
    labels = np.asarray(labels).flatten().astype(int)

    return torch.stack(imgs, 0), torch.from_numpy(labels), lengths, lp_classes


def CBLdata2iter(
    dataset: CBLDataLoader,
    batch_size=10,
    shuffle=True,
    num_workers=8,
    collate_fn=collate_fn,
):
    return DataLoader(
        dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=collate_fn
    )

def transform_labels(labels, lengths, padding_value=73, max_length=8):
    """
    Transforms the CTC-style labels into a 2D array for CRE loss.
    
    Args:
        labels (torch.Tensor): The flattened tensor of all labels.
        lengths (list): List of lengths for each sequence.
        padding_value (int): The value to pad shorter sequences with.
        max_length (int): The uniform length of each sequence.
    
    Returns:
        torch.Tensor: A 2D tensor of shape (max_length, B) with padded sequences.
    """
    # Convert labels tensor to numpy array
    labels_np = labels.numpy()
    
    # Number of sequences
    B = len(lengths)
    
    # Initialize the 2D label array with the padding value
    label_array = np.full((max_length,B ), padding_value, dtype=int)
    
    # Pointer for the start index in the flattened label list
    start_idx = 0
    
    for i, length in enumerate(lengths):
        # Calculate the end index for the current sequence
        end_idx = start_idx + length
        
        # Copy the original labels to the 2D array
        label_array[ :length,i] = labels_np[start_idx:end_idx]
        
        # Update the start index for the next sequence
        start_idx = end_idx
    
    # Convert the 2D label array back to tensor
    label_tensor = torch.tensor(label_array)
    
    return label_tensor

def prepare_rnn_input_output(x, start_token=74, end_token=74):
    """
    Prepares the input (X) and output (Y) for RNN training.
    
    Args:
        x (torch.Tensor): The tensor of shape (N, BZ) containing the transformed labels.
        start_token (int): The token to prepend to the input sequences.
        end_token (int): The token to append to the output sequences.
    
    Returns:
        torch.Tensor: The input tensor X of shape (N + 1, BZ).
        torch.Tensor: The output tensor Y of shape (N + 1, BZ).
    """
    N, BZ = x.shape
    
    # Create X by adding start_token at the beginning of each sequence
    start_tokens = torch.full((1, BZ), start_token, dtype=int)
    X = torch.cat((start_tokens, x), dim=0)
    
    # Create Y by adding end_token at the end of each sequence
    end_tokens = torch.full((1, BZ), end_token, dtype=int)
    Y = torch.cat((x, end_tokens), dim=0)
    
    return X, Y

def test_module():
    dataset = CBLDataLoader("data/CBLPRD-330k_v1/val.txt", [94, 24], 8)
    one_data = dataset[10]
    # dataset:list=>iter
    batch_iterator = CBLdata2iter(dataset)
    images, labels, lengths, lp_cla = next(batch_iterator)
    import time

    start_time = time.time()  # 记录开始时间
    for i, batch in enumerate(batch_iterator):
        # print(batch)
        pass
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算消耗的时间
    print(f"Elapsed time: {elapsed_time} seconds")
    return


if __name__ == "__main__":
    test_module()
    pass
