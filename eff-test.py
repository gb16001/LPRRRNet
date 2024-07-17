import torch
import torch.nn as nn
from torchinfo import summary

import torch.nn.functional as F

def Cost(model, inputData):
    """Get the number of parameters and FLOPs of the model."""
    model_summary = summary(model, input_data=inputData, verbose=0)
    # # Print model summary
    # summary(model, input_size=(1, *input_size))
    num_params = model_summary.total_params
    flops = model_summary.total_mult_adds * 2  # Each MAC operation corresponds to 2 FLOPs
    return num_params, flops

# Example usage with a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

import model
classNum_char=74
batchSize=1
maxLength=8
net=model.MS_CTC(75)
# net=LPRRRNet(char_classNum=74)
img=torch.randn(batchSize,3,24,94)
# input_size = (3,24,94)
labels=torch.randint(0, 75, (8,))
x=F.one_hot(labels.long(),75).to(torch.float32)
H_0=torch.randn(2,75)
# num_params, flops = Cost(net, img)
model_summary = summary(net,input_data=img, verbose=0)
# model_summary = summary(net.head,input_data=(labels,H_0), verbose=0)#  input_data=img

    # # Print model summary
    # summary(model, input_size=(1, *input_size))
num_params = model_summary.total_params
flops = model_summary.total_mult_adds * 2  # Each MAC operation corresponds to 2 FLOPs
#     return num_params, flops
print("Number of parameters:", num_params)
print("FLOPs:", flops)



