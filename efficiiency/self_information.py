import numpy as np
import torch.nn as nn
from torchinfo import summary

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

__test_net = SimpleModel()
__test_inputSize = (3, 32, 32)

def Cost(model, input_size, switch:str="params"):
    """Get the number of parameters and FLOPs of the model."""
    model_summary = summary(model, input_size=(1, *input_size), verbose=0)
    # # Print model summary
    # summary(model, input_size=(1, *input_size))
    num_params = model_summary.total_params
    flops = model_summary.total_mult_adds * 2  # Each MAC operation corresponds to 2 FLOPs
    return num_params if switch=="params" else flops

def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))

def DP_1(Delta_E, C_n1, C_n2):
    """Calculate DP(n1 - n2 | n2) using the first definition."""
    return Delta_E / np.exp((C_n1 - C_n2) / C_n2 - 1)

def DP_2(Delta_E, C_n1, C_n2):
    """Calculate DP(n1 - n2 | n2) using the second definition with sigmoid."""
    return Delta_E / np.exp(64 * sigmoid((C_n1 - C_n2)/64.) - 32)


def selfInfoEfficiency(p_err:float,net,inputSize):
    assert p_err < 1, "/{p_err/} must <1.0"
    I_err = -np.log(p_err)
    E_net = I_err  # p_err_0=1
    C_net = Cost(net, inputSize)
    P_net = E_net / C_net
    # print(f"{I_err},{E_net},{C_net},{P_net}")
    return P_net, E_net, C_net, I_err
def Delta_Efficiency(p_Net1,p_Net0,Net1,Net0,inSize):
    _, E0, C0, _ = selfInfoEfficiency(p_Net0, Net0, inSize)
    _, E1, C1, _ = selfInfoEfficiency(p_Net1, Net1, inSize)
    d_E=E1-E0
    DP_value_1=DP_1(d_E,C1,C0)
    DP_value_2=DP_2(d_E,C1,C0)
    # print("DP_value_1:", DP_value_1)
    # print("DP_value_2:", DP_value_2)
    return DP_value_1

def __test():
    


    return
if __name__=="__main__":
    # selfInfoEfficiency(0.1,__test_net,__test_inputSize)
    Delta_Efficiency(0.01,0.1,__test_net,__test_net,__test_inputSize)
    # test()
    pass
