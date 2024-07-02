import torch
import torch.nn as nn
from torchinfo import summary

def Cost(model, input_size):
    """Get the number of parameters and FLOPs of the model."""
    model_summary = summary(model, input_size=(1, *input_size), verbose=0)
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

model = SimpleModel()
input_size = (3, 32, 32)
num_params, flops = Cost(model, input_size)
print("Number of parameters:", num_params)
print("FLOPs:", flops)



