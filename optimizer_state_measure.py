import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class CustomNet(nn.Module):
    def __init__(self, input_size, layer_widths, output_size):
        super(CustomNet, self).__init__()
        self.layers = nn.ModuleList()
        
        current_size = input_size
        for width in layer_widths:
            self.layers.append(nn.Linear(current_size, width))
            current_size = width
        # Add the output layer
        self.layers.append(nn.Linear(current_size, output_size))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        return x

# Specify the architecture
input_size = 28 * 28
output_size = 10
layer_widths = [128, 64, 32]  # Specify the widths for each layer

# Create the neural network
net = CustomNet(input_size, layer_widths, output_size)

# Function to calculate the state dict size
def calculate_state_dict_size(optimizer):
    total_size = 0
    for param_state in optimizer.state.values():
        for tensor in param_state.values():
            # Calculate the size in bytes of each tensor
            total_size += tensor.numel() * tensor.element_size()
    return total_size

# Initialize optimizers
optimizer_sgd = optim.SGD(net.parameters(), lr=0.01)
optimizer_adam = optim.Adam(net.parameters(), lr=0.01)

# Calculate the storage overhead for each optimizer
sgd_state_size = calculate_state_dict_size(optimizer_sgd)
adam_state_size = calculate_state_dict_size(optimizer_adam)

print(f"SGD state dict size: {sgd_state_size} bytes")
print(f"Adam state dict size: {adam_state_size} bytes")

# Calculate the ratio of the Adam overhead compared to SGD
if sgd_state_size > 0:
    ratio = adam_state_size / sgd_state_size
else:
    ratio = "infinite"  # If SGD state size is zero, ratio is undefined
print(f"Adam overhead compared to SGD: {ratio}")
