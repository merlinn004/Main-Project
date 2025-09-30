import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, AvgPool2d, Linear, Dropout, AdaptiveAvgPool2d, Dropout2d, BatchNorm1d
from spikingjelly.clock_driven import neuron, functional, surrogate
import torch.nn.functional as F


class EmotionSNN(nn.Module):
    def __init__(self, num_classes=8, 
                 input_height=64, 
                 input_width=400, 
                 conv1_channels=32, 
                 conv2_channels=64, 
                 fc1_units=128,
                 surrogate_func='Sigmoid', 
                 detach_reset=True, 
                 dropout_rate=0.3,
                 l2_reg=0.001):
        
        super().__init__()
        
        # Choose surrogate function
        if surrogate_func == 'Sigmoid':
            surrogate_function = surrogate.Sigmoid(4.0)
        elif surrogate_func == 'ATan':
            surrogate_function = surrogate.ATan(2.0)
        else:
            surrogate_function = surrogate.Sigmoid()
        
        # Convolutional layers
        self.conv1 = Conv2d(1, conv1_channels, kernel_size=3, stride=1, padding=1)
        self.bn1  = BatchNorm2d(conv1_channels)
        self.sn1 = neuron.LIFNode(surrogate_function=surrogate_function, v_threshold=0.5, detach_reset=detach_reset,tau=2.0)
        
        self.conv2 = Conv2d(conv1_channels, conv2_channels, kernel_size=3, stride=1, padding=1)
        self.bn2  = BatchNorm2d(conv2_channels)
        self.sn2 = neuron.LIFNode(surrogate_function=surrogate_function, v_threshold=0.5, detach_reset=detach_reset,tau=2.0)

        self.pool = AdaptiveAvgPool2d((16, 100))
        self.dropout = Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Calculate flattened dimension after pooling
        flattened_size = conv2_channels * 16 * 100
        
        # Fully connected layers
        self.fc1 = Linear(flattened_size, fc1_units)
        self.bn3 = BatchNorm1d(fc1_units)
        self.sn3 = neuron.LIFNode(surrogate_function=surrogate_function, v_threshold=0.5, detach_reset=detach_reset,tau=2.0)
        self.fc1_drop = Dropout(dropout_rate * 1.5) if dropout_rate > 0 else nn.Identity()

        # Output layer - spiking for rate coding
        self.fc2 = Linear(fc1_units, num_classes)
        self.l2_reg = l2_reg

    def forward(self, x, T=None):
        
        # x shape: [B, T, C, H, W]
        if T is None:
            T = x.size(1)
        batch_size = x.size(0)
        
        out_spikes_counter = torch.zeros(batch_size, self.fc2.out_features, device=x.device)
        
        for t in range(T):
            # Get input at time step t: [B, C, H, W]
            xt = x[:, t, :, :, :]  # [batch, 1, n_mels, time_frames]
            
            # Forward pass through network
            out = self.conv1(xt)
            out = self.bn1(out)
            out = self.sn1(out)

           
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.sn2(out)
        

            out = self.pool(out)
            out = self.dropout(out)

            # Flatten for FC layers
            out = out.view(batch_size, -1)
            
            out = self.fc1(out)
            out = self.bn3(out)
            out = self.sn3(out)
            out = self.fc1_drop(out)
        
            out = self.fc2(out)
            

            # Accumulate spikes
            out_spikes_counter += out
        
        # Reset all neurons after forward pass
        functional.reset_net(self)
        
        # Return average spike rate (rate coding)
        return out_spikes_counter / T
    
    def get_l2_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.l2_reg * l2_loss