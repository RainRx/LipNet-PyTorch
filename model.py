import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(4, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.ReLU(True),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 8, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Linear(48, 48),
            nn.LSTM(bidirectional=True, num_layers=2, hidden_size=200, input_size=48)
        )

    def forward(self, x):
        
        x = self.conv(x)
        return x
        
    
