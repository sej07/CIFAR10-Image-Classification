import torch
import torch.nn as nn
from src import config

#architecture
'''
3 convolutional blocks (Conv -> BatchNorm -> ReLU -> MaxPool)
2 fully connected layers
Dropout for regularization
'''
import torch
import torch.nn as nn
from src import config


class CNN(nn.Module): 
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolutional blocks
        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels, 32x32 -> 16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 32 -> 64 channels, 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 64 -> 128 channels, 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(512, config.NUM_CLASSES)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x