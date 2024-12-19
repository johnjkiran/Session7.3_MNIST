import torch
import torch.nn as nn
import torch.nn.functional as F
# dropout_value = 0.1

class Model_13(nn.Module):
    def __init__(self):
        super(Model_13, self).__init__()
        # Input Block - Expand
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=0)  # 28x28 -> 26x26 
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, padding=0)  # 26x26 -> 24x24
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # 24x24 -> 12x12

        # Efficient feature extraction
        self.conv3 = nn.Conv2d(16, 8, 1)  # Channel reduction
        self.bn3 = nn.BatchNorm2d(8)
        
        # Focused feature extraction
        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 12x12 -> 12x12
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1)  # 12x12 -> 12x12
        self.bn5 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)  # 12x12 -> 6x6

        # Final feature refinement
        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1)  # 6x6 -> 6x6
        self.bn6 = nn.BatchNorm2d(16)
        
        # Output block
        self.conv7 = nn.Conv2d(16, 10, 1)  # Point-wise to reduce to class count
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout1 = nn.Dropout2d(0.05)  # Reduced dropout
        self.dropout2 = nn.Dropout2d(0.025)  # Lighter dropout later

    def forward(self, x):
        x = self.dropout1(self.bn1(F.relu(self.conv1(x))))
        x = self.dropout1(self.bn2(F.relu(self.conv2(x))))
        x = self.pool1(x)
        
        x = self.dropout1(self.bn3(F.relu(self.conv3(x))))
        
        x = self.dropout2(self.bn4(F.relu(self.conv4(x))))
        x = self.dropout2(self.bn5(F.relu(self.conv5(x))))
        x = self.pool2(x)
        
        x = self.dropout2(self.bn6(F.relu(self.conv6(x))))
        x = self.conv7(x)
        x = self.avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)