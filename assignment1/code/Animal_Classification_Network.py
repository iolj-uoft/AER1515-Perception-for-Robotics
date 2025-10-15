import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, n_classes):
        super(CNN, self).__init__()
        # CNN Layers
        ####################################################################
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1)
        ####################################################################

        # Max Pool Layer
        ##################################
        self.maxpool = nn.MaxPool2d(2, 2)
        ##################################

        # TODO: Add batch normalization layer and dropout layers here
        ##############################################################
        # Your Code
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(1024)
        
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.2)
        ##############################################################

        # Fully Connected Layers
        #######################################
        self.fc1 = nn.Linear(1024 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)
        #######################################


    def forward(self, x):
        # TODO: Apply your added layers in this function
        x = F.relu(self.conv1(x))
        # x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        # x = self.bn2(x)
        # x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        # x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        # x = self.bn4(x)
        # x = self.dropout(x)
        x = F.relu(self.conv5(x))
        x = x.view(-1, 1024 * 2 * 2)
        x = self.fc1(x)
        # x = self.bn_fc1(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        # x = self.bn_fc2(x)
        # x = self.dropout(x)
        x = self.fc3(x)
                
        return x