'''
define model used in cnn.py, and cnn_test.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)         # out_channels from 6 to 8
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)        # in_channels from 6 to 8
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 84)            # added
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))    # torch.Size([4, 6, 14, 14])
        # x = self.pool(F.relu(self.conv2(x)))    # torch.Size([4, 16, 5, 5])
        # x = x.view(-1, 16 * 5 * 5)              # torch.Size([4, 400])
        # x = F.relu(self.fc1(x))                 # torch.Size([4, 120])
        # x = F.relu(self.fc2(x))                 # torch.Size([4, 84])
        # x = self.fc3(x)                         # torch.Size([4, 10])
        
        '''
        x -> conv1/relu -> pool -> con2/relu -> pool -> flatten (view()) -> fc1/relu -> fc2/relu -> fc3 -> y
        x: torch.Size([4, 3, 32, 32])= (N, C, H, W). N = 4 images, C = 3 channels, H = 32 height, W = 32 width
         | (4, 3, 32, 32)
        Conv2d(in_channels=3, out_channels=6, kernel_size=5). 3: C of x, 6: specified by user, 5: (3 channels x 5x5) specified by user
        relu()
         | (4, 6, 28, 28): 32 pixel - 5 kernel_size + 1 = 28 pixels
        MaxPool2d(kernel_size=2, stride=2)
         | (4, 6, 14, 14)
        Conv2d(in_channels=6, out_channels=16, kernel_size=5): : 6 from relu; 16 specified by user, 5 specified by user
        relu()
         | (4, 16, 10, 10): 14 pixels - 5 kernel_size + 1 = 10 pixels
        MaxPool2d(kernel_size=2, stride=2)
         | (4, 16, 5, 5)
        view(-1, 16*5*5): flatten output to be fed to DNN
         | (4, 16*5*5)
        Linear(in_features=16*5*5, out_features=120): feed 4 images, each with 16*5*5 features to 120 neurons
        relu()
         | (4, 120): 4 images, each with 120 outputs
        Linear(in_features=120, out_features=84): 120 neurons to 84 neurons
        relu()
         | (4, 84)
        Linear(in_features=84, out_features=10)
         | (4, 10): 4 images, each with 10 output (10 different image classes)
        '''
        x = self.conv1(x)                       # torch.Size([4, 6, 28, 28]), h_out = h_in - kernel + 1 = 32 - 5 + 1 = 28 for height and width
        x = F.relu(x)                           # torch.Size([4, 6, 28, 28])
        x = self.pool(x)                        # torch.Size([4, 6, 14, 14]), h_out = h_in / kernel_size = 28 / 2 = 14 for height and width
        x = self.conv2(x)                       # torch.Size([4, 16, 10, 10]), h_out = h_in = kernel_size + 1 = 14 - 5 + 1 = 10
        x = F.relu(x)                           # torch.Size([4, 16, 10, 10])
        x = self.pool(x)                        # torch.Size([4, 16, 5, 5])
        x = x.view(-1, 16 * 5 * 5)              # torch.Size([4, 400])
        x = F.relu(self.fc1(x))                 # torch.Size([4, 120])
        x = F.relu(self.fc2(x))                 # torch.Size([4, 84])
        x = F.relu(self.fc4(x))                 # torch.Size([4, 84])
        x = self.fc3(x)                         # torch.Size([4, 10])
        return x