'''
download data, train model and save model
'''
import time, datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from cnn_net import Net

'''
device to train on
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('train on', device)

'''
download dataset
'''
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
'''
CIFAR-10 dataset 
60000 32x32 colour images in 10 classes, with 6000 images per class
There are 50000 training images and 10000 test images
'''
# https://stackoverflow.com/questions/53974351/pytorch-getting-started-example-not-working
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)          # cause multiprocess issue if num_workers=2 (default)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# '''
# show image
# '''
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# # show images
# imshow(torchvision.utils.make_grid(images))

net = Net()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)     # change lr from 0.001 to 0.05

# added adjust learning rate
# f = lambda epoch: 0.95
# scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer, lr_lambda=f)

# train the network
'''
[1,  2000] loss: 2.180
[1,  4000] loss: 1.829
[1,  6000] loss: 1.658
[1,  8000] loss: 1.554
[1, 10000] loss: 1.493
[1, 12000] loss: 1.459
[2,  2000] loss: 1.395
[2,  4000] loss: 1.359
[2,  6000] loss: 1.344
[2,  8000] loss: 1.328
[2, 10000] loss: 1.290
[2, 12000] loss: 1.274
Finished Training
'''
start = time.time()     # timer
for epoch in range(16):  # loop over the dataset multiple times, change from 2 to 16

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)     # inputs, labels = data for CPU version

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # scheduler.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
time_elapsed = datetime.timedelta(seconds = time.time() - start)
print('training time', str(time_elapsed))

PATH = 'cifar_net.pth'
torch.save(net.state_dict(), PATH)

print('Finished Training')



