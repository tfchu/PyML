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
download CIFAR-10 dataset and transform it

CIFAR-10 dataset 
- 60000 32x32 colour images in 10 classes, with 6000 images per class
- There are 50000 training images and 10000 test images
'''
# constants
NUM_SAMPLES = 50000
NUM_BATCH_SIZE = 32
NUM_EPOCHS = 32

# image transformations chained with Compose()
transform = transforms.Compose(
    [transforms.ToTensor(),                                     # PIL or numpy.ndarray to tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   # (mean[1], mean[2], mean[3]), [std[1], std[2], std[3]]). out = (in - mean) / std

# DataLoader issue: cause multiprocess issue if num_workers=2 or above
# - https://stackoverflow.com/questions/53974351/pytorch-getting-started-example-not-working
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=NUM_BATCH_SIZE, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=NUM_BATCH_SIZE, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')              # 10 classes

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

'''
configure network
'''
net = Net()                                         # CNN
net.to(device)                                      # to GPU if available
net.train()                                         # set network to training mode
criterion = nn.CrossEntropyLoss()                   # loss function
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)     # change lr from 0.001 to 0.05
# optimizer = optim.Adam(net.parameters(), lr=0.001)  # optimizer (update parameters)

# added adjust learning rate
# f = lambda epoch: 0.95
# scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer, lr_lambda=f)

''' 
train the network
- feed 1 batch (e.g. 4 images) to network and get outputs
- compute loss
- compute gradient descent
- update parameters
- go back to 1st step until all batches are done
- repeat number of epoch times

output: 
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
start = time.time()                                     # timer
print('[%5s, %5s] %s' % ('epoch', 'batch', 'loss'))     # statistics headers
n = NUM_SAMPLES // NUM_BATCH_SIZE // 5
for epoch in range(NUM_EPOCHS):                                 # loop over the dataset multiple times, change from 2 to ?

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):                       # enumerate() adds index (i) to iterator trainloader starting 0
        inputs, labels = data[0].to(device), data[1].to(device)     # get samples as (inputs, labels) (inputs, labels = data for CPU version)
                                                                    # data is a list of [inputs, labels]. each batch has 4 images as specified
        optimizer.zero_grad()                                       # zero the parameter gradients

        # forward + backward + optimize
        outputs = net(inputs)                                       # get output with forward propagation
        loss = criterion(outputs, labels)                           # get loss
        loss.backward()                                             # compute gradient descent
        optimizer.step()                                            # update parameters

        # scheduler.step()

        # print statistics
        running_loss += loss.item()
        # if i % 2000 == 1999:                                        # print every 2000 mini-batches (i = 1999, 3999, 5999, ...)
        if i % n == (n-1):
            print('[%5d, %5d] %.3f' %
                #   (epoch + 1, i + 1, running_loss / 2000))
                  (epoch + 1, i + 1, running_loss / n))
            running_loss = 0.0

# training time
time_elapsed = datetime.timedelta(seconds = time.time() - start)
print('training time', str(time_elapsed))

# save model
PATH = 'cifar_net.pth'
torch.save(net.state_dict(), PATH)

print('Finished Training')



