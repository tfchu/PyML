'''
test accuracy of 1 set of images
check performance of entire data set
check performance of each class
'''
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
show image
'''
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
PATH = 'cifar_net.pth'

'''
test 1 set of images (4 images)
'''
# load 1 set of images
dataiter = iter(testloader)
images, labels = dataiter.next()    # load 1 set of 4 images everytime dataiter.next() is called

# show image labels (GroundTruth)
print('GroundTruth:\t', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# use our cnn to predict
# load saved model
net = Net()
net.load_state_dict(torch.load(PATH))

# get energies for the 10 classes
# The higher the energy for a class, the more the network thinks that the image is of the particular class
outputs = net(images)

_, predicted = torch.max(outputs, 1)

# show predictions
print('Predicted:\t', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# show images
imshow(torchvision.utils.make_grid(images))

'''
performance for entire data set
'''
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data                           # get image and label
        outputs = net(images)                           # get predicted output (energies for the 10 classes)
        _, predicted = torch.max(outputs.data, 1)       # get label of the class with highest energy
        total += labels.size(0)                         # get total count (may have multiple labels per set)
        correct += (predicted == labels).sum().item()   # get correct count

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

'''
check performance of each class
'''
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# get accuracy of each class
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))