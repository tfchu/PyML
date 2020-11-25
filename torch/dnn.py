import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

d = 1
n = 200
X = torch.rand(n,d)
y = 4 * torch.sin(np.pi * X) * torch.cos(6*np.pi*X**2)

# plt.scatter(X.numpy(), y.numpy())
# plt.title('plot of $f(x)$')
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.show()

step_size = 0.05    # learning rate of optimizer
m = 0.9             # momentum of optimizer, help converge much faster in some cases
n_epochs = 6000     # go thru all samples once is 1 epoch
n_hidden_1 = 32     # hidden layer 1 neurons
n_hidden_2 = 32     # hidden layer 2 neurons 
d_out = 1           # output dimension 

# model
# 1 feature -> 32 neurons with Tanh() -> 32 neurons with Tanh() -> 1 output
# with 200 samples
# hidden layer 1: inputs (200 x 1 feature) @ weights (1 x 32) = 200 x 32
# hidden layer 2: (200 x 32) @ weights (32 x 32) = 200 x 32
# output layer: (200 x 32) @ weights (32 x 1) = (200 x 1), i.e. each sample has 1 output
neural_network = nn.Sequential(
                            nn.Linear(d, n_hidden_1), 
                            nn.Tanh(),
                            nn.Linear(n_hidden_1, n_hidden_2),
                            nn.Tanh(),
                            nn.Linear(n_hidden_2, d_out)
                            )
loss_func = nn.MSELoss()                                                        # loss function
optim = torch.optim.SGD(neural_network.parameters(), lr=step_size, momentum=m)  # optimizer

# lambda1 = lambda epoch: 0.999
# scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer=optim, lr_lambda=lambda1)

# lambda1 = lambda epoch: epoch // 30
# lambda2 = lambda epoch: 0.95 ** epoch
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda2)

# print output every 10 epochs
print('iter,\tloss')
for i in range(n_epochs):
    # forward propagation
    y_hat = neural_network(X)
    # loss
    loss = loss_func(y_hat, y)
    # zero optimizer
    optim.zero_grad()
    # backpropagation
    loss.backward()
    # gradient descent (update weights)
    optim.step()
    # # scheduler
    # scheduler.step()
    # print loss
    if i % (n_epochs // 10) == 0:
        print(optim.param_groups[0]['lr'])
        print('{},\t{:.2f}'.format(i, loss.item()))

# X is between 0 ~ 1 (50 samples), generate X_grid from 0 ~ 1, 50 points
# torch.from_numpy returns double, convert to float()
# view(-1, d=1): reshape from 1x50 to 50x1 
#   (-1 select 1st dim which is 50), d = 1
X_grid = torch.from_numpy(np.linspace(0,1,50)).float().view(-1, d)
# output y_hat from updated network (updated weights)
y_hat = neural_network(X_grid)
# plot f(x) (original) and f_hat(x) (predicted)
plt.scatter(X.numpy(), y.numpy(), label='$f(x)$')
plt.plot(X_grid.detach().numpy(), y_hat.detach().numpy(), 'r', label='$\hat{f}(x)$')
plt.title('plot of $f(x)$ and $\hat{f}(x)$')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()