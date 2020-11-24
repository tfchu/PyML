import torch
import torch.nn as nn
import numpy as np

sgd = True                                 # when true, use sgd and pick one sample at a time 

step_size = 0.1
d = 2                                       # features
n = 50                                      # samples
X = torch.randn(n,d)                        # random input (50x2)
true_w = torch.tensor([[-1.0], [2.0]])      # actual weights (2x1)
y = X @ true_w + torch.randn(n,1) * 0.1     # label (50x1)

# y_hat (50 samples x 1 output) = x (50 sample x 2 features) @ w (2x1)
#   one sample: y = x_f0 * w0 + x_f1 * w1, 2 features (f0, f1), 2 weights (w0, w1)
# model has no activation function
linear_module = nn.Linear(d, 1, bias=False) # 1st layer - 2 features, 1 neuron
loss_func = nn.MSELoss()                    # loss functionbetween y_hat (output), y (label)

optim = torch.optim.SGD(linear_module.parameters(), lr=step_size)   # optimizer (the way to update weights)

print('iter,\tloss,\tw')

for i in range(20):
    if sgd:
        rand_idx = np.random.choice(n)      
        x = X[rand_idx]                     # (1 sample x 2 features) take a random input
        y_hat = linear_module(x)            # (1 sample x 1 output) compute output y_hat
        loss = loss_func(y_hat, y[rand_idx])# scalar, only compute the loss on the single point
    else:
        y_hat = linear_module(X)            # (50x1)
        loss = loss_func(y_hat, y)          # scalar 
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    print('{},\t{:.2f},\t{}'.format(i, loss.item(), linear_module.weight.view(2).detach().numpy()))

print('\ntrue w\t\t', true_w.view(2).numpy())
print('estimated w\t', linear_module.weight.view(2).detach().numpy())