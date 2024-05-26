import torch
import math
import numpy as np

g1 = lambda t: t
g2 = lambda t: (2*t-1)**2
g3 = lambda t: torch.sin(t)/(2-torch.sin(t))
g4 = lambda t: 0.1*torch.sin(t)+0.2*torch.cos(t)+0.3*torch.sin(t)**2+0.4*torch.cos(t)**3+0.5*torch.sin(t)**3

def generate_unif_data(N, d):
    train_x = torch.rand(N, d)
    return train_x

def generate_compound_symetry_data(N, d, t = 1):
    W = torch.rand(N, d)
    U = torch.rand(N, 1)
    train_x = (W+t*U)/(t+1)
    return train_x

def generate_trimmed_AR_data(N, d, r = 0.5):
    W = torch.randn(N, d)
    train_x = torch.zeros(N, d)
    train_x[:, 0] = torch.rand(N)
    for i in np.arange(1, d):
        train_x[:, i] = r*train_x[:, i-1] + ((1-r**2)**0.5)*W[:, i]
    train_x[train_x>2.5] = 2.5
    train_x[train_x<-2.5] = -2.5
    max_train, _ = torch.max(train_x, dim=1, keepdim=True)
    min_train, _ = torch.min(train_x, dim=1, keepdim=True)
    train_x = (train_x-min_train)/(max_train-min_train)
    return train_x

def examplef1(x):
    f0 = 5 * g1(x[:, 0])
    f1 = 3 * g2(x[:, 1])
    f2 = 4 * g3(2 * math.pi * x[:, 2])
    f3 = 6 * g4(2 * math.pi * x[:, 3])
    return f0, f1, f2, f3

def examplef2(x):
    f0 = 5*g1(x[:, 0])
    f1 = 3*g2(x[:, 1])
    f2 = 4*g3(2 * math.pi * x[:, 2])
    f3 = 6*g4(2 * math.pi * x[:, 3])
    f01 = 4*g3(2 * math.pi * x[:, 1]*x[:, 0])
    f02 = 6*g2((x[:, 0] + x[:, 2])/2)
    f24 = 4*g1(x[:, 4] * x[:, 2])

    # f0 = g1(x[:, 0])
    # f1 =  g2(x[:, 1])
    # f2 =  g3(2 * math.pi * x[:, 2])
    # f3 = g4(2 * math.pi * x[:, 3])
    # f01 = g3(2 * math.pi * x[:, 1] * x[:, 0])
    # f02 = g2((x[:, 0] + x[:, 2]) / 2)
    # f24 = g1(x[:, 4] * x[:, 2])
    return f0, f1, f2, f3, f01, f02, f24


def example1(N=100, d=10, noise = 3.03, data='unif', **kwargs):
    # Example 1 in the paper nihms
    train_x = None
    if data == 'unif':
        train_x = generate_unif_data(N, d)
    if data == 'sym':
        train_x = generate_compound_symetry_data(N, d, **kwargs)
    if data == 'AR':
        train_x = generate_trimmed_AR_data(N, d, **kwargs)
    train_f0, train_f1, train_f2, train_f3 = examplef1(train_x)
    train_f = train_f0 + train_f1 + train_f2 + train_f3
    train_y = train_f + np.sqrt(noise)*torch.randn(N)
    return train_x, train_y, train_f, train_f0, train_f1, train_f2, train_f3

def example2(N=100, d=10, noise = 0.44, data='unif', **kwargs):
    # Example 1 in the paper nihms
    train_x = None
    if data == 'unif':
        train_x = generate_unif_data(N, d)
    if data == 'sym':
        train_x = generate_compound_symetry_data(N, d, **kwargs)
    if data == 'AR':
        train_x = generate_trimmed_AR_data(N, d, **kwargs)
    train_f0, train_f1, train_f2, train_f3, train_f01, train_f02, train_f23  = examplef2(train_x)
    train_f = train_f0 + train_f1 + train_f2 + train_f3 + train_f01 + train_f02 + train_f23
    train_y = train_f + np.sqrt(noise)*torch.randn(N)
    return train_x, train_y, train_f, train_f0, train_f1, train_f2, train_f3, train_f01, train_f02, train_f23

# from scipy.signal import savgol_filter

# def anova_center(example, r=1):
#     t = torch.arange(0, 1001, 1)/1000
#     dt = t[1]-t[0]
#     F = example(t)
#     c = list()
#     for f in F:
#         if r==1:
#             c.append(torch.sum(f)*dt+f[-1]-f[0])
#         if r==2:
#             df1 = torch.diff(f) / dt
#             df1smooth = savgol_filter(df1, window_length=11, polyorder=2)
#             c.append(torch.sum(f) * dt + f[-1] - f[0] + df1smooth[-1] - df1smooth[0])
#     return c