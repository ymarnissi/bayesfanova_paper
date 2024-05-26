from bayesanova.models import sobolev_kernel
import torch
import numpy as np
from sklearn.utils import shuffle



def get_adaptive_scale_from_covariance_trace(train_x, poly_order, model_order=1, c=None, correction=False):
    D = train_x.shape[1]
    N = train_x.shape[0]
    adaptive_scale_main_effect = torch.ones(D)
    adaptive_scale_interaction_effect = None
    covar_module = sobolev_kernel.univ_main_effect(poly_order, c=c)
    for i in range(D):
        adaptive_scale_main_effect[i] = covar_module(train_x[:, i]).evaluate().diag().sum()
    if model_order == 2:
        adaptive_scale_interaction_effect = torch.ones(D*(D-1)//2)
        index = torch.tensor([[i, j] for i in np.arange(0, D - 1)
                                for j in np.arange(i + 1, D)], dtype=torch.long)
        for i in range(D*(D-1)//2):
            covar_module = sobolev_kernel.univ_first_order_interaction_effect(poly_order, c=c, correction=correction)
            adaptive_scale_interaction_effect[i] = covar_module(train_x[:, index[i,:]]).evaluate().diag().sum()
    adaptive_scale = (adaptive_scale_main_effect, adaptive_scale_interaction_effect)
    return adaptive_scale




def k_fold(train_x, train_y, tab, method, folds=4, random_state=12345):
    L = tab.numel()
    N = train_y.numel()
    # tab, ind = torch.sort(tab, descending=True)
    fold_size = int(np.floor(N / folds))
    rand_index = shuffle([i for i in range(N)], random_state=random_state)
    #rand_index = np.random.permutation(range(N))
    loss = torch.zeros(L, folds)
    for i in range(L):
        for k in range(folds):
            select_test_index = rand_index[k * fold_size:(k + 1) * fold_size]
            test_x_k = train_x[select_test_index, :]
            test_y_k = train_y[select_test_index]
            train_x_k = torch.cat((train_x[0:select_test_index[0], :], train_x[select_test_index[-1]:, :]), dim=0)
            train_y_k = torch.cat((train_y[0:select_test_index[0]], train_y[select_test_index[-1]:]), dim=0)
            loss[i, k] = method(train_x_k, train_y_k, test_x_k, test_y_k, tab[i])
    loss = loss.mean(dim=1)
    # Remove None values : where the result is nan 
    loss[torch.isnan(loss)] = torch.tensor(np.inf)
    
    return tab[int(torch.argmin(loss))], loss









