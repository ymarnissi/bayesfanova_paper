import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch



def bernoulli_poly(t, r=1):
    """Bernouilli polynomial

    Args:
        t (float, tensor): the variable
        r (int, optional): the polynomial order. Defaults to 1.

    Returns:
        value (float, tensor)
    """
    if r==1:
        b = t-1/2
    elif r==2:
        b = t**2-t+1/6
    elif r==3:
        b = t**3-3/2*(t**2)+1/2*t
    elif r==4:
        b = t**4-2*(t**3)+t**2-1/30
    else:
        b = t
    return b

def get_parameters_from_GPmodel(model):
    """Get the parameters of the GP model

    Args:
        model (gpytorch.model): GP model

    Returns:
        dict: dictinnary containing the name and the value of the parameters
    """
    parameters = dict()
    for param_name, param in model.named_parameters():
        parameters[param_name] = param
    return parameters

def first_order_InteractionEffect_to_index(D, k=None):
    """Get the index of the interaction effect component between the couple in k=[i,j]

    Args:
        D (int): dimension
        k (list, optional): a list that contains the indexes of the covariables. Defaults to None.

    Returns:
        int: index of the interaction effect between i=k[0] and j=k[1]
    """
    if k is None:
        k = [0, 1]
    if k[0]>=k[1] or np.array(k).size != 2:
        raise ValueError("Input index k for first order interaction effect should be bi-dimensional and k[0]<k[1]")
    index_keys = [str([i,j]) for i in np.arange(0, D - 1)
                        for j in np.arange(i + 1, D)]
    index_values = torch.arange(0, D*(D-1)/2, 1)
    index = dict(zip(index_keys, list(index_values.to(int).numpy())))
    return index[str(k)]

def positive_definite_full_covar(train_covar,test_train_covar, test_covar, my_eps=1e-6):
    i = 0
    test = False
    while (test == False):
        try : 
            full_covar = train_covar.add_jitter(my_eps*i).cat_rows(test_train_covar, test_covar.add_jitter(my_eps*(2**i)))
            test = True
        except:
            test = False
            i = i+1
    return full_covar


