# Script adapted from https://github.com/amzn/orthogonal-additive-gaussian-processes

# download UCI datasets from https://github.com/duvenaud/additive-gps/ or https://github.com/hughsalimbeni/bayesian_benchmarks/ and save to ./data directory

import os
import urllib.request
import pandas as pd 
from scipy import io
import numpy as np 
import pickle
from scipy.io import loadmat
from bayesanova.examples.simulated_data import example2, example1
import torch 

pd.options.display.max_columns = None
pd.options.display.width = 250

data_path_prefix0 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/simulated0")
) + '/'

data_path_prefix1 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/simulated1")
) + '/'


# Create if data does not exist
if not os.path.exists(data_path_prefix0):
        os.makedirs(data_path_prefix0)
        
np.random.seed(seed=18)
torch.manual_seed(18)

generate_data1 = False
if generate_data1 :
    # Example 1 
    Q = 10
    N = 100
    d = 55
    noise = [1.00, 2.19, 5.19] 
    rho = [0.5, 0.8]

    # Uniform training dataset 
    for k in range(Q):
        for i, j in enumerate(noise):
            train_x, train_y, train_f, train_f0, train_f1, train_f2, train_f3 = example1(N=N, d=d, noise=j, data='unif')
            test_x, test_y, test_f, test_f0, test_f1, test_f2, test_f3 = example1(N=50, d=d, noise=0, data='unif')
            train_active = torch.cat([train_f0, train_f1, train_f2, train_f3], dim=0)
            test_active = torch.cat([test_f0, test_f1, test_f2, test_f3], dim=0)
            foldername = data_path_prefix0 + '01-Uniform/noise_'+str(i)
            
            if not os.path.exists(foldername):
                os.makedirs(foldername)

            
            io.savemat(f'{foldername}/dataset_{k}.mat', {'train_x': train_x.numpy(), 'train_y':train_y.numpy(), 
                                            'train_f':train_f.numpy(), 
                                            'train_active': train_active.numpy(), 
                                            'test_x': test_x.numpy(),
                                            'test_y':test_y.numpy(), 
                                            'test_f':test_f.numpy(),
                                            'test_active': test_active.numpy()})   
            
    # AR training dataset 
    for k in range(Q):
        for i, j in enumerate(noise):
            for ii, r in enumerate(rho):
                foldername = data_path_prefix0+'02-AR/rho_'+str(ii)+'/noise_'+str(i)
                if not os.path.exists(foldername):
                    os.makedirs(foldername)
                
                train_x, train_y, train_f, train_f0, train_f1, train_f2, train_f3 = example1(N=N, d=d, noise=j, data='AR', r=r)
                test_x, test_y, test_f, test_f0, test_f1, test_f2, test_f3 = example1(N=50, d=d, noise=0, data='unif')
                train_active = torch.cat([train_f0, train_f1, train_f2, train_f3], dim=0)
                test_active = torch.cat([test_f0, test_f1, test_f2, test_f3], dim=0)
                io.savemat(f'{foldername}/dataset_{k}.mat', {'train_x': train_x.numpy(), 'train_y':train_y.numpy(), 
                                            'train_f':train_f.numpy(), 
                                            'train_active': train_active.numpy(), 
                                            'test_x': test_x.numpy(),
                                            'test_y':test_y.numpy(), 
                                            'test_f':test_f.numpy(),
                                            'test_active': test_active.numpy()})   
                
                
                foldername = data_path_prefix0+'03-AR/rho_neg_'+str(ii)+'/noise_'+str(i)
                if not os.path.exists(foldername):
                    os.makedirs(foldername)
                    
                train_x, train_y, train_f, train_f0, train_f1, train_f2, train_f3 = example1(N=N, d=d, noise=j, data='AR', r=-r)
                test_x, test_y, test_f, test_f0, test_f1, test_f2, test_f3 = example1(N=50, d=d, noise=0, data='unif')
                train_active = torch.cat([train_f0, train_f1, train_f2, train_f3], dim=0)
                test_active = torch.cat([test_f0, test_f1, test_f2, test_f3], dim=0)
                io.savemat(f'{foldername}/dataset_{k}.mat', {'train_x': train_x.numpy(), 'train_y':train_y.numpy(), 
                                            'train_f':train_f.numpy(), 
                                            'train_active': train_active.numpy(), 
                                            'test_x': test_x.numpy(),
                                            'test_y':test_y.numpy(), 
                                            'test_f':test_f.numpy(),
                                            'test_active': test_active.numpy()})   
            
        

    # CS dataset 
    for k in range(Q):
        for i, j in enumerate(noise):
            for ii, r in enumerate(rho):
                
                foldername = data_path_prefix0+'04-CS/rho_'+str(ii)+'/noise_'+str(i)
                if not os.path.exists(foldername):
                    os.makedirs(foldername)
                
                t = np.sqrt(r/(1-r))
                train_x, train_y, train_f, train_f0, train_f1, train_f2, train_f3 = example1(N=N, d=d, noise=j, data='sym', t=t)
                test_x, test_y, test_f, test_f0, test_f1, test_f2, test_f3 = example1(N=50, d=d, noise=0, data='unif')
                train_active = torch.cat([train_f0, train_f1, train_f2, train_f3], dim=0)
                test_active = torch.cat([test_f0, test_f1, test_f2, test_f3], dim=0)
                io.savemat(f'{foldername}/dataset_{k}.mat', {'train_x': train_x.numpy(), 'train_y':train_y.numpy(), 
                                            'train_f':train_f.numpy(), 
                                            'train_active': train_active.numpy(), 
                                            'test_x': test_x.numpy(),
                                            'test_y':test_y.numpy(), 
                                            'test_f':test_f.numpy(),
                                            'test_active': test_active.numpy()})   
        
                
        






torch.manual_seed(100)
np.random.seed(seed=200)


generate_data2 = True

if generate_data2:


    # Example 2


    Q = 10
    N = 100
    d = 10
    noise = [1.00, 2.19, 5.19] 
    rho = [0.5, 0.8]


    # Uniform training dataset 
    for k in range(Q):
        for i, j in enumerate(noise):
            train_x, train_y, train_f, train_f0, train_f1, train_f2, train_f3, train_f01, train_f02, train_f23 = example2(N=N, d=d, noise=j, data='unif')
            test_x, test_y, test_f, test_f0, test_f1, test_f2, test_f3, test_f01, test_f02, test_f23 = example2(N=50, d=d, noise=0, data='unif')
            train_active = torch.cat([train_f0, train_f1, train_f2, train_f3, train_f01, train_f02, train_f23], dim=0)
            test_active = torch.cat([test_f0, test_f1, test_f2, test_f3, test_f01, test_f02, test_f23], dim=0)

            foldername = data_path_prefix1 + '01-Uniform/noise_'+str(i)
            
            if not os.path.exists(foldername):
                os.makedirs(foldername)

            
            io.savemat(f'{foldername}/dataset_{k}.mat', {'train_x': train_x.numpy(), 'train_y':train_y.numpy(), 
                                            'train_f':train_f.numpy(), 
                                            'train_active': train_active.numpy(), 
                                            'test_x': test_x.numpy(),
                                            'test_y':test_y.numpy(), 
                                            'test_f':test_f.numpy(),
                                            'test_active': test_active.numpy()})   
            
    # AR training dataset 
    for k in range(Q):
        for i, j in enumerate(noise):
            for ii, r in enumerate(rho):
                foldername = data_path_prefix1+'02-AR/rho_'+str(ii)+'/noise_'+str(i)
                if not os.path.exists(foldername):
                    os.makedirs(foldername)
                
                train_x, train_y, train_f, train_f0, train_f1, train_f2, train_f3, train_f01, train_f02, train_f23 = example2(N=N, d=d, noise=j, data='AR', r=r)
                test_x, test_y, test_f, test_f0, test_f1, test_f2, test_f3, test_f01, test_f02, test_f23 = example2(N=50, d=d, noise=0, data='unif')
                train_active = torch.cat([train_f0, train_f1, train_f2, train_f3, train_f01, train_f02, train_f23], dim=0)
                test_active = torch.cat([test_f0, test_f1, test_f2, test_f3, test_f01, test_f02, test_f23], dim=0)
                io.savemat(f'{foldername}/dataset_{k}.mat', {'train_x': train_x.numpy(), 'train_y':train_y.numpy(), 
                                            'train_f':train_f.numpy(), 
                                            'train_active': train_active.numpy(), 
                                            'test_x': test_x.numpy(),
                                            'test_y':test_y.numpy(), 
                                            'test_f':test_f.numpy(),
                                            'test_active': test_active.numpy()})   
                
                
                foldername = data_path_prefix1+'03-AR/rho_neg_'+str(ii)+'/noise_'+str(i)
                if not os.path.exists(foldername):
                    os.makedirs(foldername)
                    
                train_x, train_y, train_f, train_f0, train_f1, train_f2, train_f3, train_f01, train_f02, train_f23 = example2(N=N, d=d, noise=j, data='AR', r=-r)
                test_x, test_y, test_f, test_f0, test_f1, test_f2, test_f3, test_f01, test_f02, test_f23 = example2(N=50, d=d, noise=0, data='unif')
                train_active = torch.cat([train_f0, train_f1, train_f2, train_f3, train_f01, train_f02, train_f23], dim=0)
                test_active = torch.cat([test_f0, test_f1, test_f2, test_f3, test_f01, test_f02, test_f23], dim=0)
                io.savemat(f'{foldername}/dataset_{k}.mat', {'train_x': train_x.numpy(), 'train_y':train_y.numpy(), 
                                            'train_f':train_f.numpy(), 
                                            'train_active': train_active.numpy(), 
                                            'test_x': test_x.numpy(),
                                            'test_y':test_y.numpy(), 
                                            'test_f':test_f.numpy(),
                                            'test_active': test_active.numpy()})   
            
        

    # CS dataset 
    for k in range(Q):
        for i, j in enumerate(noise):
            for ii, r in enumerate(rho):
                
                foldername = data_path_prefix1+'04-CS/rho_'+str(ii)+'/noise_'+str(i)
                if not os.path.exists(foldername):
                    os.makedirs(foldername)
                
                t = np.sqrt(r/(1-r))
                train_x, train_y, train_f, train_f0, train_f1, train_f2, train_f3, train_f01, train_f02, train_f23 = example2(N=N, d=d, noise=j, data='sym', t=t)
                test_x, test_y, test_f, test_f0, test_f1, test_f2, test_f3, test_f01, test_f02, test_f23 = example2(N=50, d=d, noise=0, data='unif')
                train_active = torch.cat([train_f0, train_f1, train_f2, train_f3, train_f01, train_f02, train_f23], dim=0)
                test_active = torch.cat([test_f0, test_f1, test_f2, test_f3, test_f01, test_f02, test_f23], dim=0)
                io.savemat(f'{foldername}/dataset_{k}.mat', {'train_x': train_x.numpy(), 'train_y':train_y.numpy(), 
                                            'train_f':train_f.numpy(), 
                                            'train_active': train_active.numpy(), 
                                            'test_x': test_x.numpy(),
                                            'test_y':test_y.numpy(), 
                                            'test_f':test_f.numpy(),
                                            'test_active': test_active.numpy()})   
        
                
        



