import math
import os
import sys
import time

import git
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from git import RemoteProgress, Repo


def draw_parameters_densities(samples, num_col=2):
    extended_samples = dict()
    for name, s in samples.items():
        for i in range(0, s.shape[1]):
            if s.shape[1] > 1:
                name_extended = name + ' ' + str(i)
            else:
                name_extended = name
            extended_samples[name_extended] = s[:, i]
    pd_samples =  pd.DataFrame(extended_samples)
    
    numbers_of_param = pd_samples.shape[1]
    fig, axis = plt.subplots(math.ceil(numbers_of_param/num_col),
    num_col, figsize=(15, 15))

    for key, ax in zip(pd_samples.columns, axis.ravel()):
        ax.set_title(key, fontsize=7)
        sns.histplot(pd_samples[key],
                    ax=ax,
                    bins=100,
                    color="blue",
                    kde=True)
                    #axlabel=False,
                #hist_kws=dict(edgecolor="black"))

    plt.subplots_adjust(hspace=0.5)
    plt.show()

def variance_estimation_from_samples(S, bn=None, hq=None):
    num_samples, size = S.shape
    if bn is None:
        bn = int(np.floor(np.sqrt(num_samples)))
    if hq is None:
        hq = 0.95
    lq = 1 - hq

    nboot = int(num_samples - bn)
    Lq = torch.zeros(nboot, size)
    Hq = torch.zeros(nboot, size)
    for ii in range(nboot):
        Si = S[ii:ii + bn]
        Lq[ii, :] = torch.quantile(Si, lq, dim=0)
        Hq[ii, :] = torch.quantile(Si, hq, dim=0)
    mlq = torch.mean(Lq, dim=0, keepdim=True)
    mhq = torch.mean(Hq, dim=0, keepdim=True)
    vhq = bn / (num_samples - bn + 1) * torch.sum((Hq - mhq) ** 2, dim=0)
    vlq = bn / (num_samples - bn + 1) * torch.sum((Lq - mlq) ** 2, dim=0)
    return vhq, vlq, Lq, Hq


class CloneProgress(RemoteProgress):
    def update(self, op_code, cur_count, max_count=None, message=''):
        if message:
            print(message)
            

def clone_git(repo_dir=None):
    """This function adds the Sliced_Kernelized_Stein_Discrepancy needed for GPSliceSVGD 
    in the current repository
    """
    git_url = 'https://github.com/WenboGong/Sliced_Kernelized_Stein_Discrepancy'
    if repo_dir is None:
        repo_dir = os.path.join(os.path.dirname(os.getcwd()),
                                    'Sliced_Kernelized_Stein_Discrepancy')
        #print(repo_dir)
        #repo_dir = os.path.join(os.getcwd(), 'Sliced_Kernelized_Stein_Discrepancy')
    Repo.clone_from(git_url, repo_dir, progress=CloneProgress())  
    sys.path.append(os.path.join(repo_dir, 'src'))
    sys.path.append(os.path.join(os.path.join(repo_dir, 'src'), 'Divergence'))










