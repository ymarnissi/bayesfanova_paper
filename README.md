<!-- PROJECT LOGO -->

  <h3 align="center">Bayesanova project</h3>

[![Paper](http://img.shields.io/badge/paper-arxiv-b31b1b.svg)](https://openreview.net/pdf?id=dV9QGostQk)
[![Conference](http://img.shields.io/badge/ICML-2024-4b44ce)](https://https://icml.cc)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ymarnissi/Sampling#license)

[![Python](https://img.shields.io/badge/-Python_3.8-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/PyTorch_1.8-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![GPytorch](https://img.shields.io/badge/GPytorch-1.4-blue)](https://gpytorch.ai/)


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Paper](#Paper)


<!-- ABOUT THE PROJECT -->
## About The Project

This package implements non-parametric regression with Anova Decompsoition and Sobolev Kernels. It provides: (1) Implementation of the state-of-the-art methods using optimization (ss-Anova, COSSO, ACOSSO) and (2) Estimation methods including MAP and Bayesian sampling for fonctional component selection with MCMC or Stein methods.  


<!-- Project Structure -->
## Project Structure

The directory structure of this package is as follows:

```
├── bayesanova                   <- Source code
│   ├── anova_estimators         <- estimators ss, cosso, lgbaycoss(Local Global BAYesian COmponent Selection)
│   ├── models                   <- model (gaussian process model), prior, sobolev_kernel, likelihood 
│   ├── samplers                 <- stein and mcmc
│   ├── examples                 <- some simulated data and utils
|
├── notebooks                    <- demos
│   ├── Demo_anova_estimators.ipynb    
│
├── .gitignore                   <- List of files/folders ignored by git
├── requirements.txt             <- File for installing python dependencies
├── setup.cfg                    <- Configuration of linters 
└── README.md

```


<!-- GETTING STARTED -->

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```


<!-- USAGE EXAMPLES -->
## Usage

see demos in notebooks folder

```python
from bayesanova.anova_estimators.ss import SS # Smoothing-Spline
from bayesanova.anova_estimators.cosso import COSSO # Component-Selection Smoothing-Spline
from bayesanova.anova_estimators.lgbayescos import LGBayesCOS # Local-Global Bayesian Component selection


'''
# Define the estimators :  refer to the demo notebooks for the inputs of each estimator
'''
ss_estimator = SS(ndims=10, weight=None, model_order=1, poly_order=1)

cosso_estimator = COSSO(ndims=10, weight=None, model_order=1, poly_order=1)

mcmc_estimator = LGBayesCOS(model=gpmodel, 
                            constraints_dict=constraints_dict,
                            priors_dict=priors_dict, 
                            sampler='HMC', # or Stein
                            # Now define the kwargs of the sampler
                            .....) 



''' Training
# Train the estimator given train data 
'''
estimator.train(train_x, train_y, **kwargs) 

''' Prediction 
# For SS and COSSO, predict returns tensors or dicts {key:tensor}
# For Bayesian methods, predict returns distributions or dicts {key:distributions}
# Several ways to call predict
''' 
estimator.predict(test_x)  # Return the latent function f=c+\sum_k f_k
estimator.predict(test_x, kind='outcome') # Return the outcome, only available for Bayesian estimators
estimator.predict(test_x, kind='main', component_index=0) # Return the main component 0
estimator.predict(test_x, kind='main', component_index=[0, 1]) # Return the main components 0 and 1
estimator.predict(test_x, kind='main') # Return all main components
estimator.predict(test_x, kind='interaction', component_index=[0,1]) # Return the interaction component (0, 1)
estimator.predict(test_x, kind='interaction', component_index=[[0,1], [1,3]]) # Return the interaction components (0, 1) and (1, 3)
estimator.predict(test_x, kind='interaction') # Return all interaction components
estimator.predict(test_x, kind='residual') # Return residual component (only available for Bayesian method)
```

<!-- Paper -->
## Paper
```
@inproceedings{marnissi2024unified,
  title={A Unified View of FANOVA: A Comprehensive Bayesian Framework for Component Selection and Estimation},
  author={Marnissi, Yosra and Leiber, Maxime},
  booktitle={International Conference on Machine Learning},
  pages={34866--34894},
  year={2024},
  organization={PMLR}
}
```
