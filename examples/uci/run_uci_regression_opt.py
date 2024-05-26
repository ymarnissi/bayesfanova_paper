
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import io
from sklearn.model_selection import KFold
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


import gpytorch
import copy 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import pickle

from bayesanova.anova_estimators import ss, cosso, lgbayescos
from bayesanova.models import model, prior
from bayesanova.anova_estimators.utils import get_adaptive_scale_from_covariance_trace

import logging
_log = logging.getLogger(__name__)


normalize = True # Normalize y 

matplotlib.rcParams.update({"font.size": 20})

# +
# data from repo: https://github.com/duvenaud/additive-gps/blob/master/data/regression/
# this script is for experiments in Sec 5.1 for regression problems in the paper
data_path_prefix = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/uci")
) + '/'


num_samples = 2_000  # 2_000
warmup_steps = 3_000  # 3_000
num_samples_inf_default = 100 # 50
subsample = num_samples // num_samples_inf_default # 10
noise_scale = 1.0
mode = 'estimation'

dataset_names = ["autompg", "housing", "concrete", "pumadyn", "energy", "yacht", "stock",
                "forest",  "servo", "container", "demand", "stock2" ]
                #"fertility", "machine", "pendulum", "servo", "wine"]

filenames = [
    data_path_prefix + "autompg.mat", # N = 392, D = 7
    data_path_prefix + "housing.mat", # N = 506, D = 13 
    data_path_prefix + "r_concrete_1030.mat", # N = 1030, D = 8
    data_path_prefix + "pumadyn8nh.mat", # N = 8192, D = 8
    data_path_prefix + "energy", # N = 768, D = 8
    data_path_prefix + "yacht", # N = 306, D = 6
    data_path_prefix + "stock", # N = 536, D = 8
    data_path_prefix + "forest", # N = 517,  D = 12    
    #data_path_prefix + "fertility", # N = 100, D = 10
    #data_path_prefix + "machine", # N = 209, D = 7
    #data_path_prefix + "pendulum", # N = 630, D = 9
    #data_path_prefix + "servo", # N = 167, D = 4
    #data_path_prefix + "wine" # N = 178, D = 14    
    data_path_prefix + "servo", # N = 166, D = 4
    #data_path_prefix + "wine" # N = 178, D =
    data_path_prefix + "container", # N = 15, D = 2
    data_path_prefix + "demand", # N = 60, D = 12
    #data_path_prefix + "slump", # N = 103, D =
    data_path_prefix + "stock2", # N = 315, D = 11
    #data_path_prefix + "computer", # N = 209, D =
]

filenames = dict(zip(dataset_names, filenames))



with open(data_path_prefix+'data_info_inputs.pkl', 'rb') as f:
    covariate_names = pickle.load(f)

with open(data_path_prefix+'data_info_outputs.pkl', 'rb') as f:
    output_names = pickle.load(f)
    
# Transform samples to estimates    
def samples2estimate(m, v):
    global_mean = np.mean(m, axis=0, keepdims=True)
    out_variance = v + np.abs(m - global_mean) ** 2
    mean = np.mean(m, axis=0)
    var = np.mean(out_variance, axis=0)
    return mean, var

# Sometimes the variance approximation does not work: some issues with gyptorch.lazy_tensor
def get_variance(d):
    try: 
        a = d.variance
    except:
        a = torch.diagonal(d.covariance_matrix, dim1=-2, dim2=-1)
    return a 

# Coverage estimated from quantiles

def credible_interval_equal_tailed(f_preds, epsilon):
    # f_preds is a scale mixture of Gaussian
    # Use 10000 samples to estimate uncertainty intervals 
    samples_t = f_preds.sample(sample_shape=torch.Size([10_000]))
    samples_t = torch.cat([s for s in samples_t], dim=0)
    upper = torch.quantile(samples_t, 1-epsilon, dim=0)
    lower = torch.quantile(samples_t, epsilon, dim=0)
    return lower, upper


def main():
    """
    :param dataset_name: name of the dataset, should be one of the above dataset_names
    :param k: number of train-test fold, default to 5.
    :param model : either ss, cosso, bayescos, default ss
    :mixing : laplace, horshoe, dirichlet 
    :return: fit the model
    on the dataset, saves the model,
    the model predictive performance,
    and the plot on cumulative Sobol indices.
    """
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--dataset_name", default="pumadyn", type=str, help="dataset name"
    )
    
    args_parser.add_argument(
        "--k", type=int, default=5, help="k-fold train-test splits"
    )
    
    args_parser.add_argument(
        "--num_samples_inf", type=int, default=num_samples_inf_default, help="number of samples for inference"
    )
    
    args_parser.add_argument(
        "--model_name",  default='lgbayescos_opt', help="model name"
    )
    
    args_parser.add_argument(
        "--mixing",  default='Exp', help="mixing density for lgbacos"
    )
    
    args_parser.add_argument(
        "--adaptive",  default='NonAdaptive', help="use adaptation from ss estimator"
    )
    
    args_parser.add_argument(
        "--init_strategy",  default='prior', help="initialize strategy"
    )
    
    args_parser.add_argument(
        "--isotropic_info",  default=False, help="isotropic Dirichlet"
    )
    

    args, unknown = args_parser.parse_known_args()
    dataset_name, k, num_samples_inf, model_name, mixing, adaptive, init_strategy, isotropic_info =\
        args.dataset_name, args.k, args.num_samples_inf, args.model_name, args.mixing, args.adaptive, args.init_strategy, args.isotropic_info 

    # save results to outputs folder
    
    kind = adaptive
    
    
    
    output_prefix = os.path.abspath(
        os.path.join(os.path.dirname(__file__), f"./outputs/{dataset_name}/{model_name}/{kind}")
    )
    
    num_samples_inference = num_samples_inf # Number of samples used for inference
    
    if not(model_name=='ss') and not(model_name=='cosso'): 
        model_name = 'lgbayescos_opt'
    
    if model_name=='lgbayescos_opt':
        output_prefix = os.path.abspath(
        os.path.join(os.path.dirname(__file__), f"./outputs/{dataset_name}/{model_name}_{noise_scale}/{mixing}/{kind}")
        )
        
            
         
    
    if not os.path.exists(output_prefix):
        os.makedirs(output_prefix)
        
    logging.basicConfig(filename=f'{output_prefix}/log.log', level=logging.INFO)
    

    np.random.seed(seed=18)
    torch.manual_seed(18)
    
    
    filename = filenames[dataset_name]
    d = io.loadmat(filename)
    
    X, y = d["X"], d["y"]

    idx = np.random.permutation(range(X.shape[0]))

    X = X[idx, :]
    y = y[idx]
    kf = KFold(n_splits=k)
    fold = 0
    
    N, D = X.shape
    _log.info(f"Method {model_name} running...\n")
    _log.info(f"Dataset {dataset_name} of dimension  = {(N, D)}")
    _log.info(f"Number of components to be estimated={D*(D+1)//2} corresponding to {D} main effects and {D*(D-1)//2} interaction effects\n")
    
    RMSE = []
    MAE = []
    R2 = []
    COVERAGE1 = []
    COVERAGE2 = []
    Number_active_components = []
    
    for train_index, test_index in kf.split(X):
        try :
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32).squeeze()
            
            # Normalize on [0,1]
            X_min, _ = torch.min(X_train, 0, keepdim=True)
            X_max, _ = torch.max(X_train, 0, keepdim=True)
            X_train = (X_train-X_min)/(X_max-X_min)
            
            # Normalize y
            if normalize:
                y_mean = np.asscalar(torch.mean(y_train).detach().numpy())
                y_std = np.asscalar(torch.std(y_train).detach().numpy())
                scaler_y = lambda t: (t-y_mean)/y_std
                inverse_scaler_y = lambda t : t*y_std + y_mean
                correct_variance_y = lambda t: t*(y_std**2)
            else:
                scaler_y = lambda t: t
                inverse_scaler_y = lambda t: t 
                correct_variance_y = lambda t: t
                
            
            y_train = scaler_y(y_train)
                
            

            
            
            N, D = X_train.shape
            
            
            # Set the model parameters 
            poly_order = 1 # Polynomial order
            model_order = 2 # Model order 
            residual = False # If true include residual component  (Only available for Bayesian)
            coef = None # A constant to weight Bernoulli Polynomials. If None, use factorials 
            correction = False # Correct the interaction effects if True as in [Brian, 2008]
            
            # We can actually set the weights/priors from SS estimator which is known as the adaptive model 
            (trace1, trace2) = get_adaptive_scale_from_covariance_trace(X_train, poly_order, model_order=2)
            a1 = trace1.mean()/N
            a2 = trace2.mean()/N
        
            

            
            if adaptive == 'Adaptive': 
                # Charger résultats SS
                outfile = os.path.abspath(
                    os.path.join(os.path.dirname(__file__),
                                f"./outputs/{dataset_name}/ss/NonAdaptive/prediction_components_{fold}.npz")
                            )
                npzfile = np.load(outfile)
                ss_prediction_components = npzfile['prediction_components']
                comp_test_main_effect_ss = ss_prediction_components[:D,:]
                comp_test_interaction_effect_ss = ss_prediction_components[D:,]
                
                
                
                
                
                
                
                p = 2 # p norm of component 
                
                main_effect_adaptive_scale = torch.tensor((comp_test_main_effect_ss**p).mean(axis=1))
                main_effect_adaptive_scale = main_effect_adaptive_scale/main_effect_adaptive_scale.mean()
                
                interaction_effect_adaptive_scale = torch.tensor((comp_test_interaction_effect_ss**p).mean(axis=1))
                interaction_effect_adaptive_scale =\
                                interaction_effect_adaptive_scale/interaction_effect_adaptive_scale.mean()
                weight = (main_effect_adaptive_scale, interaction_effect_adaptive_scale)
                
                
                global_main_effect_adaptive_scale = torch.tensor(np.std(np.sum(comp_test_main_effect_ss,
                                                                            axis=0)))*D/a1
                global_interaction_effect_adaptive_scale =\
                            torch.tensor(np.std(np.sum(comp_test_interaction_effect_ss,
                                                axis=0)))*D*(D-1)/2/a2
                
                if model_name == 'lgbayescos_opt' and isotropic_info and mixing=='Dirichlet':
                    def get_number_active_components(val, threshold=0.9):
                        sorted_val, _ = torch.sort(val/val.sum(), descending=True)
                        mask = torch.cumsum(sorted_val, dim=0)<threshold
                        num_active = mask[mask==True].size()[0] + 1
                        a = num_active/val.size()[0]
                        return a*torch.ones_like(val)
                    
                    

                    main_effect_adaptive_scale = get_number_active_components(torch.tensor([(i**p).mean()
                                                                            for i in comp_test_main_effect_ss]))
                        
                    interaction_effect_adaptive_scale = torch.tensor([(i**p).mean() for i in
                                                                        comp_test_interaction_effect_ss])
                    print(main_effect_adaptive_scale.shape, interaction_effect_adaptive_scale.shape)
                    interaction_effect_adaptive_scale = get_number_active_components(interaction_effect_adaptive_scale)
                            
            else:
                
                global_main_effect_adaptive_scale = torch.tensor([1.0])*D/a1
                global_interaction_effect_adaptive_scale = torch.tensor([1.0])*D*(D-1)/2/a2
                if mixing == 'Dirichlet':
                    main_effect_adaptive_scale = 0.6*torch.ones(D)
                    interaction_effect_adaptive_scale = 0.4*torch.ones(D * (D - 1) // 2)
                else:
                    main_effect_adaptive_scale =  1/D*torch.ones(D)
                    interaction_effect_adaptive_scale = 1/(D * (D - 1)/ 2)*torch.ones(D * (D - 1) // 2)
                weight = None
            
            if model_name=='ss':
                L = 10 # number of grid points
                tab_alpha = 10. ** torch.linspace(-7, 1, L) 
                estimator = ss.SS(ndims=D, weight=weight, model_order=model_order,
                                poly_order=poly_order)
                kwargs = {'GCV':False, 'alpha':tab_alpha, 'folds':4, 'random_state':123456}
                
            elif model_name=='cosso': 
                L = 10 # number of grid points
                tab_M = 10. ** torch.linspace(-7, 1, L) 
                try:
                    # Get alpha_star of SS method if it exists
                    outfile = os.path.abspath(
                        os.path.join(os.path.dirname(__file__),
                                    f"./outputs/{dataset_name}/ss/NonAdaptive/alpha_star_{fold}.npz")
                                )
                    npzfile = np.load(outfile)  
                    alpha_star = npzfile['alpha_star']
                except:
                    alpha_star = 1e-6 

                    
                estimator = cosso.COSSO(ndims=D, weight=weight,
                                            model_order=model_order, poly_order=poly_order)
                kwargs = {'GCV':False, 'alpha':alpha_star, 'M':tab_M, 'folds':4,
                                'solver':'lasso', 'tol':1e-3, 'max_iter':100, 'random_state':123456}
                
            else:
                
                gpmodel = model.SobolevGPModel(X_train, y_train, poly_order=poly_order, 
                                        model_order=model_order, residual=residual, coef=coef,
                                        correction=correction)
                    

                constraints_dict = dict()
                priors_dict = dict()
                
                
                
                # Global scales
                global_scale_constraint = (gpytorch.constraints.Positive(),
                                        gpytorch.constraints.Positive())
                global_scale_prior =  (prior.HalfCauchyPrior(scale=global_main_effect_adaptive_scale),
                                    prior.HalfCauchyPrior(scale=global_interaction_effect_adaptive_scale))
                
                # Local scales
                if mixing=='Exp':
                    main_effect_SparsePrior = prior.LaplaceMixingPrior(scale=main_effect_adaptive_scale)
                    interaction_effect_SparsePrior = prior.LaplaceMixingPrior(scale=interaction_effect_adaptive_scale)
                elif mixing=='Horshoe':
                    main_effect_SparsePrior = prior.HoreshoeMixingPrior(scale=main_effect_adaptive_scale)
                    interaction_effect_SparsePrior = prior.HoreshoeMixingPrior(scale=interaction_effect_adaptive_scale)
                elif mixing=='Dirichlet':
                    main_effect_SparsePrior = prior.DirichletMixingPrior(scale=main_effect_adaptive_scale)
                    interaction_effect_SparsePrior = prior.DirichletMixingPrior(scale=interaction_effect_adaptive_scale)
                local_scale_constraint =  (gpytorch.constraints.Positive(), gpytorch.constraints.Positive())
                local_scale_prior =  (main_effect_SparsePrior, interaction_effect_SparsePrior)
                
                # Mean 

                mean_prior = gpytorch.priors.NormalPrior(0, 1.0) 
                mean_constraint = None # No constraint in mean component

                # Noise variance 
                noise_constraint = gpytorch.constraints.Positive()
                noise_prior = prior.HalfCauchyPrior(noise_scale) # Vague prior

                # Add to dict
                constraints_dict['global'] = global_scale_constraint
                priors_dict['global'] = global_scale_prior
                constraints_dict['local'] = local_scale_constraint
                priors_dict['local'] = local_scale_prior
                constraints_dict['mean'] = mean_constraint
                priors_dict['mean'] = mean_prior
                constraints_dict['noise'] = noise_constraint
                priors_dict['noise'] = noise_prior
                
                
                # Init strategy 
                if init_strategy == 'prior':
                    pass
                else:
                    init_noise =  torch.tensor([1.0])
                    init_mean = torch.tensor([0.0])
                    init_local_main = main_effect_adaptive_scale
                    init_global_main = global_main_effect_adaptive_scale/2
                    init_local_interaction = interaction_effect_adaptive_scale
                    init_global_interaction = global_interaction_effect_adaptive_scale/2

                    if model_order == 1:
                        init_strategy = {'noise':init_noise, 'local main':init_local_main, 'global main':init_global_main, 'mean':init_mean}
                    elif model_order == 2: 
                        init_strategy = {'noise':init_noise, 'local main':init_local_main, 'global main':init_global_main, 'mean':init_mean, 
                                        'local interaction':init_local_interaction, 'global interaction':init_global_interaction}

                estimator = lgbayescos.LGBayesCOS(model=gpmodel, 
                                                constraints_dict=constraints_dict,
                                                priors_dict=priors_dict, 
                                                sampler='HMC', 
                                                # Now define the kwargs of the sampler
                                                num_samples=num_samples,
                                                warmup_steps=warmup_steps, 
                                                step_size =1e-8,
                                                num_steps=5,
                                                adapt_step_size=True,
                                                adapt_mass_matrix=False,
                                                full_mass=False,
                                                target_accept_prob=0.7,
                                                num_chains=1,
                                                init_strategy=init_strategy) 
                kwargs = {}
            
            # Train the model
            

            print(f"\nFold {fold} : Training...", end=" ")
            
            gpmodel = estimator.model
            
            gpmodel.train()
            gpmodel.likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.1) 
            
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(gpmodel.likelihood, gpmodel)
            training_iter = 3_000
            
            for iter in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = gpmodel(X_train)
                # Calc loss and backprop gradients
                loss = -mll(output, y_train)
                loss.backward()
                if iter%500 == 1: 
                    print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
                        iter + 1, training_iter, loss.item(),
                        gpmodel.likelihood.noise.item()
                    ))
                optimizer.step()
            print("ended")
            
            gpmodel.eval()
            gpmodel_batch = gpmodel.to_batch(N=1)
            param_dict = dict()
            for param_name, param in gpmodel.named_parameters(): 
                param_dict[param_name] = param.expand(1, *param.shape)
            gpmodel_batch.initialize(**param_dict)
            gpmodel = gpmodel_batch
            gpmodel.eval()
            train_inputs = gpmodel.train_inputs[0]
            train_inputs_pred = list(gpmodel.train_inputs) if gpmodel.train_inputs is not None else []
            train_prior_dist = gpmodel.forward(train_inputs)
            gpmodel.prediction_strategy = gpytorch.models.exact_prediction_strategies.DefaultPredictionStrategy(
                        train_inputs=train_inputs_pred,
                        train_prior_dist=train_prior_dist,
                        train_labels=gpmodel.train_targets,
                        likelihood=gpmodel.likelihood)#,
            
            
            
            estimator.model_batch = gpmodel
            estimator.num_samples_inference = 1
            estimator.model_batch.eval()


                    
            
            
            
         
        
            
            
            # Shrinking behaviour only for component selecton methods 
            
            index_interaction = None
            interaction_order = [i for i in range(D*(D-1)//2)]
            num_interaction = D*(D-1)//2
            
            
                
            # Test performance.
            
            # Set test data to the appropriate range
            X_test = torch.tensor(X_test, dtype=torch.float32)
            X_test = (X_test-X_min)/(X_max-X_min)
            X_test = torch.clamp(X_test, min=0, max=1)
            
            y_test = y_test.squeeze()
            
            # Prediction 
            
            
            
            # Predict the outcome function 
            
            print(f"Fold {fold} : Outcome prediction...", end=" ")
            
            f_pred = estimator.predict(X_test, kind='outcome')
            
            print("ended")
            
            # For Bayesian model, prediction returns a distribution, so take the mean and the variance 
                
            if model_name=='lgbayescos_opt':
                
               
                
                with gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(100):
                    # Samples for variances 
                    f_pred_var =  f_pred.variance.detach().numpy() 
                    f_pred_var = correct_variance_y(f_pred_var)

                # Samples for mean
                f_pred_mean = f_pred.mean.detach().numpy()
                f_pred_mean = inverse_scaler_y(f_pred_mean)

                # Point estimates for mean and variance 
                f_pred_out, f_pred_var_out = samples2estimate(f_pred_mean, f_pred_var)
            else:
                f_pred_out = inverse_scaler_y(f_pred.detach().numpy())
            
            #print("ended")
            
            

            #  Global metrics computed on the estimated outcome 
            
            mae = np.abs(f_pred_out- y_test).mean()
            
            rmse = np.sqrt(((f_pred_out - y_test) ** 2).mean())
            tss = (
                (y_test - y_test.mean() * np.ones(y_test.shape)) ** 2
            ).mean()
            r2 = 1 - rmse**2 / tss
            
            # Coverage 
            if model_name=='lgbayescos_opt':
                mask_inf1 = f_pred_out +  2*np.sqrt(f_pred_var_out) >= y_test
                mask_sup1 = f_pred_out -  2*np.sqrt(f_pred_var_out) <= y_test
                coverage1 = np.sum(mask_inf1*mask_sup1)/mask_inf1.shape[0]
                
                coverage2 = coverage1
                print('Coverage', (coverage1, coverage2))
            else:
                coverage1 = 0
                coverage2 = 0
            
            RMSE.append(rmse)
            MAE.append(mae)
            R2.append(r2)
            COVERAGE1.append(coverage1)
            COVERAGE2.append(coverage2)
            _log.info('Fold %0.0f, rmse=%0.4f, mae=%0.4f, r2=%0.4f, coverage mean +/- 2.std =%0.4f, coverage equal tailed 0.9 interval = %0.4f '%(fold,\
                                                                                    rmse, mae, r2, coverage1, coverage2))
            
         
            # Save optimal parameters for ss and cosso 
            
            if model_name=='ss':
                alpha_star, ss_loss = out
                
                np.savez(output_prefix + "/alpha_star_%d" % fold, alpha_star=alpha_star)
                
                _log.info(f"Best results for alpha={alpha_star} giving error={ss_loss.min()}")
                
            elif model_name=='cosso':
                M_star, cosso_loss = out
                np.savez(output_prefix + "/M_star_%d" % fold, M_star=M_star)
                _log.info(f"Best results for M={M_star} giving error={cosso_loss.min()}")
            
            
            # Clear variable 
            del  f_pred
            
            
            # Predict the main components
            

            
            print(f"Fold {fold} : Main components prediction...", end=" ")
            
            if model_name=='lgbayescos_opt':
                    
                    with gpytorch.settings.skip_posterior_variances(state=True):
                        main_pred = estimator.predict(X_test, kind='main')
            else: 
                main_pred = estimator.predict(X_test, kind='main')
            
            
        

            if model_name=='lgbayescos_opt':
                main_pred_out = {k: v.mean.detach().numpy().mean(axis=0) for (k, v) in main_pred.items()}
            else:
                main_pred_out = {k: v.detach().numpy() for (k, v) in main_pred.items()}
            
            print("ended")
            del  main_pred
            
            # Predict the interaction components 
            
            print(f"Fold {fold} : Interaction components prediction...", end=" ")
            
            if model_name=='lgbayescos_opt':

                    with gpytorch.settings.skip_posterior_variances(state=True):
                        interaction_pred = estimator.predict(X_test, kind='interaction', component_index=index_interaction)
            else: 
                interaction_pred = estimator.predict(X_test, kind='interaction', component_index=index_interaction)
            
           
            
            
            
            if model_name=='lgbayescos_opt':
                interaction_pred_out = {k: v.mean.detach().numpy().mean(axis=0) for (k, v) in interaction_pred.items()}
            else:
                interaction_pred_out = {k: v.detach().numpy() for (k, v) in interaction_pred.items()}
            
            print("ended\n")
            
            del  interaction_pred
            
            
            # Calculate variances (energies) of components
            main_prediction_components = np.array([i for i in main_pred_out.values()])
            interaction_prediction_components = np.array([i for i in interaction_pred_out.values()])
            prediction_components = np.concatenate((main_prediction_components, interaction_prediction_components), axis=0)
            components_variance = np.mean(prediction_components**2, axis=1)-np.mean(prediction_components, axis=1)**2

            # Sort components by variance 
            cumulative_variance, rmse_component = [], []
            order = np.argsort(components_variance)[::-1]
            y_pred_component = np.zeros_like(y_test)
            y_pred_mean = estimator.model_batch.mean_module(X_test).detach().numpy()
            if model_name == 'lgbayescos_opt':
                y_pred_mean = y_pred_mean.mean()
            y_pred_component = y_pred_component + y_pred_mean

            for n in order:
                # add predictions of the terms one by one ranked by variance
                y_pred_component +=prediction_components[n, :] 
                unscaled_y_pred_component = inverse_scaler_y(y_pred_component)
                error_component = np.sqrt(
                    (( unscaled_y_pred_component- y_test) ** 2).mean()
                )
                rmse_component.append(error_component)
                cumulative_variance.append(components_variance[n])
                
            cumulative_variance = np.cumsum(cumulative_variance)
            total_variance = np.sum(components_variance)
            cumulative_variance = cumulative_variance/total_variance
            
            idx = cumulative_variance[cumulative_variance<0.99].size
            number_active_components = idx
            
            Number_active_components.append(number_active_components)
            _log.info(f"Number of active components (giving 99% of the energy) is {number_active_components:.0f} out of {D*(D+1)/2:.0f}\n")
                    
            
            
            # Generate plots  
        
            
            # Plot 10 most informative components  
            plt.figure(figsize=(8, 4))
            main_labels = [i[:4]+'.' if len(i)>4 else i[:4] for i in covariate_names[dataset_name]] # les 4 premiers lettres du nom
            inter_labels = ['['+main_labels[i]+','+main_labels[j]+']' for i in np.arange(0, D - 1)
                            for j in np.arange(i + 1, D)]
            inter_labels = [inter_labels[i] for i in np.sort(interaction_order)]
            labels = main_labels + inter_labels
            
            best_components_labels = [labels[i] for i in order[:10]]
            best_components_variances = [components_variance[i]/total_variance for i in order[:10]]
            #plt.errorbar(best_components_labels, best_components_variances, yerr=0, linewidth=3)
            fig, ax = plt.subplots()
            ax.plot(np.arange(len(best_components_labels)), best_components_variances, "b", linewidth=3)
            ax.plot(np.arange(len(best_components_labels)), best_components_variances, "*b", linewidth=3)
            
            plt.xticks(np.arange(len(best_components_labels)), best_components_labels, fontsize=16, rotation=90)
            plt.grid()
            
            plt.xlabel('Active components')
            plt.ylabel('Variance')
            plt.title(dataset_name)
            plt.tight_layout()
            plt.savefig(output_prefix + "/Active_components_variances_%d.png" % fold)

            
            
            # RMSE=f(Number of components added)
            plt.figure(figsize=(8, 4))
            fig, ax1 = plt.subplots()

            ax2 = ax1.twinx()
            ax1.plot(np.arange(len(order)), rmse_component, "r", linewidth=3)
            ax2.plot(np.arange(len(order)), cumulative_variance, "-.k", linewidth=3)

            ax1.set_xlabel("Number of added components")
            ax1.set_ylabel("RMSE", color="r")
            ax2.set_ylabel("Cumulative variance", color="k")
        

            plt.title(dataset_name)
            plt.grid()
            plt.tight_layout()
            plt.savefig(output_prefix + "/cumulative_variance_%d.png" % fold)
            
            # save learned model
            trained_model_filename = f"{output_prefix}/Trained_model_{fold}.pkl"
            torch.save(estimator.state_dict(), trained_model_filename)

            
            # save model performance metrics
            np.savez(
                output_prefix + "/metrics_%d" % fold,
                number_active_components=number_active_components,
                coverage1=coverage1,
                coverage2=coverage2,
                order=order,
                rmse=rmse,
                mae=mae,
                r2=r2)
            
            # Save prediction components : only for SS to use for adaptive methods
            if model_name=='ss':
                np.savez(output_prefix + "/prediction_components_%d" % fold, 
                        prediction_components=prediction_components)
            
            np.savez(output_prefix + "/constant_component_%d" % fold, 
                        constant_component=y_pred_mean)
            
            fold += 1
            _log.info(f"\nResults over {fold} folds : rmse={np.mean(RMSE):.4f}+/-{np.std(RMSE):.4f}, mae={np.mean(MAE):.4f}+/-{np.std(MAE):.4f}, r2={np.mean(R2):.4f}+/-{np.std(R2):.4f}, Coverage1={np.mean(COVERAGE1):.4f}+/-{np.std(COVERAGE1):.4f}, Coverage2={np.mean(COVERAGE2):.4f}+/-{np.std(COVERAGE2):.4f}")
        except:
            _log.info(f"\nSome covergence issues for fold{fold}")
            fold +=1

        
    _log.info(f"\nResults : rmse={np.mean(RMSE):.4f}+/-{np.std(RMSE):.4f}, mae={np.mean(MAE):.4f}+/-{np.std(MAE):.4f}, r2={np.mean(R2):.4f}+/-{np.std(R2):.4f}, Coverage1={np.mean(COVERAGE1):.4f}+/-{np.std(COVERAGE1):.4f}, Coverage2={np.mean(COVERAGE2):.4f}+/-{np.std(COVERAGE2):.4f}, active={np.mean(Number_active_components):.4f}+/-{np.std(Number_active_components):.4f}")


if __name__ == "__main__":
    main()