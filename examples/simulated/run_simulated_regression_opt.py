
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
from bayesanova.models.utils import first_order_InteractionEffect_to_index

import logging
_log = logging.getLogger(__name__)


normalize = True # Normalize y 

matplotlib.rcParams.update({"font.size": 20})

model_order = 1 # Model order 
if model_order == 1: 
    data_path_prefix = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../data/simulated0")
    ) + '/'
else: 
    data_path_prefix = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../data/simulated1")
    ) + '/'

prefix = 'simulated0' if model_order == 1 else 'simulated1'

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
from glob import glob

num_samples = 10_000  # 2_000
warmup_steps = 10_000  # 3_000
num_samples_inf_default = 1000 # 50
subsample = num_samples // num_samples_inf_default # 10
noise_scale = 1





def main():
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
        "--model_name",  default='lgbayescos', help="model name"
    )
    
    args_parser.add_argument(
        "--mixing",  default='Exp', help="mixing density for lgbacos"
    )
    
    args_parser.add_argument(
        "--init_strategy",  default='prior', help="initialize strategy"
    )
    
    

    args, unknown = args_parser.parse_known_args()
    dataset_name, k, num_samples_inf, model_name, mixing, init_strategy =\
        args.dataset_name, args.k, args.num_samples_inf, args.model_name, args.mixing, args.init_strategy

    # save results to outputs folder
    
   
    output_prefix = os.path.abspath(
         os.path.join(os.path.dirname(__file__), f"./outputs/{prefix}/{dataset_name[len(data_path_prefix):]}/{model_name}")
    )
    
    
    if not(model_name=='ss') and not(model_name=='cosso'): 
        model_name = 'lgbayescos_opt'
    
    if model_name=='lgbayescos_opt':
        output_prefix = os.path.join(os.path.dirname(__file__), f"./outputs/{prefix}/{dataset_name[len(data_path_prefix):]}/{model_name}/{mixing}")
        

    if not os.path.exists(output_prefix):
        os.makedirs(output_prefix)
        
    logging.basicConfig(filename=f'{output_prefix}/log.log', level=logging.INFO)
    

    np.random.seed(seed=18)
    torch.manual_seed(18)
    
    
    num_samples_inference = num_samples_inf # Number of samples used for inference
    
    RMSE = []
    MAE = []
    R2 = []
    COVERAGE1 = []
    COVERAGE2 = []
    Number_active_components = []
    TP_RATE = []
    TN_RATE = []
    FP_RATE = []
    FN_RATE = []
    
  
     
     # All .mat file in the folder 
    filenames = np.sort(glob(dataset_name+'/*.mat', recursive=True))   
    
     
    for id, filename in enumerate(filenames[:10]): 
        d = io.loadmat(filename) 
        X_train, y_train, f_train, active_train = d['train_x'], d['train_y'], d['train_f'], d['train_active']
        X_test, y_test, f_test, active_test = d['test_x'], d['test_y'], d['test_f'], d['test_active']
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).squeeze()
        
        #print(X_train.shape)
        
        if id == 0: 
            N, D = X_train.shape
            _log.info(f"Method {model_name} running...\n")
            _log.info(f"Dataset of dimension  = {(N, D)}")
            _log.info(f"Number of components to be estimated={D*(D+1)//2} corresponding to {D} main effects and {D*(D-1)//2} interaction effects\n")
        
        
        
        true_main_active_components = [0, 1, 2, 3] 
        if model_order == 2:
            k = [[0, 1], [0, 2], [2, 4]]
            true_interaction_active_components = [first_order_InteractionEffect_to_index(D, list(j)) for j in k]
            
            true_active_components = true_main_active_components + [D+i for i in true_interaction_active_components]
        else: 
            true_active_components = true_main_active_components

        
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
        # Set the model parameters 
        poly_order = 1 # Polynomial order
        residual = False # If true include residual component  (Only available for Bayesian)
        coef = None # A constant to weight Bernoulli Polynomials. If None, use factorials 
        correction = False # Correct the interaction effects if True as in [Brian, 2008]
            
        # We can actually set the weights/priors from SS estimator which is known as the adaptive model 
        (trace1, trace2) = get_adaptive_scale_from_covariance_trace(X_train, poly_order, model_order=2)
        a1 = trace1.mean()/N
        a2 = trace2.mean()/N
        
        # if model_order == 1: 
        #     global_main_effect_adaptive_scale = torch.tensor([10.0])#*D/a1
        #     global_interaction_effect_adaptive_scale = torch.tensor([10.0])*D*(D-1)/2/a2
        # else:
        #     global_main_effect_adaptive_scale = torch.tensor([10.0])
        #     #global_main_effect_adaptive_scale = torch.tensor([1.0])*D/a1
        #     #global_interaction_effect_adaptive_scale = torch.tensor([0.005])*D*(D-1)/2/a2
        #     global_interaction_effect_adaptive_scale = torch.tensor([50.0])
        # if mixing == 'Dirichlet':
        #         global_main_effect_adaptive_scale = torch.tensor([10.0])*D/a1
        #         global_interaction_effect_adaptive_scale = torch.tensor([10.0])*D*(D-1)/a2
        #         main_effect_adaptive_scale = 0.5*torch.ones(D)
        #         interaction_effect_adaptive_scale = 0.4*torch.ones(D * (D - 1) // 2)
        # else:
        #         main_effect_adaptive_scale =  torch.ones(D)
        #         interaction_effect_adaptive_scale = 1/(D * (D - 1)/ 2)*torch.ones(D * (D - 1) // 2)
        #         #interaction_effect_adaptive_scale = torch.ones(D * (D - 1) // 2)
        #weight = None
        
        global_main_effect_adaptive_scale = torch.tensor([10.0])*D/a1
        global_interaction_effect_adaptive_scale = torch.tensor([10.0])*D*(D-1)/2/a2
        if mixing == 'Dirichlet':
                    main_effect_adaptive_scale = 0.6*torch.ones(D)
                    interaction_effect_adaptive_scale = 0.2*torch.ones(D * (D - 1) // 2)
        else:
                    main_effect_adaptive_scale =  1/D*torch.ones(D)
                    interaction_effect_adaptive_scale = 1/(D * (D - 1)/ 2)*torch.ones(D * (D - 1) // 2)
        weight = None
        
        
        # Define models 
        
        if model_name=='ss':
                L = 100 # number of grid points
                #tab_alpha = 10. ** torch.linspace(-8, 1, L) 
                tab_alpha = 10. ** torch.linspace(-10, 5, L) 
                estimator = ss.SS(ndims=D, weight=weight, model_order=model_order,
                                poly_order=poly_order)
                kwargs = {'GCV':False, 'alpha':tab_alpha, 'folds':4, 'random_state':123456}
                
        elif model_name=='cosso': 
                L = 100 # number of grid points
                #tab_M = 10. ** torch.linspace(-2, 5, L) 
                tab_M = 10. ** torch.linspace(-10, 5, L) 
                try:
                    # Get alpha_star of SS method if it exists
                    print('OK')
                    outfile = os.path.abspath(
                        os.path.join(os.path.dirname(__file__),
                                    f"./outputs/{dataset_name}/ss/NonAdaptive/alpha_star_{id}.npz")
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
                elif mixing=='Student':
                    main_effect_SparsePrior = prior.StudentMixingPrior(scale=main_effect_adaptive_scale, freedom=1.0)
                    interaction_effect_SparsePrior = prior.StudentMixingPrior(scale=interaction_effect_adaptive_scale, freedom=1.0)
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
                                                step_size =1e-6,
                                                num_steps=5,
                                                adapt_step_size=True,
                                                adapt_mass_matrix=False,
                                                full_mass=False,
                                                target_accept_prob=0.7,
                                                num_chains=1,
                                                init_strategy=init_strategy) 
                kwargs = {}
                
        
        
        # Train the model with optimization 
        print(f"\Dataset {id} : Training...", end=" ")
        
        gpmodel = estimator.model
        
        gpmodel.train()
        gpmodel.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.1) 
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(gpmodel.likelihood, gpmodel)
        training_iter = 5_000
        
        for iter in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = gpmodel(X_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train)
            loss.backward()
            if iter%1000 == 1: 
                print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
                    iter + 1, training_iter, loss.item(),
                    gpmodel.likelihood.noise.item()
                ))
            optimizer.step()
            
            
        print("ended")
        
        
        # Shrinking behaviour only for component selecton methods 
            
        index_interaction = None
        interaction_order = [i for i in range(D*(D-1)//2)]
        num_interaction = D*(D-1)//2
            
        
        
        # Prediction 
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = y_test.squeeze()
            

                
        # Predict the outcome function 
            
        print(f"Dataset {id} : Latent prediction...", end=" ")
        gpmodel.eval()
        f_pred = gpmodel(X_test)
        
        
            
        
        
        # For Bayesian model, prediction returns a distribution, so take the mean and the variance 
                
        if model_name=='lgbayescos_opt':
                
                #lower, upper = credible_interval_equal_tailed(f_pred, epsilon=0.025)
                #lower = inverse_scaler_y(lower)
                #upper = inverse_scaler_y(upper)
                
                with gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(10):
                    # Samples for variances 
                    f_pred_var =  f_pred.variance.detach().numpy() 
                    f_pred_var = correct_variance_y(f_pred_var)

                # Mean
                f_pred_out = f_pred.mean.detach().numpy()
                f_pred_out = inverse_scaler_y(f_pred_out)

                # Point estimates for mean and variance 
                f_pred_var_out = f_pred_var
        else:
                f_pred_out = inverse_scaler_y(f_pred.detach().numpy())
        
        print("ended")    
        
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
                #mask_inf2 = upper.detach().numpy() >= y_test
                #mask_sup2 = lower.detach().numpy() <= y_test
                #coverage2 = np.sum(mask_inf2*mask_sup2)/mask_inf2.shape[0]
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
        _log.info('Dataset %0.0f, rmse=%0.4f, mae=%0.4f, r2=%0.4f, coverage mean +/- 2.std =%0.4f, coverage equal tailed 0.9 interval = %0.4f '%(id,\
                                                                                    rmse, mae, r2, coverage1, coverage2))
    
            
            
            
        # Clear variable 
        del  f_pred
        
        
        
        # Predict the main components
        if model_name=='lgbayescos_opt':
                # gpmodel = gpmodel.to_batch(N=1)
                # train_inputs = gpmodel.train_inputs[0]
                # train_inputs_pred = list(gpmodel.train_inputs) if gpmodel.train_inputs is not None else []
                # train_prior_dist = gpmodel.forward(train_inputs)
                # gpmodel.prediction_strategy = gpytorch.models.exact_prediction_strategies.DefaultPredictionStrategy(
                #         train_inputs=train_inputs_pred,
                #         train_prior_dist=train_prior_dist,
                #         train_labels=gpmodel.train_targets,
                #         likelihood=gpmodel.likelihood)#,
                
                estimator.model_batch = gpmodel
                
                
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
                
            
        print(f"Dataset {id} : Main components prediction...", end=" ")
            
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
        
        
        if model_order == 2:
        # Predict the interaction components 
            
            print(f"Dataset {id} : Interaction components prediction...", end=" ")
                
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
        if model_order == 2:
            interaction_prediction_components = np.array([i for i in interaction_pred_out.values()])
            prediction_components = np.concatenate((main_prediction_components, interaction_prediction_components), axis=0)
        else: 
            prediction_components = main_prediction_components
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
        number_active_components = idx + 1
            
        Number_active_components.append(number_active_components)
        J = D*(D+1)/2 if model_order == 2 else D
        _log.info(f"Number of active components (giving 99% of the energy) is {number_active_components:.0f} out of {J:.0f}\n")
        
        
        # True and False positive 
        active_component_id = order[:number_active_components]
        true_selected = float(len(set(active_component_id) & set(true_active_components)))
        false_selected = number_active_components - true_selected
        print(true_selected, false_selected, active_component_id, true_active_components)
        
        non_active_component_id = order[number_active_components:]
        Number_non_active_components = len(non_active_component_id)
        true_non_active_componenst = list(set(order)-set(true_active_components))
        true_unselected = float(len(set(true_non_active_componenst)& set(non_active_component_id)))
        false_unselected = Number_non_active_components-true_unselected
        print(true_unselected, false_unselected)
        
        
        TP_RATE.append(true_selected)
        TN_RATE.append(true_unselected)
        FP_RATE.append(false_selected)
        FN_RATE.append(false_unselected)
        
        
        _log.info(f"TP: {TP_RATE[id]:.5f}, TN : {TN_RATE[id]:.5f}, FP : {FP_RATE[id]:.5f}, FN : {FN_RATE[id]:.5f}\n")
        
        
        # save model performance metrics
        np.savez(
            output_prefix + "/metrics_%d" % id,
            number_active_components=number_active_components,
            coverage1=coverage1,
            coverage2=coverage2,
            tp=TP_RATE[id], 
            tn=TN_RATE[id],
            fp=FP_RATE[id], 
            fn=FN_RATE[id],
            order=order,
            rmse=rmse,
            mae=mae,
            r2=r2)
        
        
        # Save prediction components : only for SS to use for adaptive methods
        if model_name=='ss':
                np.savez(output_prefix + "/prediction_components_%d" % id, 
                        prediction_components=prediction_components)
            
        np.savez(output_prefix + "/constant_component_%d" % id, 
                        constant_component=y_pred_mean)
            
        _log.info(f"\nResults over {id} datasets : rmse={np.mean(RMSE):.4f}+/-{np.std(RMSE):.4f}, mae={np.mean(MAE):.4f}+/-{np.std(MAE):.4f},\
                  r2={np.mean(R2):.4f}+/-{np.std(R2):.4f}, Coverage1={np.mean(COVERAGE1):.4f}+/-{np.std(COVERAGE1):.4f},\
                      Coverage2={np.mean(COVERAGE2):.4f}+/-{np.std(COVERAGE2):.4f},\
                      TP={np.mean(TP_RATE):.4f}+/-{np.std(TP_RATE):.4f}, TN={np.mean(TN_RATE):.4f}+/-{np.std(TN_RATE):.4f}, \
                          FP={np.mean(FP_RATE):.4f}+/-{np.std(FP_RATE):.4f}, FN={np.mean(FN_RATE):.4f}+/-{np.std(FN_RATE):.4f}")
        
  

        
        
if __name__ == "__main__":
    main()
        


    

