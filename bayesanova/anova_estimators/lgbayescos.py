import gpytorch
import torch
import copy
from bayesanova.anova_estimators.utils import (
    get_adaptive_scale_from_covariance_trace,
)
from bayesanova.samplers.mcmc import HMC, NUTS
from bayesanova.samplers.stein import GPSVGD,  GPSliceSVGD



def add_constraints_priors(gpmodel, constraints_dict, priors_dict):
    """This function adds constraints and priors from two dictionries to a GP model

    Args:
        gpmodel (models.model): Gaussian process model
        constraints_dict (dict): constraints
        priors_dict (dict): priors

    Returns:
        models.model:Gaussian process model updated
    """
    if constraints_dict['noise'] is not None:
        gpmodel.set_noise_constraint(constraints_dict['noise'])
    gpmodel.set_noise_prior(priors_dict['noise'])
        
    if constraints_dict['mean'] is not None:
        gpmodel.set_mean_constraint(constraints_dict['mean'])
    gpmodel.set_mean_prior(priors_dict['mean'])
        
        
    if constraints_dict['global'] is not None:
        gpmodel.set_global_scale_constraint(constraints_dict['global'])
    gpmodel.set_global_scale_prior(priors_dict['global'])
        
    if constraints_dict['local'] is not None:
        gpmodel.set_local_scale_constraint(constraints_dict['local'])

    gpmodel.set_local_scale_prior(priors_dict['local'])

    return gpmodel
    

class LGBayesCOS(torch.nn.Module):
    """This a bayesian local-global shrinking GP model for solving anova models with sampling.

    Args:
        model (models.model): Gaussian process model
        sampler (str): name of sampling algorithm
        constraints (dict of gpytorch.constraints): constraints for parameters
        priors (dict of models.prior): priors for parameters 
        kwargs (named arguments): arguments for sampler 
    """
    def __init__(self, model, constraints_dict, priors_dict, sampler, **kwargs):
        super(LGBayesCOS, self).__init__()
        
        self.sampler = sampler
        self.base_model = add_constraints_priors(model, constraints_dict, priors_dict)
        self.model = copy.deepcopy(self.base_model)
        
        if sampler=='HMC':
            self.estimator = HMC(self.model, **kwargs)
        elif sampler=='NUTS':
            self.estimator = NUTS(self.model, **kwargs)
        elif sampler =='SVGD':
            # For Stein methods, we need model in a batch mode
            num_particles = kwargs.get('num_particles', 2)
            batch_model = model.to_batch(num_particles)
            self.model = add_constraints_priors(batch_model, constraints_dict, priors_dict)
            self.estimator = GPSVGD(self.model, **kwargs)
        #elif sampler =='SliceSVGD':
        #    num_particles = kwargs.get('num_particles', 2)
        #    self.model = add_constraints_priors(model.to_batch(num_particles), 
        #                                        constraints_dict, priors_dict)
        #    self.estimator = GPSliceSVGD(self.gmodel, **kwargs)
        else:
            raise ValueError('sampler is not implemented yet')

        self.model_batch = None 
        self.__names = {'likelihood.noise_prior':'noise variance', 
                        'mean_module.mean_prior':'constant mean',
                        '_SobolevGPModel__MainEffect_covar_module.outputscale_prior':'main effect global scale',
                        '_SobolevGPModel__MainEffect_covar_module.base_kernel.outputscale_prior':'main effect local scale', 
                        '_SobolevGPModel__InteractionEffect_covar_module.outputscale_prior':'interaction effect global scale',
                        '_SobolevGPModel__InteractionEffect_covar_module.base_kernel.outputscale_prior':'interaction effect local scale',
                        '_SobolevGPModel__ResidualEffect_covar_module.outputscale_prior':'residual effect global scale',
                        'latent_prior':'latent'}
        
    
    def train(self, train_x, train_y):
        """This method train the LGBayesCOS given training data 

        Args:
            train_x (tensor): training inputs
            train_y (tensor): training outputs
        """
        self.train_x = train_x
        self.train_y = train_y
        self.model_batch = None
        self.estimator.run(train_x, train_y)
        
        return None
    
    def get_samples(self):
        """Get obtained samples from the trained sampler.

        Returns:
            dict: a dictionnary containing samples of the GP model
        """
        return self.estimator.get_samples()
    
    def get_named_samples(self, names=None):
        """Get obtained samples from the trained sampler and rename them for readibility. 

        Returns:
            dict: a dictionnary containing samples of the GP model
        """
        if names is None:
            names = self.__names
        
        raw_samples = self.estimator.get_samples()
        param_names = [names[i] for i in raw_samples.keys()]

        return dict(zip(param_names, list(raw_samples.values())))
    
    @property
    def selection_parameter(self):
        """The selection parameter  with an equivalent interpretation of that for COSSO.  

        Returns:
            tuple: containing the selection parameters
        """
        adaptive_scale_main_effect, _ =\
        get_adaptive_scale_from_covariance_trace(self.train_x, self.model.poly_order,
                                                model_order=self.model.model_order,
                                                c=None, correction=False)
        N = self.train_x.shape[-1]
        samples = self.get_named_samples()
        
        
        if self.model.model_order==1:
            # Main effects bars
            main_weight =\
                samples['main effect local scale']*samples['main effect global scale']*adaptive_scale_main_effect/N
            inter_weight = None
        else:
            adaptive_scale_main_effect, adaptive_scale_interaction_effect =\
                    get_adaptive_scale_from_covariance_trace(self.train_x, self.model.poly_order, 
                                                            model_order=self.model.model_order,
                                                            c=None, correction=False)
            # Main effects bars
            main_weight =\
                samples['main effect local scale']*samples['main effect global scale']*adaptive_scale_main_effect/N
            inter_weight =\
                samples['interaction effect local scale']*samples['interaction effect global scale']*adaptive_scale_interaction_effect/N
        
        return (main_weight.detach().numpy(), inter_weight.detach().numpy())
    
    def train_summary(self):
        """Display training summary
        """
        return self.estimator.summary(names=self.__names)
    
    def predict_on(self, num_samples=1, subsample=1):
        """This methods computes caches to allow fast computations in repetitive prediction tasks

        Args:
            num_samples (int, optional): number  of samples used for prediction. Defaults to 1.

        """
        
        raw_samples = self.estimator.get_samples()
        self.named_samples = self.get_named_samples()
    
        samples_inference = dict()
        for name, s in raw_samples.items():
            samples_inference[name] = s[:num_samples*subsample:subsample]

        self.model_batch = copy.deepcopy(self.base_model)
        samples = copy.deepcopy(samples_inference)
        self.num_samples_inference = samples_inference[name].shape[0] 

        # Precompute cache for prediction
        self.model_batch.predict_on(samples)
        
        
    
    def predict(self, x, kind='latent', component_index=None, **kwargs):
        """This is the prediction step of the model

        Args:
            x (tensor): test data
            kind (str, optional): main, intraction, residual, latent or outcome. Defaults to 'latent'.
            component_index (int or list, optional): target component. Defaults to None.
            **kwargs : diag=True or False for Sparse SparseSobolevGPModel

        Returns:
            _type_: _description_
        """
        if self.model_batch is None:
            raise('Please call predict_on first to compute cache for fast computation')   
        if kind=='main':  
            return self.model_batch.predictive_main_effect_component(x, component_index, **kwargs)
        elif kind=='interaction':
            if self.model.model_order==2:
                return self.model_batch.predictive_interaction_effect_component(x, component_index, **kwargs)
            else:
                ValueError('No interaction component for model_order=1')
        elif kind=='latent':
            return self.model_batch.predictive_latent(x, **kwargs)
        elif kind=='outcome':
            #expanded_x = x.unsqueeze(0).repeat(self.num_samples_inference, 1, 1)
            return self.model_batch.predictive_latent(x, outcome=True, **kwargs)
        elif kind=='residual':
            return self.model_batch.predictive_residual_effect_component(x, **kwargs)
        else:
            ValueError('kind can be one of the followings : main, interaction residual, latent, outcome')
            
    @property
    def constant_mean(self):
        """This gives the estimated constant mean

        Args:
            x (tensor): constant mean
        """
        samples = self.get_named_samples()
        return samples['constant mean']
            
            
                
        
        
        
        
    
    
    



            
        
    
            
        