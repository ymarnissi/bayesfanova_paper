import copy

import pyro
from pyro.infer.mcmc.util import print_summary
from bayesanova.models.model import SobolevGPModel, SparseSobolevGPModel


def marginal_pyro_model(model, x, y):
    """This is a pyro routine with marginalized model
    """
    model.pyro_sample_from_prior()
    output = model(x)
    loss = model.marginal_posterior.pyro_factor(output, y)
    return y


def joint_pyro_model(model, x, y):
    """This is a pyro routine with joint model
    """
    model.pyro_sample_from_prior()
    output = model.latent_forward()
    loss = model.joint_posterior.pyro_factor(output, y)
    return y

def expected_pyro_model(model, x, y):
    """This is a pyro routine with expected model
    """
    model.pyro_sample_from_prior()  
    output = model.conditioning(x, diag = True)
    loss = model.expected_posterior.pyro_factor(output, y)
    return y

def mcmc_init_strategy(kind='uniform'):
    """This fonction defines the initialization strategy of mcmc

    Args:
        kind (str or dict, optional): _description_. Defaults to 'uniform'.

    Raises:
        ValueError: kind can be either str ('uniform', 'prior') or a dictionary 
        {key:value} with key in ["noise", "mean", "global main", "local main",
                                "global interaction", "local interaction"]

    Returns:
        pyro.infer.autoguide.initialization: init strategy
    """
    mcmc_init = None
    if kind=='uniform':
        # This is the default init strategy : uniform sampling following constraints
        mcmc_init = pyro.infer.autoguide.initialization.init_to_uniform
    else: 
        if kind=='prior':
            # This is another init stratgy : sample from the priors (only available is sampling is possible) 
            mcmc_init = pyro.infer.autoguide.initialization.init_to_sample
        else:
            try: 
                # Initialize to the value specified in values
                param_name = {'noise':'likelihood.noise_prior', 
                            'mean':'mean_module.mean_prior',
                        'global main':'_SobolevGPModel__MainEffect_covar_module.outputscale_prior',
                        'local main':'_SobolevGPModel__MainEffect_covar_module.base_kernel.outputscale_prior', 
                        'global interaction':'_SobolevGPModel__InteractionEffect_covar_module.outputscale_prior',
                        'local interaction':'_SobolevGPModel__InteractionEffect_covar_module.base_kernel.outputscale_prior'}
                param_values = kind.values()
                param_name = [param_name[i] for i in kind.keys()] 
                def generate():
                    values = {k: v for (k, v) in zip(param_name, param_values)}
                    return pyro.infer.autoguide.initialization.init_to_value(values=values)
                mcmc_init = pyro.infer.autoguide.initialization.init_to_generated(generate=generate)
            except:
                raise ValueError('kind should be either string or a dict')
    return mcmc_init

    
class HMC(pyro.infer.mcmc.MCMC):
    def __init__(
            self,
            model,
            step_size=1,
            num_steps=5,
            adapt_step_size=True,
            adapt_mass_matrix=True,
            full_mass=False,
            num_samples=1000,
            warmup_steps=1000,
            num_chains=1,
            target_accept_prob=0.7,
            jit_compile=False,
            init_strategy=None,
            min_stepsize=1e-10, 
            max_stepsize=1000.0
):

        if isinstance(model, SobolevGPModel):
            if model.latent_estimate is True:
                pyro_model = lambda x, y: joint_pyro_model(model, x, y)
            else:
                pyro_model = lambda x, y: marginal_pyro_model(model, x, y)
        elif isinstance(model,  SparseSobolevGPModel):
            pyro_model = lambda x, y : expected_pyro_model(model, x, y)
            
        
        if init_strategy is None:
            init_strategy = pyro.infer.autoguide.initialization.init_to_uniform
        else:
            init_strategy = mcmc_init_strategy(init_strategy)
            
            
        #print(min_stepsize, max_stepsize, step_size)
        hmc_kernel = pyro.infer.mcmc.HMC(pyro_model, step_size=step_size, num_steps=num_steps,
                                        adapt_step_size=adapt_step_size,
                                        adapt_mass_matrix=adapt_mass_matrix, full_mass=full_mass,
                                        target_accept_prob=target_accept_prob,
                                        jit_compile=jit_compile,
                                        init_strategy=init_strategy, 
                                        #min_stepsize=min_stepsize, 
                                        #max_stepsize=max_stepsize, 
                                        )
        super(HMC, self).__init__(hmc_kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains)

    def get_samples(self):
        samples = copy.deepcopy(self._samples)
        # Add a third dimension for univariate parameters
        for name, s in samples.items():
            samples[name] = s.squeeze(0)
            if samples[name].ndimension() == 1:
                samples[name] = samples[name].unsqueeze(-1)
        return samples
        
    def summary(self, names=None):
        raw_samples = self._samples
        if names is not None:
            new_names = [names[i] for i in raw_samples.keys()]
            new_samples = dict(zip(new_names, raw_samples.values()))
        else:
            new_samples = raw_samples
        print_summary(new_samples, prob=0.9, group_by_chain=True)
            
            
        

class NUTS(pyro.infer.mcmc.MCMC):
    def __init__(
            self,
            model,
            step_size=1,
            adapt_step_size=True,
            adapt_mass_matrix=True,
            full_mass=False,
            num_samples=1000,
            warmup_steps=1000,
            num_chains=1,
            init_strategy=None,
            target_accept_prob=0.7,
            jit_compile=False,
            max_tree_depth=10,
            use_multinomial_sampling=False):

        if isinstance(model, SobolevGPModel):
            if model.latent_estimate is True:
                pyro_model = lambda x, y: joint_pyro_model(model, x, y)
            else:
                pyro_model = lambda x, y: marginal_pyro_model(model, x, y)
        elif isinstance(model,  SparseSobolevGPModel):
            pyro_model = lambda x, y : expected_pyro_model(model, x, y)
            
        
        if init_strategy is None:
            init_strategy = pyro.infer.autoguide.initialization.init_to_uniform
        else:
            init_strategy = mcmc_init_strategy(init_strategy)

        nuts_kernel = pyro.infer.mcmc.NUTS(pyro_model, step_size=step_size,
                                        adapt_step_size=adapt_step_size,
                                        adapt_mass_matrix=adapt_mass_matrix, full_mass=full_mass,
                                        target_accept_prob=target_accept_prob, jit_compile=jit_compile,
                                        use_multinomial_sampling=use_multinomial_sampling,
                                        max_tree_depth=max_tree_depth, init_strategy=init_strategy)

        super(NUTS, self).__init__(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains)

    def get_samples(self):
        samples = self._samples
        # Add a third dimension for univariate parameters
        for name, s in samples.items():
            samples[name] = s.squeeze(0)
            if samples[name].ndimension() == 1:
                samples[name] = samples[name].unsqueeze(-1)
        return samples
