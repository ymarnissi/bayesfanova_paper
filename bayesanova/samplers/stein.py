import copy
import math
import os
import time

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch
import tqdm
from torch.optim import SGD, Adagrad, Adam

import bayesanova.models
from bayesanova.models.likelihood import ExpectedLogLikelihood
from bayesanova.models.model import SobolevGPModel, SparseSobolevGPModel

""" This module is adapted from pyro.stein for gaussian process with gpytorch"""


    
class SteinRBFkernel(torch.nn.Module):
    """This is the RBF kernel form stein methods
    """
    def __init__(self, bandwidth=None):
        super(SteinRBFkernel, self).__init__()
        if bandwidth is not None and bandwidth <= 0:
            raise ValueError("Error in kernel bandwidth : it must be positive")
        self.bandwidth = bandwidth


    def forward(self, particles, mode='univ'):
        delta_x = particles.unsqueeze(0) - particles.unsqueeze(1)  # N N D
        norm_sq = delta_x.pow(2.0)  # N N D
        assert delta_x.dim() == 3
        num_particles = particles.size(0)

        if self.bandwidth is None:
            if mode == 'univ':
                index = torch.arange(num_particles)
                norm_sq_s = norm_sq[index > index.unsqueeze(-1), ...]
                median = norm_sq_s.median(dim=0)[0]
                h = median / math.log(num_particles + 1)
                bandwidth = torch.maximum(h, torch.tensor(1e-9))
            else:
                index = torch.arange(num_particles)
                norm_sq_s = torch.sum(norm_sq, dim=2, keepdim=True)
                median = norm_sq_s[index > index.unsqueeze(-1), ...].median(dim=0, keepdim=True)[0]
                h = median / math.log(num_particles + 1)
                bandwidth = torch.maximum(h, torch.tensor(1e-9))
        else:
            bandwidth = self.bandwidth
        log_kernel = -(norm_sq / bandwidth)  # N N D
        grad_log_kernel = -2.0 * delta_x / bandwidth  # N N D

        assert log_kernel.shape == grad_log_kernel.shape

        return log_kernel, grad_log_kernel





class SteinIMQkernel(torch.nn.Module):
    """This is the IMQ kernel for Stein methods
    """
    def __init__(self, alpha=0.5, beta=-0.5, bandwidth=None):
        super(SteinIMQkernel, self).__init__()
        if bandwidth is not None and bandwidth <= 0:
            raise ValueError("Error in kernel bandwidth : it must be positive")
        assert alpha > 0.0, "alpha must be positive."
        assert beta < 0.0, "beta must be negative."
        self.bandwidth = bandwidth
        self.alpha = alpha
        self.beta = beta

    def forward(self, particles, mode='univ'):
        delta_x = particles.unsqueeze(0) - particles.unsqueeze(1)  # N N D
        norm_sq = delta_x.pow(2.0)  # N N D
        assert delta_x.dim() == 3
        num_particles = particles.size(0)

        if self.bandwidth is None:
            if mode == 'univ':
                index = torch.arange(num_particles)
                norm_sq_s = norm_sq[index > index.unsqueeze(-1), ...]
                median = norm_sq_s.median(dim=0)[0]
                h = median / math.log(num_particles + 1)
                bandwidth = torch.maximum(h, torch.tensor(1e-9))
            else:
                index = torch.arange(num_particles)
                norm_sq_s = torch.sum(norm_sq, dim=2, keepdim=True)
                median = norm_sq_s[index > index.unsqueeze(-1), ...].median(dim=0, keepdim=True)[0]
                h = median / math.log(num_particles + 1)
                bandwidth = torch.maximum(h, torch.tensor(1e-9))
        else:
            bandwidth = self.bandwidth
        base_term = self.alpha + norm_sq / bandwidth
        log_kernel = self.beta * torch.log(base_term)  # N N D
        grad_log_kernel = (2.0 * self.beta) * delta_x / bandwidth  # N N D
        grad_log_kernel = grad_log_kernel / base_term
        assert log_kernel.shape == grad_log_kernel.shape

        return log_kernel, grad_log_kernel




class GPSVGD(torch.nn.Module):
    """This is the SVGD algorithm for gaussian processes.
    It is an adapted version of the pyro.stein.svgd that accept gpytorch.model and to meet the same 
    requirements as samplers.mcmc. 
    This class implements gaussian process svgd
    Whether to use a Kernelized Stein Discrepancy that makes use of multivariate test functions 
    (as in [1]) or univariate test functions (as in [2]). Defaults to univariate.
        Args:
            model (gpytorch.model): GP model
            kernel (stein.kernel, optional): kernel. Defaults to None.
            optimizer (torch.optim.optimizer, optional): optimizer. Defaults to None.
            num_particles (int, optional): particles number. Defaults to 50.
            mode (str, optional): Whether to use a Kernelized Stein Discrepancy that makes
                                    use of multivariate test functions (as in [1]) or univariate
                                    test functions (as in [2]). Defaults to 'univ'.
            optimizer_params (dict, optional): optimizer parameters. Defaults to dict().
            r_coef (int, optional): Repulsive coefficient. Large value = large repulsive force. Defaults to 1.
            num_chains (int, optional): chains number. Defaults to 1.
            warmup_steps (int, optional): number of iteration. Defaults to 100.
            scheduler (torch.optim.scheduler., optional): scheduler. Defaults to None.
            scheduler_params (dict, optional): scheduler parameters. Defaults to dict().
            stochastic (bool, optional): whether to add or not random perturbation. Defaults to False.
            init_strategy (str, optional): how to initialize. Defaults to 'uniform'.
            
            
            [1] “Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm,”
            Qiang Liu, Dilin Wang

            [2] “Kernelized Complete Conditional Stein Discrepancy,”
            Raghav Singhal, Saad Lahlou, Rajesh Ranganath
    """
    
    
    def __init__(self, model,
                kernel=None,
                optimizer=None,
                num_particles=50,
                mode='univ',
                optimizer_params = dict(), 
                r_coef=1,
                num_chains=1, 
                warmup_steps=100,
                scheduler=None, 
                scheduler_params = dict(),
                stochastic=False, 
                init_strategy='prior'):

        super(GPSVGD, self).__init__()
        self.model = model
        self.kernel = kernel
        self.num_particles = num_particles
        self.parameters_size = [c.squeeze(-1).squeeze(-1).shape[-1]
                                if c.squeeze(-1).squeeze(-1).ndimension()>=2 else 1 
                                for c in self.model.parameters()]
        self.parameters_shape = [c.shape for c in self.model.parameters()]
        self.name_priors = [i for i,_,_,_,_ in self.model.named_priors()]
        self.mode = mode
        self.init_stratgey = init_strategy
        self.num_chains = num_chains
        self.warmup_steps = warmup_steps


        self.optimizer = dict()
        self.scheduler = dict()

        for name, param in zip(self.name_priors, self.model.parameters()):
            self.optimizer[name] = optimizer([param], **optimizer_params)
            if scheduler is not None:
                self.scheduler[name] = scheduler(self.optimizer[name], **scheduler_params)
            else:
                self.scheduler[name] = None
        
        self.r_coef = r_coef  # Repulsive coefficient. Large value = large repulsive force
        self.stochastic = stochastic  # Add random perturbation to update



    def log_GP_Likelihood(self):
        """logarithm of gaussian process likelihood + log prior
        We can whether use the marginalized or the expected joint version

        Returns:
            tensor: loss
        """
        obs_loss = None
        if isinstance(self.model, SobolevGPModel):
            output = self.model(self.model.train_inputs[0])
            gp_mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            obs_loss = gp_mll(output, self.model.train_targets)
        elif isinstance(self.model, SparseSobolevGPModel):
            output = self.model.conditioning(self.model.train_inputs[0], diag=False)
            ell = ExpectedLogLikelihood(self.model.likelihood, self.model)
            obs_loss = ell(output, self.model.train_targets)
        else:
            output = self.model(self.model.train_inputs[0])
            gp_mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            obs_loss = gp_mll(output, self.model.train_targets)
        return obs_loss
    
    
    @torch.no_grad()
    def step(self):
        """one step of svgd

        Returns:
            tensor: gradient norm at the end of the step
        """
        for name in self.optimizer.keys():
            self.optimizer[name].zero_grad(set_to_none=True)
            
        with torch.enable_grad():
            # compute gradients of log joint model
            logP = self.log_GP_Likelihood()
            loss = torch.sum(logP)
            loss.backward()

            # Cat the gradient in one vector
            dlogP = torch.cat([c.grad.squeeze(-1).squeeze(-1).reshape(self.num_particles, -1) 
                            for c in self.model.parameters()], dim=1).reshape(self.num_particles, -1)  # N*D

        # Compute values and gradients of kernel
        particles = torch.cat([c.detach().squeeze(-1).squeeze(-1).reshape(self.num_particles, -1) 
                            for c in self.model.parameters()], dim=1)

        logK, dlogK = self.kernel(particles, mode='univ')
        if self.mode == 'univ':
            K = logK.exp()
            assert K.shape == (
            self.num_particles, self.num_particles, particles.reshape(self.num_particles, -1).size(-1))
            attractive_grad = torch.einsum("nmd,md->nd", K, dlogP)
            repulsive_grad = torch.einsum("nmd,nmd->nd", K, dlogK)

        else:
            K = logK.sum(-1).exp()
            assert K.shape == (self.num_particles, self.num_particles)
            attractive_grad = torch.mm(K, dlogP)
            repulsive_grad = torch.einsum("nm,nm...->n...", K, dlogK)
            #print(torch.norm(attractive_grad.detach()), torch.norm(repulsive_grad.detach()))

        phi = (attractive_grad + self.r_coef * repulsive_grad).reshape(particles.shape) / self.num_particles
        norm_grad = phi.detach().pow(2.0).mean()
        

        # Set the SVGD gradient
        g = torch.split(phi, self.parameters_size, dim=1)
        
        for c, gc, shape in zip(self.model.parameters(), g, self.parameters_shape):
            c.grad = -gc.reshape(shape)
            
        # Do a gradient step for each parameter
        for k, name in enumerate(self.optimizer.keys()):
            self.optimizer[name].step()
            
            if self.scheduler[name] is not None:
                try:
                    self.scheduler[name].step()
                except:
                    self.scheduler[name].step(g[k].detach().pow(2.0).mean())
                    
        
                    
        # Add perturbation for stochastic mode 
        if self.stochastic:
            with torch.no_grad():
                L = gpytorch.utils.cholesky.psd_safe_cholesky(K.double(),
                                        jitter=gpytorch.settings.cholesky_jitter.value())
                if self.mode == 'univ':
                    perturbation = torch.einsum("nmd,md->nd", 
                                        L.to(dlogP.dtype), torch.randn_like(dlogP))
                else:
                    perturbation = torch.mm(L.to(dlogP.dtype), torch.randn_like(dlogP))
                
                perturbation = torch.split(perturbation, 
                                    self.parameters_size, dim=1)
                

                for c, pc, name in zip(self.model.parameters(), perturbation, self.optimizer.keys()):
                    lr = self.optimizer[name].param_groups[0]['lr']
                    c += pc*np.sqrt(2 * lr/self.num_particles)

        return norm_grad

    def get_named_particles(self):
        """Get constrained particles

        Returns:
            dict: samples
        """
        named_particles = dict()
        model_copy = copy.deepcopy(self.model)
        for param_name, (param_name_raw, param_raw)in zip(self.name_priors, model_copy.named_parameters()):
            constraint = self.model.constraint_for_parameter_name(param_name_raw)
            prior = eval('self.model.'+param_name)
            if constraint is not None:
                named_particles[param_name] = prior.transform(constraint.transform(param_raw.detach()))
            else:
                named_particles[param_name] = prior.transform(param_raw.detach())
        return named_particles
    
    
    def initialize(self):
        """Initialize the algorithm by setting the unconstrained paramters.
        For the moment, only the prior version is implemented

        Raises:
            NotImplementedError: TO DO implement uniform init
            NotImplementedError: TO DO implement init to values
        """
        if self.init_stratgey=='uniform':
            raise NotImplementedError('Uniform initialization is not implemented yet')
        elif self.init_stratgey=='prior':
            self.model.pyro_sample_from_prior()
        else:
            raise NotImplementedError('Initialization to given values is not implemented yet')
    
    def summary(self,**kwargs):
        """
        Display the evolution of gradient norm per iteration for each chain
        """
        plt.figure()
        for i in range(self.num_chains):
            plt.plot(torch.log(torch.tensor(self.gradnorm[i]).T), label='Chain '+str(i))
        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel('Gradient norm [log]')
        
        
    def run(self, train_x, train_y):
        """This is the running process of svgd in training mode.

        Args:
            train_x (tensor): train data
            train_y (tensor): train outcome 
        """
        #self.train_x = train_x.expand(self.num_particles, -1, -1)
        #self.train_y = train_y.expand(self.num_particles, -1)
        self.train_x = self.model.train_inputs[0]
        self.train_y = self.model.train_targets
        self.gradnorm = []
        tab_samples = []
        self.initial_particles = []
        self.train()
        for chain in range(self.num_chains):
            # Initialize : sample from priors (default)
            self.initialize()
            self.initial_particles.append(self.get_named_particles())
            gradnorm = []
            
            if self.num_chains>1:
                text = "Chain "+str(chain)+': '
            else:
                text = 'Warm up'
            postfix = None
            progressbar = tqdm.tqdm(range(self.warmup_steps), desc=text, postfix=postfix)

            for it in progressbar:
                
                gradnorm_step = self.step()
                with torch.no_grad():
                    gradnorm.append(gradnorm_step)
                    current_lr = np.mean([self.optimizer[name].param_groups[-1]['lr'] for name in self.optimizer.keys()])
                    postfix = 'gradnorm='+str(round(gradnorm_step.item(), 4))+', lr='+str(round(current_lr, 6))
                    progressbar.postfix = postfix
                    self.gradnorm.append(gradnorm)
                    
            tab_samples.append(self.get_named_particles())
        self._samples = dict()
        for name in tab_samples[0].keys():
            self._samples[name] = torch.cat([tab_samples[i][name] for i in range(self.num_chains)],
                                            dim=0)


    def get_samples(self):
        """Get the obtained samplers in the running step of the svgd

        Returns:
            dict(): samples 
        """
        
        self.eval()
        samples = copy.deepcopy(self._samples)
        # Add a third dimension for univariate parameters
        for name, s in samples.items():
            samples[name] = s.squeeze(0)
            if samples[name].ndimension() == 1:
                samples[name] = samples[name].detach().unsqueeze(-1)
            else:
                samples[name] = samples[name].detach()
        return samples




class GPSliceSVGD(GPSVGD):
    """
    This is the sliced version of SVGD [3]. 

        Args:
            model (gpytorch.model): GP model
            kernel (stein.kernel, optional): kernel. Defaults to None.
            optimizer (torch.optim.optimizer, optional): optimizer. Defaults to None.
            num_particles (int, optional): particles number. Defaults to 50.
            n_g_update (int, optional): number of g updates steps. Defaults to 10.
            optimizer_params (dict, optional): optimizer parameters. Defaults to dict().
            r_coef (int, optional): Repulsive coefficient. Large value = large repulsive force. Defaults to 1.
            num_chains (int, optional): chains number. Defaults to 1.
            warmup_steps (int, optional): number of iteration. Defaults to 100.
            scheduler (torch.optim.scheduler., optional): scheduler. Defaults to None.
            scheduler_params (dict, optional): scheduler parameters. Defaults to dict()..
            init_strategy (str, optional): how to initialize. Defaults to 'prior'.
            
    [3] Gong, W., Li, Y., & Hernández-Lobato, J. M. (2020).
    Sliced kernelized Stein discrepancy. arXiv preprint arXiv:2006.16531.
    """   

    

    def __init__(self, model, kernel, optimizer,
                num_particles=50,
                n_g_update=10,
                r_coef=1,
                optimizer_params = dict(),
                num_chains=1, 
                warmup_steps=100,
                scheduler=None, 
                scheduler_params=None,
                init_strategy='prior'
                ):
        super(GPSliceSVGD, self).__init__(model=model, kernel=kernel, r_coef=r_coef,
                                        optimizer=optimizer, optimizer_params=optimizer_params,
                                        num_particles=num_particles, num_chains=num_chains, 
                                        warmup_steps=warmup_steps, scheduler=scheduler,
                                        scheduler_params=scheduler_params,
                                        init_strategy=init_strategy)
        
    
        
        try:
            import Sliced_Kernelized_Stein_Discrepancy.src as src
            import Sliced_Kernelized_Stein_Discrepancy.src.Divergence as Divergence
            import Sliced_Kernelized_Stein_Discrepancy.src.Kernel as Kernel
            self.n_g_update = n_g_update  # g update steps
            size = int(sum(self.parameters_size))
            self.g = torch.eye(size).requires_grad_()
            self.r = torch.eye(size)
            self.optimize_g = Adam([self.g], lr=0.001, betas=(0.9, 0.99)) # Same params from git
            self.SE_kernel = Kernel.SE_kernel
            self.d_SE_kernel = Kernel.d_SE_kernel
            self.dd_SE_kernel = Kernel.dd_SE_kernel
            self.compute_max_DSSD_eff_Tensor = Divergence.compute_max_DSSD_eff_Tensor
            self.max_DSSVGD_Tensor = Divergence.max_DSSVGD_Tensor
            self.repulsive_SE_kernel = Kernel.repulsive_SE_kernel
            self.kernel_hyper_maxSVGD = {'bandwidth': None}
            
            #self.optimize_g = Adagrad([self.g], lr=0.001)
        except ImportError as error:
            text = 'Please install Sliced_Kernelized_Stein_Discrepancy package from '\
                        +'https://github.com/WenboGong/Sliced_Kernelized_Stein_Discrepancy'\
                        +' or call clone_git from bayesanova.samplers.utils'
            print(error.__class__.__name__ + ": " + text) 




    def slice_direction_step(self, latent_samples, dlogP):
        """one step of slice direction

        Returns:
            tensor: sliced direction
        """
        self.optimize_g.zero_grad()
        samples1 = latent_samples.clone().detach()
        samples2 = latent_samples.clone().detach()
        r_n = self.r / (torch.norm(self.r, 2, dim=-1, keepdim=True) + 1e-10)
        g_n = self.g / (torch.norm(self.g, 2, dim=-1, keepdim=True) + 1e-10)
        maxSSD, _ = self.compute_max_DSSD_eff_Tensor(samples1, samples2, None,
                                                        self.SE_kernel, self.d_SE_kernel,
                                                        self.dd_SE_kernel,
                                                        flag_U=False, 
                                                        kernel_hyper=self.kernel_hyper_maxSVGD,
                                                        r=r_n, g=g_n, score_samples1=dlogP,
                                                        score_samples2=dlogP.clone(),
                                                        flag_median=True,
                                                        median_power=0.5,
                                                        bandwidth_scale=0.35
                                                        )
        (-maxSSD).mean().backward()
        self.optimize_g.step()
        return maxSSD, g_n
    

    @torch.no_grad()
    def step(self):
        """one step of Slice svgd

            Returns:
                tensor: gradient norm at the end of the step
        """
        for name in self.optimizer.keys():
            self.optimizer[name].zero_grad(set_to_none=True)
            
        with torch.enable_grad():
            logP = self.log_GP_Likelihood()
            loss = torch.sum(logP)
            loss.backward()
            
        # Cat the gradient in one vector
        dlogP = torch.cat([c.grad.squeeze(-1).squeeze(-1).reshape(self.num_particles, -1) 
                            for c in self.model.parameters()], dim=1).reshape(self.num_particles, -1)  # N*D
        
        
        # Compute values and gradients of kernel
        particles = torch.cat([c.detach().squeeze(-1).squeeze(-1).reshape(self.num_particles, -1) 
                                for c in self.model.parameters()], dim=1)
        with torch.enable_grad():
            for i in range(self.n_g_update):
                    _, g_n = self.slice_direction_step(particles, dlogP)

        phi = self.max_DSSVGD_Tensor(particles, None, self.SE_kernel, 
                                                        self.repulsive_SE_kernel,
                                                        r=self.r, g=g_n,
                                                        flag_median=True, median_power=0.5,
                                                        kernel_hyper=self.kernel_hyper_maxSVGD,
                                                        score=dlogP,
                                                        bandwidth_scale=0.35,
                                                        repulsive_coef=self.r_coef)  # * x sam x dim

        norm_grad = phi.detach().pow(2.0).mean()
        
        
        # Set the SVGD gradient
        g = torch.split(phi.squeeze(0), self.parameters_size, dim=1)
        
        for c, gc, shape in zip(self.model.parameters(), g, self.parameters_shape):
            c.grad = -gc.reshape(shape)
            
        # Do a gradient step for each parameter
        for k, name in enumerate(self.optimizer.keys()):
            self.optimizer[name].step()
            
            if self.scheduler[name] is not None:
                try:
                    self.scheduler[name].step()
                except:
                    self.scheduler[name].step(g[k].detach().pow(2.0).mean())

        return norm_grad
    


