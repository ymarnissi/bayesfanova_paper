import numbers
import warnings

import copy 
import gpytorch
import numpy as np
import torch
from gpytorch.lazy import BatchRepeatLazyTensor
from gpytorch.models.exact_prediction_strategies import \
    DefaultPredictionStrategy

from bayesanova.models import likelihood, sobolev_kernel
from bayesanova.models.utils import first_order_InteractionEffect_to_index, positive_definite_full_covar
from bayesanova.models.prior import DirichletMixingPrior
method_inv_root = "lanczos"


class SobolevGPModel(gpytorch.models.ExactGP):
    """This is the basic SobolevGPModel. 
    It defines main Effects covariance functions, first order interaction effect covariance functions
    and residual effect covariance functions, their respective forward models and their respective predictive models
    Args:
        train_x (tensor): train data inputs
        train_y (tensor): train data outputs
        poly_order (int): polynomial order of the kernels
        model_order (int): model order (if 1 then only main effect, if 2 then main effect + first order interaction effect)
        batch_shape (torch.Size, optional): batch size to work in a bacth mode
        likelihood (gpytorch.likelihood, optional): likelihood. Defaults to None.
        coef (float or tensor, optional): constant to weight Bernoulli Polynomials. Defaults to None.
        correction (bool, optional): Correct the interaction effects if True as in [Brian, 2008]. Defaults to False.
        residual (bool, optional): Add the residual effects component if True. Defaults to False.
        normalization (bool, optional): Normalize covariance with likelihood noise if true. Defaults to False.
    """

    def __init__(self, train_x, train_y, poly_order, model_order,
                batch_shape=torch.Size([]),
                likelihood=None, 
                coef=None, 
                correction=False, 
                residual=False,  
                normalization=False):  

        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch_shape)
        super(SobolevGPModel, self).__init__(train_x, train_y, likelihood=likelihood)
        
        N, ndims = train_x.shape[-2], train_x.shape[-1]
        self.num_train = N
        self.dim_train = ndims
        self.model_order = model_order
        self.poly_order = poly_order
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        if coef is not None:
            coef = torch.tensor([coef])
        self.coef = coef
        self.correction = correction
        self.residual = residual
        self.normalization = normalization

        # Default : for Gaussian likelihood, use the marginal posterior (without estimation of latent)
        if isinstance(self.likelihood, gpytorch.likelihoods.GaussianLikelihood):
            self.latent_estimate = False
        else:
            self.latent_estimate = True
            warnings.warn('Predictions are not implemented for non-Gaussian likelihoods.')
            

        # Main Effect kernel
        main_effect_base_kernel = sobolev_kernel.univ_main_effect(poly_order, batch_shape=batch_shape, c=self.coef)
        self.__MainEffect_covar_module = sobolev_kernel.ScaleKernel(
            sobolev_kernel.ScaleAdditiveStructureKernel(main_effect_base_kernel, ndims, batch_shape=batch_shape),
            batch_shape=batch_shape, likelihood=self.likelihood if self.normalization else None)

        # Interaction Effect Kernel
        InteractionEffect_base_kernel = None
        if self.model_order == 2:
            if correction and coef is not None and coef > 1:
                InteractionEffect_base_kernel = (
                    sobolev_kernel.univ_main_effect(poly_order, c=1, batch_shape=batch_shape),
                    sobolev_kernel.StatBernPoly(poly_order, c=torch.sqrt(self.coef - 1),
                                                batch_shape=batch_shape))
            else:
                InteractionEffect_base_kernel = sobolev_kernel.univ_main_effect(poly_order, c=self.coef,
                                                                                batch_shape=batch_shape)

            self.__InteractionEffect_covar_module = sobolev_kernel.ScaleKernel(
                sobolev_kernel.ScaleAdditiveStructureKernel(
                    InteractionEffect_base_kernel, ndims * (ndims - 1) // 2
                    , batch_shape=batch_shape),
                batch_shape=batch_shape, likelihood=self.likelihood if self.normalization else None)

        # Residual effect kernel
        if self.residual is True:
            full_kernel = sobolev_kernel.FullKernel(main_effect_base_kernel, batch_shape=batch_shape)
            base_kernel = gpytorch.kernels.ProductStructureKernel(base_kernel=full_kernel, num_dims=ndims)
            
            mainEffect_active_kernel = sobolev_kernel.AdditiveStructureKernel(main_effect_base_kernel,
                                                                            num_dims=ndims, batch_shape=batch_shape)
            active_kernel = (mainEffect_active_kernel,)

            if self.model_order == 2:

                InteractionEffect_active_kernel = sobolev_kernel.AdditiveStructureKernel(InteractionEffect_base_kernel,
                                                                                          num_dims=ndims * (
                                                                                                 ndims - 1) // 2,
                                                                                        batch_shape=batch_shape)
                active_kernel = active_kernel + (InteractionEffect_active_kernel,)

            self.__ResidualEffect_covar_module = sobolev_kernel.ScaleKernel(sobolev_kernel.ResidualKernel(
                base_kernel, active_kernel,  batch_shape=batch_shape), batch_shape=batch_shape,
                likelihood=self.likelihood if self.normalization else None)

        self.latent_num = ndims if self.model_order == 1 else ndims * (ndims + 1) // 2
        if self.residual:
            self.latent_num = self.latent_num + 1

        if self.latent_estimate:
            self.enable_latent()
        self.product_model = False

        self.prediction_strategy = None
        # Private
        self.__mainEffect = None
        self.__interactionEffect = None
        self.__residualEffect = None
        self.__local_scale = None
        self.__global_scale = None
        self.__covar_module = None
        self.__true_latent = None
        self.__marginal_posterior = None
        self.__joint_posterior = None
        self.__dtype = train_x.dtype
        self.__cache = dict()  # Cache to contain precomputed quantities

    def enable_latent(self):
        self.latent_estimate = True
        latent = torch.zeros(tuple(self.train_inputs[0][..., 0, 0].shape) +
                            (self.latent_num, self.num_train)).squeeze(0)
        self.register_parameter('latent', torch.nn.parameter.Parameter(latent))
        self.register_prior("latent_prior",
                            gpytorch.priors.NormalPrior(torch.tensor([0.0]),
                                                        torch.tensor([1.0])), 'latent')

    @property
    def mainEffect(self):
        return self.__MainEffect_covar_module

    @property
    def interactionEffect(self):
        c = None
        if self.model_order == 2:
            c = self.__InteractionEffect_covar_module
        return c

    @property
    def residualEffect(self):
        c = None
        if self.residual:
            c = self.__ResidualEffect_covar_module
        return c

    @property
    def covar_module(self):
        c = None
        c = self.mainEffect
        if self.model_order == 2:
            c = c + self.interactionEffect
        if self.residual:
            c = c + self.residualEffect
        return c

    @property
    def local_scale(self):
        # local scales = (main effect, interaction effect)
        s = (self.mainEffect.base_kernel.outputscale,)
        if self.model_order == 2:
            s = s + (self.interactionEffect.base_kernel.outputscale,)
        return s

    @local_scale.setter
    def local_scale(self, s):
        self.mainEffect.base_kernel.outputscale = s[0]
        if self.model_order == 2:
            self.interactionEffect.base_kernel.outputscale = s[1]

    @property
    def global_scale(self):
        # global scales = (main effect, interaction effect, residual effect)
        s = (self.mainEffect.outputscale,)
        if self.model_order == 2:
            s = s + (self.interactionEffect.outputscale,)
        if self.residual is True:
            s = s + (self.residualEffect.outputscale,)
        return s

    @global_scale.setter
    def global_scale(self, s):
        self.mainEffect.outputscale = s[0]
        if self.residual:
            self.residualEffect.outputscale = s[-1]
        if self.model_order == 2:
            self.interactionEffect.outputscale = s[1]

    def __latent_R_covar(self):
        main_dist = self.main_effect_forward(self.train_inputs[0])
        R1 = main_dist.lazy_covariance_matrix
        R = R1.add_jitter()
        if self.model_order == 2:
            int_dist = self.first_order_interaction_effect_forward(self.train_inputs[0])
            R2 = int_dist.lazy_covariance_matrix
            R = gpytorch.lazy.CatLazyTensor(R, R2.add_jitter(), dim=-3)
        if self.residual:
            res_dist = self.residual_effect_forward(self.train_inputs[0])
            R3 = res_dist.lazy_covariance_matrix.unsqueeze(-3)
            R = gpytorch.lazy.CatLazyTensor(R, R3.add_jitter(), dim=-3)
        return R

    @property
    def true_latent(self):
        # latent = L(X)^-1*(true_latent)=M(X).T*(true_latent)
        if 'latent_R' in self.__cache.keys():
            R = self.__cache['latent_R']
        else:
            R = self.__latent_R_covar()
            self.__cache['latent_R'] = R

        # Get cached L. (or compute it if we somehow don't already have the covariance cache)
        # epoch_time_start = time.time()
        if 'latent_R.root_decomposition' in self.__cache.keys():
            L = self.__cache['latent_R.root_decomposition']
        else:
            L = R.root_decomposition(method=method_inv_root).root
            self.__cache['latent_R.root_decomposition'] = L

        if 'latent_R.root_decomposition.inverse' in self.__cache.keys():
            M = self.__cache['latent_R.root_decomposition.inverse']
        else:
            M = R.root_inv_decomposition(method=method_inv_root).root
            self.__cache['latent_R.root_decomposition.inverse'] = M

        # print('Seconds since epoch', time.time() - epoch_time_start)
        z = L.matmul(self.latent.unsqueeze(-1)).squeeze(-1)
        # Multiply with local and global scales
        local_scale = self.local_scale + (torch.ones_like(self.global_scale[-1]),) \
            if self.residual else self.local_scale

        scales = [torch.sqrt(torch.tensor([local_scale[i]]) * self.global_scale[i]) if local_scale[i].ndim == 0 else
                  torch.sqrt(local_scale[i] * self.global_scale[i]) for i in range(len(self.global_scale))]
        z = torch.cat(scales, dim=-1).unsqueeze(-1).mul(z)
        return z.to(self.__dtype)

    @true_latent.setter
    def true_latent(self, s):

        if 'latent_R' in self.__cache.keys():
            R = self.__cache['latent_R']
        else:
            if 'latent_R' in self.__cache.keys():
                R = self.__cache['latent_R']
            else:
                R = self.__latent_R_covar()
            self.__cache['latent_R'] = R

        if 'latent_R.root_decomposition' in self.__cache.keys():
            L = self.__cache['latent_R.root_decomposition']
        else:
            L = R.root_decomposition(method=method_inv_root).root
            self.__cache['latent_R.root_decomposition'] = L

        if 'latent_R.root_decomposition.inverse' in self.__cache.keys():
            M = self.__cache['latent_R.root_decomposition.inverse']
        else:
            M = R.root_inv_decomposition(method=method_inv_root).root
            self.__cache['latent_R.root_decomposition.inverse'] = M
        # latent = L.add_jitter().inv_matmul(s.unsqueeze(-1)).squeeze(-1)
        latent = M.transpose(-1, -2).matmul(s.unsqueeze(-1)).squeeze(-1)
        # Multiply with local and global scales
        local_scale = self.local_scale + (torch.ones_like(self.global_scale[-1]),) \
            if self.residual else self.local_scale
        scales = [torch.sqrt(torch.tensor([local_scale[i]]) * self.global_scale[i]) if local_scale[i].ndim == 0 else
                  torch.sqrt(local_scale[i] * self.global_scale[i]) for i in range(len(self.global_scale))]
        latent = torch.cat(scales, dim=-1).unsqueeze(-1).mul(latent)
        self.latent = torch.nn.Parameter(latent.to(self.__dtype))

    def latent_forward(self):
        '''This function computes f(x)=\mu + \sum_j f_j(x)'''
        if self.latent_estimate is False:
            raise ValueError("Latent is not defined. Enable latent first.")
        m = self.mean_module(self.train_inputs[0])
        f = self.true_latent.sum(dim=-2)
        return m + f

    def __clear_latent_cache(self):
        # Clear cache for latent variable
        del self.cache['latent_R']
        del self.cache['latent_R.root_decomposition']

    @property
    def marginal_posterior(self):
        return gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

    @property
    def joint_posterior(self):
        return likelihood.JointLogLikelihood(self.likelihood, self)
    

    def set_global_scale_constraint(self, constraint):
        self.mainEffect.register_constraint("raw_outputscale", constraint[0])
        if self.residual:
            self.residualEffect.register_constraint("raw_outputscale", constraint[-1])
        if self.model_order == 2:
            self.interactionEffect.register_constraint("raw_outputscale", constraint[1])

    def set_local_scale_constraint(self, constraint):
        self.mainEffect.base_kernel.register_constraint("raw_outputscale", constraint[0])
        if self.model_order == 2:
            self.interactionEffect.base_kernel.register_constraint("raw_outputscale", constraint[1])

    def set_global_scale_prior(self, prior):
        self.mainEffect.register_prior("outputscale_prior", prior[0], "outputscale")
        if self.residual:
            self.residualEffect.register_prior("outputscale_prior", prior[-1], "outputscale")
        if self.model_order == 2:
            self.interactionEffect.register_prior("outputscale_prior", prior[1], "outputscale")

    def set_local_scale_prior(self, prior):
        self.mainEffect.base_kernel.register_prior("outputscale_prior", prior[0], "outputscale")
        if self.model_order == 2:
            if prior[1] is not None: 
                self.interactionEffect.base_kernel.register_prior("outputscale_prior", prior[1], "outputscale")
            elif self.product_model is False:  
                main_scales = self.local_scale[0]
                shape = self.local_scale[1].shape
                weights = torch.tensor([main_scales[i]*main_scales[j] for\
                    i in range(self.dim_train-1) for j in range(i+1, self.dim_train)]).reshape(shape)
                prior_inter = copy.deepcopy(prior[0]) 
                prior_inter = DirichletMixingPrior(scale=weights/weights.sum(-1))
                self.interactionEffect.base_kernel.register_prior("outputscale_prior", prior_inter, "outputscale")

    
 

    def set_noise_prior(self, prior):
        self.likelihood.register_prior("noise_prior", prior, "noise")

    def set_noise_constraint(self, constraint):
        self.likelihood.noise_covar.raw_noise_constraint = constraint

    def set_mean_prior(self, prior):
        self.mean_module.register_prior("mean_prior", prior, "constant")

    def setf_mean_constraint(self, constraint):
        self.mean_module.register_constraint("constant", constraint)

    def get_main_effect_base_kernel(self, **kwargs):
        return sobolev_kernel.univ_main_effect(self.poly_order, c=self.coef, **kwargs)

    def get_interaction_effect_base_kernel(self, **kwargs):
        return sobolev_kernel.univ_first_order_interaction_effect(self.poly_order, c=self.coef,
                                                                correction=self.correction,
                                                                  **kwargs)

    def main_effect_forward(self, x):
        mean_x = torch.tensor([0.0])
        base_kernel = self.get_main_effect_base_kernel()
        covar_x = base_kernel(x, last_dim_is_batch=True)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def first_order_interaction_effect_forward(self, x):
        D = x.shape[-1]
        index = torch.tensor([[i, j] for i in np.arange(0, D - 1)
                            for j in np.arange(i + 1, D)], dtype=torch.long)

        X = x[..., index].transpose(-2, -3)
        mean_x = torch.tensor([0.0])
        base_kernel = self.get_interaction_effect_base_kernel()
        covar_x = base_kernel(X)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def residual_effect_forward(self, x):
        mean_x = torch.tensor([0.0])
        covar_x = self.residualEffect(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def forward(self, x):
        mean_x = self.mean_module(x)
        # covar_x = gpytorch.lazy.LazyEvaluatedKernelTensor(x, x, self.covar_module)
        covar_x = self.covar_module(x).add_jitter()
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def to_batch(self, N):
        expanded_train_x = self.train_inputs[0].expand(N, -1, -1)
        expanded_train_y = self.train_targets.expand(N, -1)
        return SobolevGPModel(expanded_train_x,
                            expanded_train_y,
                            self.poly_order,
                            self.model_order,
                            batch_shape=torch.Size([N]),
                            likelihood=None, 
                            coef=self.coef, 
                            correction=self.correction, 
                            residual=self.residual,  
                            normalization=self.normalization)


    '''   Prediction part   '''

    # Prediction mode
    def predict_on(self, samples):
        self.pyro_load_from_samples(samples)
        self.eval()
        if self.prediction_strategy is None:
            train_inputs = self.train_inputs[0]
            train_inputs_pred = list(self.train_inputs) if self.train_inputs is not None else []
            train_prior_dist = self.forward(train_inputs)
            #train_output = self.likelihood(train_prior_dist)
            #train_covar = train_output.lazy_covariance_matrix.detach().add_jitter()
            #train_covar_inv_root = train_covar.root_inv_decomposition(
            #    method=method_inv_root).root
            self.prediction_strategy = DefaultPredictionStrategy(
                train_inputs=train_inputs_pred,
                train_prior_dist=train_prior_dist,
                train_labels=self.train_targets,
                likelihood=self.likelihood)#,
                #inv_root=train_covar_inv_root)

    def predictive_latent(self, x, outcome=False):
        # Give the predictive distributions of the latent 
        train_inputs = self.train_inputs[0]
        batch_size, train_size, test_size, ndims = train_inputs.shape[0], train_inputs.shape[1], x.shape[0], \
                                                train_inputs.shape[-1]
        joint_size = train_size + test_size

        expanded_test_x = x.unsqueeze(0).repeat(batch_size, 1, 1)


        # Train part
        train_dist = self.forward(train_inputs)
        train_output = self.likelihood(train_dist)
        train_mean, train_covar = train_output.mean.detach(), train_output.lazy_covariance_matrix.detach()

        # Test part
        test_dist = self.forward(expanded_test_x)
        test_mean, test_covar = test_dist.mean.detach(), test_dist.lazy_covariance_matrix.detach()
        test_train_covar = self.covar_module(expanded_test_x, train_inputs)

        full_covar = positive_definite_full_covar(train_covar, test_train_covar, test_covar)
        
        full_mean = torch.zeros(batch_size, joint_size)
        full_mean[:, :train_size] = train_mean
        full_mean[:, train_size:] = test_mean
        predictive_mean, predictive_covar =  self.prediction_strategy.exact_prediction(full_mean, full_covar)
        
        if outcome is True: 
            noise = gpytorch.lazy.DiagLazyTensor(self.likelihood.noise.expand(self.likelihood.noise.shape[0], predictive_covar.shape[-1]))
            predictive_covar = predictive_covar + noise 

        predictive_density = gpytorch.distributions.MultivariateNormal(predictive_mean, predictive_covar)

        return predictive_density


    def predictive_main_effect_component(self, x, k=None):
        if not(isinstance(self.likelihood, gpytorch.likelihoods.GaussianLikelihood)):
            raise NotImplementedError('Predictions are not implemented yet for non-gaussian likelihood.')
        # Give the predictive distributions of the main effect components k
        train_inputs = self.train_inputs[0]
        batch_size, train_size, test_size, ndims = train_inputs.shape[0], train_inputs.shape[1], x.shape[0], \
                                                train_inputs.shape[-1]
        joint_size = train_size + test_size

        # If k is None then provide all the main-effect_components
        if k is None:
            k = torch.arange(0, ndims)
        
        # If k is a number, return a tensor
        if isinstance(k, numbers.Number):
            k = torch.tensor([k])
            
        # If k is list or numpy, change to tensor
        if not(torch.is_tensor(k)):
            k = torch.tensor(k)
        
        test_xk = x[..., k]
        train_xk = train_inputs[0, :, k]

        # Train part
        train_dist = self.forward(train_inputs)
        train_output = self.likelihood(train_dist)
        train_mean, train_covar = train_output.mean.detach(), train_output.lazy_covariance_matrix.detach().add_jitter()

        # Test part
        covar_module = self.get_main_effect_base_kernel()
        l_outputscale = self.local_scale[0][:, k]
        g_outputscale = self.global_scale[0]
        if self.mainEffect.likelihood is not None:
            g_outputscale = g_outputscale * self.mainEffect.likelihood.noise
        weight = g_outputscale.mul(l_outputscale)
        weight = torch.transpose(weight, 1, 0).unsqueeze(-1)  # k \times num_samples \times k \times 1

        predictive_density = dict()

        # Through main components
        for i in torch.arange(k.numel()):
            test_covar = covar_module(test_xk[..., i])  # M \times M
            test_covar = BatchRepeatLazyTensor(test_covar, batch_repeat=torch.Size((batch_size,)))
            test_covar = test_covar.mul(weight[i].squeeze(0).unsqueeze(-1))
            test_train_covar = covar_module(test_xk[..., i], train_xk[..., i])  # M \times N
            test_train_covar = BatchRepeatLazyTensor(test_train_covar, batch_repeat=torch.Size((batch_size,)))
            test_train_covar = test_train_covar.mul(weight[i].squeeze(0).unsqueeze(-1))
            # t1 = time.time()
            full_covar = positive_definite_full_covar(train_covar, test_train_covar, test_covar)
            
            
            # print('construction', time.time()-t1)
            full_mean = torch.zeros(batch_size, joint_size)
            full_mean[:, :train_size] = train_mean
            # Make the prediction
            # t1 = time.time()
            predictive_mean, predictive_covar = self.prediction_strategy.exact_prediction(full_mean, full_covar)
            # print('prediction', time.time() - t1)
            predictive_density[str(int(k[i]))] = gpytorch.distributions.MultivariateNormal(predictive_mean,
                                                                                        predictive_covar)

        return predictive_density

    def predictive_interaction_effect_component(self, x, k=None):  # Only available in th batch mode
        # Give the predictive distribution of the first order interaction effect component between k[0] and k[1]
        
        if not(isinstance(self.likelihood, gpytorch.likelihoods.GaussianLikelihood)):
            raise NotImplementedError('Predictions are not implemented yet for non-gaussian likelihood.')
        
        train_inputs = self.train_inputs[0]
        batch_size, train_size, test_size, ndims = train_inputs.shape[0], train_inputs.shape[1], x.shape[0], \
                                                train_inputs.shape[-1]
        joint_size = train_size + test_size
        test_x = x.unsqueeze(0).repeat(batch_size, 1, 1)

        # If k is None then provide all the first_order-effect_components
        if k is None:
            k = torch.tensor([[i, j] for i in np.arange(0, ndims - 1)
                                    for j in np.arange(i + 1, ndims)], dtype=torch.long)
        
        # If k is list or numpy, change to tensor
        if not(torch.is_tensor(k)):
            k = torch.tensor(k)
        
        # If there only one target component, expand dimension
        if k.ndimension()==1:
            k = k.expand(1, -1)
        
        # Get the outputscale for the target component
        index = [first_order_InteractionEffect_to_index(ndims, list(j)) for j in k.numpy()]
        #print(index)

        # Compute the training part
        train_dist = self.forward(train_inputs)
        train_output = self.likelihood(train_dist)
        train_mean, train_covar = train_output.mean, train_output.lazy_covariance_matrix.add_jitter()

        # Compute the test part
        covar_module = self.get_interaction_effect_base_kernel()


        # Compute the test part
        predictive_density = dict()

        # Through first_order_interaction components
        for i in torch.arange(len(index)):
            l_outputscale = self.local_scale[1][:, index[i]]
            g_outputscale = self.global_scale[1]
            if self.interactionEffect.likelihood is not None:
                g_outputscale = g_outputscale * self.interactionEffect.likelihood.noise
            weight = g_outputscale.mul(l_outputscale.unsqueeze(1))
            test_covar = covar_module(test_x[:, :, k[i]]).mul(weight.unsqueeze(2))
            test_train_covar = covar_module(test_x[:, :, k[i]], train_inputs[:, :, k[i]]).mul(weight.unsqueeze(2))
            
            full_covar = positive_definite_full_covar(train_covar, test_train_covar, test_covar)

            full_mean = torch.zeros(batch_size, joint_size)
            full_mean[:, :train_size] = train_mean
            # Make the prediction
            predictive_mean, predictive_covar = self.prediction_strategy.exact_prediction(full_mean, full_covar)
            predictive_density[str(k[i].numpy())] = gpytorch.distributions.MultivariateNormal(predictive_mean,
                                                                                        predictive_covar)

        return predictive_density

    def predictive_residual_effect_component(self, x):  # Only available in th batch mode
        # Give the predictive distribution of the residual effect component
        
        if not(isinstance(self.likelihood, gpytorch.likelihoods.GaussianLikelihood)):
            raise NotImplementedError('Predictions are not implemented yet for non-gaussian likelihood.')
        
        train_inputs = self.train_inputs[0]
        batch_size, train_size, test_size, ndims = train_inputs.shape[0], train_inputs.shape[1], x.shape[0], \
                                                train_inputs.shape[-1]
        joint_size = train_size + test_size

        expanded_test_x = x.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Compute the training part
        train_dist = self.forward(train_inputs)
        train_output = self.likelihood(train_dist)
        train_mean, train_covar = train_output.mean, train_output.lazy_covariance_matrix
        

        # Compute the test part
        covar_module = self.residualEffect
        test_covar = covar_module(expanded_test_x).add_jitter()
        
        test_train_covar = covar_module(expanded_test_x, train_inputs)
        
        full_covar = train_covar.cat_rows(test_train_covar, test_covar)
        
        full_covar = positive_definite_full_covar(train_covar, test_train_covar, test_covar)
                
        full_mean = torch.zeros(batch_size, joint_size)
        full_mean[:, :train_size] = train_mean
        full_output = gpytorch.distributions.MultivariateNormal(full_mean, full_covar)

        # Determine the shape of the joint distribution
        batch_shape = full_output.batch_shape
        joint_shape = full_output.event_shape
        tasks_shape = joint_shape[1:]  # For multitask learning
        test_shape = torch.Size([joint_shape[0] - self.prediction_strategy.train_shape[0], *tasks_shape])

        # Make the prediction
        predictive_mean, predictive_covar = self.prediction_strategy.exact_prediction(full_mean, full_covar)

        # Reshape predictive mean to match the appropriate event shape
        predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()

        return full_output.__class__(predictive_mean, predictive_covar)





'''                        Sparse models                   '''

class SparseSobolevGPModel(SobolevGPModel):
    """This is the Sparse version of the SobolevGPModel.
    
    It uses inducing points. 
    

    Args:
        train_x (tensor): train data inputs
        train_y (tensor): train data outputs
        poly_order (int): polynomial order of the kernels
        model_order (int): model order (if 1 then only main effect, if 2 then main effect + first order interaction effect)
        batch_shape (torch.Size, optional): batch size to work in a bacth mode
        likelihood (gpytorch.likelihood, optional): likelihood. Defaults to None.
        coef (float or tensor, optional): constant to weight Bernoulli Polynomials. Defaults to None.
        correction (bool, optional): Correct the interaction effects if True as in [Brian, 2008]. Defaults to False.
        residual (bool, optional): Add the residual effects component if True. Defaults to False.
        normalization (bool, optional): Normalize covariance with likelihood noise if true. Defaults to False.
        inducing_points (tensor, optional): Inducing points to apporoximate the GP. Defaults to None.
        latent (tensor, optional): Latent function. Defaults to None.


    """
    # This is the Sparse version of the SobolevGPModel
    def __init__(self, train_x, train_y,  poly_order, model_order,
                likelihood=None,
                batch_shape=torch.Size([]),
                coef=None,
                correction=False,
                residual=False,
                normalization=False,
                inducing_points=None,
                latent=None):
            
        super(SparseSobolevGPModel, self).__init__(train_x=train_x, train_y=train_y, 
                                                poly_order=poly_order, model_order=model_order, 
                                                batch_shape=batch_shape, coef=coef, 
                                                correction=correction, likelihood=likelihood,
                                                residual=residual, normalization=normalization)


        if inducing_points is None:
            inducing_points = train_x[:10, :]
        self.inducing_points = inducing_points
        self.inducing_num_samples = inducing_points.shape[-2]
        if latent is None:
            latent = torch.zeros(self.inducing_num_samples)
        self.latent = torch.nn.parameter.Parameter(latent)
        self.__dtype = train_x.dtype
        self._true_latent = None
        
        self.register_prior("latent_prior",
                    gpytorch.priors.MultivariateNormalPrior(torch.zeros(self.inducing_num_samples),
                                                            torch.eye(self.inducing_num_samples)), 'latent')

    
    def cholesky_factor(self, induc_induc_covar):
        L = gpytorch.utils.cholesky.psd_safe_cholesky(gpytorch.delazify(induc_induc_covar).double(),
                                                    jitter=gpytorch.settings.cholesky_jitter.value())
        return gpytorch.lazy.TriangularLazyTensor(L)

    @property
    def expected_posterior(self):
        return likelihood.ExpectedLogLikelihood(self.likelihood, self)
    
    @property
    def true_latent(self):
        # latent = L^-1*(true_latent-mean)
        dis = self.forward(self.inducing_points)
        m, R = dis.mean, dis.lazy_covariance_matrix
        L = self.cholesky_factor(R)
        z = L.matmul(self.latent.double()) + m
        return z.to(self.__dtype)

    @true_latent.setter
    def true_latent(self, s):
        dis = self.forward(self.inducing_points)
        m, R = dis.mean, dis.lazy_covariance_matrix
        L = self.cholesky_factor(R)
        z = s - m
        latent = L.inv_matmul(z.double())
        self.latent = latent.to(self.__dtype)

    
    
    def conditioning(self, x, diag=False):
        if x.ndimension() > 2:
            batch_inducing_points = self.inducing_points.unsqueeze(0).repeat(x.shape[0], 1, 1)
        else:
            batch_inducing_points = self.inducing_points
        full_inputs = torch.cat([batch_inducing_points, x], dim=-2)
        full_output = self.forward(full_inputs)
        full_covar = full_output.lazy_covariance_matrix

        # Covariance terms
        num_induc = self.inducing_points.size(-2)
        x_mean = full_output.mean[..., num_induc:]
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter()
        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        L = self.cholesky_factor(induc_induc_covar)
        # inv(L)*RZX
        A = L.inv_matmul(induc_data_covar.double()).to(self.__dtype)

        predictive_mean = torch.matmul(A.transpose(-1, -2), self.latent.unsqueeze(-1)).squeeze(-1) + x_mean

        if diag:
            # only needs to compute the variance (diagonal matrix)
            row_col_iter = torch.arange(0, data_data_covar.matrix_shape[-1], dtype=torch.long)
            predictive_covar = gpytorch.lazy.DiagLazyTensor(
                data_data_covar.add_jitter(1e-4)[..., row_col_iter, row_col_iter])
            D = A.transpose(-1, -2).matmul(A)[..., row_col_iter, row_col_iter]
            predictive_covar.add_diag(D.mul(-1))
        else:
            predictive_covar = gpytorch.lazy.SumLazyTensor(
                data_data_covar.add_jitter(1e-4),
                gpytorch.lazy.MatmulLazyTensor(A.transpose(-1, -2), A.mul(-1)),
            )
        # Return the distribution
        return gpytorch.distributions.MultivariateNormal(predictive_mean, gpytorch.delazify(predictive_covar))
    

    
    
    ''' Prediction  '''
    
    def main_effect_conditioning(self, x, k=0, diag=False):  # Only available in th batch mode
        # Give the predictive distribution of the k^th main effect component
        batch_size = x.size(0)
        batch_inducing_points = self.inducing_points.unsqueeze(0).repeat(batch_size, 1, 1)

        # Compute the latent part
        induc_induc_covar = self.covar_module(batch_inducing_points).add_jitter()

        # Get the outputscale for the target component
        l_outputscale = self.local_scale[0][:, k]
        g_outputscale = self.global_scale[0]
        if self.mainEffect.likelihood is not None:
            g_outputscale = g_outputscale * self.mainEffect.likelihood.noise

        weight = g_outputscale.mul(l_outputscale.unsqueeze(1))

        # Compute the test part
        covar_module = self.get_main_effect_base_kernel(batch_shape=torch.Size([batch_size]))
        data_data_covar = covar_module(x[:, :, k].unsqueeze(-1)).mul(weight.unsqueeze(2))
        induc_data_covar = covar_module(batch_inducing_points[:, :, k].unsqueeze(-1), x[:, :, k].unsqueeze(-1)).mul(
            weight.unsqueeze(2)).evaluate()

        # Conditionning
        L = self.cholesky_factor(induc_induc_covar)
        # inv(L)*RZX
        A = L.inv_matmul(induc_data_covar.double()).to(self.__dtype)
        predictive_mean = torch.matmul(A.transpose(-1, -2), self.latent.unsqueeze(-1)).squeeze(-1)

        if diag:
            # only needs to compute the variance (diagonal matrix)
            row_col_iter = torch.arange(0, data_data_covar.matrix_shape[-1], dtype=torch.long)
            predictive_covar = gpytorch.lazy.DiagLazyTensor(
                data_data_covar.add_jitter(1e-4)[..., row_col_iter, row_col_iter])
            D = A.transpose(-1, -2).matmul(A)[..., row_col_iter, row_col_iter]
            predictive_covar.add_diag(D.mul(-1))
        else:
            predictive_covar = gpytorch.lazy.SumLazyTensor(
                data_data_covar.add_jitter(1e-4),
                gpytorch.lazy.MatmulLazyTensor(A.transpose(-1, -2), A.mul(-1)),
            )
        # Return the distribution
        return gpytorch.distributions.MultivariateNormal(predictive_mean, gpytorch.delazify(predictive_covar))
    
    
    

    def interaction_effect_conditioning(self, x, k=None, diag=False):  # Only available in th batch mode
        # Give the predictive distribution of the first order interaction effect component between k[0] and k[1]
        if k is None:
            k = [0, 1]
        # Give the predictive distribution of the k^th main effect component
        batch_size = x.size(0)
        batch_inducing_points = self.inducing_points.unsqueeze(0).repeat(batch_size, 1, 1)

        # Compute the latent part
        induc_induc_covar = self.covar_module(batch_inducing_points).add_jitter()

        # Get the outputscale for the target component
        index = int(np.sum([1 for i in range(k[0] + 1) for j in np.arange(k[0] + 1, k[1] + 1)]) - 1)
        l_outputscale = self.local_scale[1][:, index]
        g_outputscale = self.global_scale[1]
        if self.interactionEffect.likelihood is not None:
            g_outputscale = g_outputscale * self.interactionEffect.likelihood.noise
        weight = g_outputscale.mul(l_outputscale.unsqueeze(1))

        # Compute the test part
        covar_module = self.get_interaction_effect_base_kernel(batch_shape=torch.Size([batch_size]))
        data_data_covar = covar_module(x[:, :, k]).mul(weight.unsqueeze(2))
        induc_data_covar = covar_module(batch_inducing_points[:, :, k], x[:, :, k]).mul(weight.unsqueeze(2)).evaluate()

        # Conditionning
        L = self.cholesky_factor(induc_induc_covar)
        # inv(L)*RZX
        A = L.inv_matmul(induc_data_covar.double()).to(self.__dtype)
        predictive_mean = torch.matmul(A.transpose(-1, -2), self.latent.unsqueeze(-1)).squeeze(-1)

        if diag:
            # only needs to compute the variance (diagonal matrix)
            row_col_iter = torch.arange(0, data_data_covar.matrix_shape[-1], dtype=torch.long)
            predictive_covar = gpytorch.lazy.DiagLazyTensor(
                data_data_covar.add_jitter(1e-4)[..., row_col_iter, row_col_iter])
            D = A.t().matmul(A)[..., row_col_iter, row_col_iter]
            predictive_covar.add_diag(D.mul(-1))
        else:
            predictive_covar = gpytorch.lazy.SumLazyTensor(
                data_data_covar.add_jitter(1e-4),
                gpytorch.lazy.MatmulLazyTensor(A.transpose(-1, -2), A.mul(-1)),
            )
        # Return the distribution
        return gpytorch.distributions.MultivariateNormal(predictive_mean, gpytorch.delazify(predictive_covar))
    
    
    def predictive_latent(self, x, diag=False):
        # Give the predictive distributions of the latent, only available in batch form
        train_inputs = self.train_inputs[0]
        batch_size, train_size, test_size, ndims = train_inputs.shape[0], train_inputs.shape[1], x.shape[0], \
                                                train_inputs.shape[-1]
        expanded_x = x.unsqueeze(0).repeat(batch_size, 1, 1)
        return self.conditioning(expanded_x, diag=diag)
    
    
    def predictive_main_effect_component(self, x, k=None, diag=False):
        if not(isinstance(self.likelihood, gpytorch.likelihoods.GaussianLikelihood)):
            raise NotImplementedError('Predictions are not implemented yet for non-gaussian likelihood.')
        # Give the predictive distributions of the main effect components k
        train_inputs = self.train_inputs[0]
        batch_size, train_size, test_size, ndims = train_inputs.shape[0], train_inputs.shape[1], x.shape[0], \
                                                train_inputs.shape[-1]
        joint_size = train_size + test_size
        
        expanded_x = x.unsqueeze(0).repeat(batch_size, 1, 1)

        # If k is None then provide all the main-effect_components
        if k is None:
            k = torch.arange(0, ndims)
        
        # If k is a number, return a tensor
        if isinstance(k, numbers.Number):
            k = torch.tensor([k])
            
        # If k is list or numpy, change to tensor
        if not(torch.is_tensor(k)):
            k = torch.tensor(k)
    
        predictive_density = dict()

        # Through main components
        for i in torch.arange(k.numel()):
            predictive_density[str(int(k[i]))] = self.main_effect_conditioning(expanded_x, k=int(k[i]), 
                                                                            diag=diag)
            

        return predictive_density
    
    
    def predictive_interaction_effect_component(self, x, k=None, diag=False):  # Only available in th batch mode
        # Give the predictive distribution of the first order interaction effect component between k[0] and k[1]
        
        if not(isinstance(self.likelihood, gpytorch.likelihoods.GaussianLikelihood)):
            raise NotImplementedError('Predictions are not implemented yet for non-gaussian likelihood.')
        
        train_inputs = self.train_inputs[0]
        batch_size, train_size, test_size, ndims = train_inputs.shape[0], train_inputs.shape[1], x.shape[0], \
                                                train_inputs.shape[-1]
        joint_size = train_size + test_size
        
        expanded_x = x.unsqueeze(0).repeat(batch_size, 1, 1)

        # If k is None then provide all the first_order-effect_components
        if k is None:
            k = torch.tensor([[i, j] for i in np.arange(0, ndims - 1)
                                    for j in np.arange(i + 1, ndims)], dtype=torch.long)
        
        # If k is list or numpy, change to tensor
        if not(torch.is_tensor(k)):
            k = torch.tensor(k)
        
        # If there only one target component, expand dimension
        if k.ndimension()==1:
            k = k.expand(1, -1)
        
        # Get the outputscale for the target component
        index = [first_order_InteractionEffect_to_index(ndims, list(j)) for j in k.numpy()]
        #print(index)

        # Compute the test part
        predictive_density = dict()

        # Through first_order_interaction components
        for i in torch.arange(len(index)):
            predictive_density[str(k[i].numpy())] = self.interaction_effect_conditioning(expanded_x,
                                                                                k=list(k[i].numpy()),
                                                                                diag=diag)

        return predictive_density


    def predictive_residual_effect_component(self, x, diag=False):  # Only available in th batch mode
        # Give the predictive distribution of the residual effect component
        
        if not(isinstance(self.likelihood, gpytorch.likelihoods.GaussianLikelihood)):
            raise NotImplementedError('Predictions are not implemented yet for non-gaussian likelihood.')
        
        train_inputs = self.train_inputs[0]
        batch_size, train_size, test_size, ndims = train_inputs.shape[0], train_inputs.shape[1], x.shape[0], \
                                                train_inputs.shape[-1]
        joint_size = train_size + test_size

        expanded_test_x = x.unsqueeze(0).repeat(batch_size, 1, 1)

        batch_inducing_points = self.inducing_points.unsqueeze(0).repeat(batch_size, 1, 1)
        
        
        
        # Compute the latent part
        induc_induc_covar = self.covar_module(batch_inducing_points).add_jitter()
        

        # Compute the test part
        covar_module = self.residualEffect
        data_data_covar = covar_module(expanded_test_x).add_jitter()
        induc_data_covar = covar_module(batch_inducing_points, expanded_test_x).evaluate()
        
        # Conditionning
        L = self.cholesky_factor(induc_induc_covar)
        # inv(L)*RZX
        A = L.inv_matmul(induc_data_covar.double()).to(self.__dtype)
        predictive_mean = torch.matmul(A.transpose(-1, -2), self.latent.unsqueeze(-1)).squeeze(-1)
        
        if diag:
            # only needs to compute the variance (diagonal matrix)
            row_col_iter = torch.arange(0, data_data_covar.matrix_shape[-1], dtype=torch.long)
            predictive_covar = gpytorch.lazy.DiagLazyTensor(
                data_data_covar.add_jitter(1e-4)[..., row_col_iter, row_col_iter])
            D = A.transpose(-1, -2).matmul(A)[..., row_col_iter, row_col_iter]
            predictive_covar.add_diag(D.mul(-1))
        else:
            predictive_covar = gpytorch.lazy.SumLazyTensor(
                data_data_covar.add_jitter(1e-4),
                gpytorch.lazy.MatmulLazyTensor(A.transpose(-1, -2), A.mul(-1)),
            )
        
        

        return gpytorch.distributions.MultivariateNormal(predictive_mean, gpytorch.delazify(predictive_covar))

       
