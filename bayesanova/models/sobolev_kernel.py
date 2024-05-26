import math

import gpytorch
import numpy as np
import torch
from gpytorch.constraints import Positive
from gpytorch.lazy import NonLazyTensor, lazify

from bayesanova.models.utils import bernoulli_poly


# Data is of size N*d
class VarBernPoly(gpytorch.kernels.Kernel):
    """This is the non-stationary part of the Sobolev kernel. 
        K1(s, t) = \sum_i B_i(s) B_i(t)/c_i where B_i 
        where B_i is the ith Bernoulli polynomial
        c_i = (i!)^2 if c is None 

    Args:
            poly_order (int, optional): polynomial order. Defaults to 2.
            c (float or tensor, optional): a constant to wight the kernel. Defaults to None.
    """
    is_stationary = False

    def __init__(self, poly_order=2, c=None, **kwargs):
        super().__init__(**kwargs)
        self.poly_order = poly_order
        if c is None:
            c = 1 / (math.factorial(self.poly_order) ** 2)
        self.c = c

    def forward(self, x1, x2, last_dim_is_batch=False, **params):
        if last_dim_is_batch is False:
            assert x1.shape[-1] == 1
            assert x1.shape[-1] == 1
            bx1 = bernoulli_poly(x1, self.poly_order).squeeze()
            bx2 = bernoulli_poly(x2, self.poly_order).squeeze()
        else:
            bx1 = bernoulli_poly(x1, self.poly_order).transpose(-1, -2)
            bx2 = bernoulli_poly(x2, self.poly_order).transpose(-1, -2)
        BX1 = bx1.unsqueeze(-1)
        BX2 = bx2.unsqueeze(-2)
        K = BX1 * BX2
        return lazify(K * self.c)


class StatBernPoly(gpytorch.kernels.Kernel):
    """This is the stationary part of the Sobolev kernel. 
        K2(s, t) = B_2i(|s-t|)/(-1)^(i+1)/(2i)!/c
        where B_i is the ith Bernoulli polynomial

    Args:
            poly_order (int, optional): polynomial order. Defaults to 2.
            c (float or tensor, optional): a constant to wight the kernel. Defaults to None.
    """
    is_stationary = True

    def __init__(self, poly_order=2, c=None, **kwargs):
        super().__init__(**kwargs)
        self.poly_order = poly_order
        if c is None:
            c = 1
        self.c = c

    def forward(self, x1, x2, **params):
        diff = self.covar_dist(x1, x2, **params)
        r = self.poly_order * 2
        K = bernoulli_poly(diff, r) * ((-1) ** (self.poly_order + 1) / math.factorial(r))
        return lazify(K * self.c)


class ConstantKernel(gpytorch.kernels.Kernel):
    """This is the kernel for the RKHS for constant functions
    """
    is_stationary = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

    def forward(self, x1, x2, last_dim_is_batch=False, **params):
        # x1 : ....\times N \times D  
        # x2 : .....  \times M  \times D
        if last_dim_is_batch:
            v1 = torch.ones(x1.transpose(-1, -2).shape).unsqueeze(-1)
            v2 = torch.ones(x2.transpose(-1, -2).shape).unsqueeze(-2)
        else:
            v1 = torch.ones(x1.shape[:-1]).unsqueeze(-1)
            v2 = torch.ones(x2.shape[:-1]).unsqueeze(-2)
        return gpytorch.lazify(v1).matmul(gpytorch.lazify(v2))
        

class FullKernel(gpytorch.kernels.Kernel):
    """This module compute the full covariance module from a given base kernel
    full_covariance(s,t) = (1+k(s, t)) for a scalar s and t 
    k is the base kernel and 1 refers to the constant functional component 
    Args:
        base_kernel (gpytorch.kernel): base kernel to compute the covariance

    """
    is_stationary = False

    def __init__(self, base_kernel, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel 

    def forward(self, x1, x2, **params):
        K = self.base_kernel(x1, x2, **params) 
        C = ConstantKernel().forward(x1, x2, **params)
        full = K + C
        # We have to delazify the output otherwise there is a gpytorch error on root decomposition
        return full
    
    def __call__(self, x1_, x2_=None, **params):
        """
        We cannot lazily evaluate this kernel otherwise we get
        a gpytorch error on root decomposition
        """
        res = super().__call__(x1_, x2_, **params)
        res = gpytorch.delazify(res)
        return res 



class ResidualKernel(gpytorch.kernels.Kernel):
    """This is the residual kernel for the RKHS minus the main effect and the first order interaction effect (and the constant).
        the kernel is given by:
        K = base_covariance-constant_covariance-active_covariance
        base_covariance(s,t) = prod_i full_covar(s_i, t_i) =  prod_i(1+k(s_i, t_i))
        active covariance is the sum of main effect (and first order iteraction effect)

        Args:
            base_kernel (gpytorch.kernel): this is the kernel of the base covariance
            active_kernel (gpytorch.kernel, tuple of gpytorch.kernel): a tuple containing all the active kernels 
    """
    is_stationary = False

    def __init__(self, base_kernel, active_kernel, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        if not isinstance(active_kernel, tuple):
            if issubclass(type(active_kernel), gpytorch.kernels.Kernel):
                active_kernel = (active_kernel,)
            else:
                raise ValueError(
                    "Active kernel should be a gpytorch.kernels.Kernel or a tuple of gpytorch.kernels.Kernel")
        self.active_kernel = active_kernel

    def forward(self, x1, x2, **params):
        K0 = self.base_kernel(x1, x2, **params)
        K1 = gpytorch.lazy.ZeroLazyTensor(1)
        # First, compute active effects
        for i in range(len(self.active_kernel)):
            K1 = K1 + self.active_kernel[i](x1, x2, **params)
        # Then, compute constant component
        C = ConstantKernel().forward(x1, x2, **params)
        # Compute the resulting active covariance
        K1 = K1 + C
        # Finally, remove the active covariance from full kernel
        K = K0 - K1
        return K


class ScaleKernel(gpytorch.kernels.ScaleKernel):
    """This module is a slightly modified version of the gpytorch ScaleKernel in order to be used
    With the ScaleAdditiveStructureKernel class
    Modifications are : the size of outputscale + possibility to normalize it with likelihood.noise

    Args:
        base_kernel (gpytorch.kernels): base kernel
        outputscale_prior (gpytorch.priors, optional): prior for outputscale. Defaults to None.
        outputscale_constraint (gpytorch.constraint, optional): constraint for outputsale. Defaults to None.
        likelihood (gpytorch.lieklihood, optional): likelihood, needed if we want to normalize by the noise. Defaults to None.
    """

    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if base kernel is stationary.
        """
        return self.base_kernel.is_stationary

    def __init__(self, base_kernel, outputscale_prior=None, outputscale_constraint=None, likelihood=None, **kwargs):
        if base_kernel.active_dims is not None:
            kwargs["active_dims"] = base_kernel.active_dims
        super(ScaleKernel, self).__init__(base_kernel, outputscale_prior, outputscale_constraint, **kwargs)
        if outputscale_constraint is None:
            outputscale_constraint = Positive()

        self.base_kernel = base_kernel
        self.likelihood = likelihood
        outputscale = torch.zeros(*self.batch_shape, 1) if len(self.batch_shape) else torch.tensor([0.0])
        self.register_parameter(name="raw_outputscale", parameter=torch.nn.Parameter(outputscale))
        if outputscale_prior is not None:
            self.register_prior(
                "outputscale_prior", outputscale_prior, lambda: self.outputscale, lambda v: self._set_outputscale(v)
            )

        self.register_constraint("raw_outputscale", outputscale_constraint)

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, value):
        self._set_outputscale(value)

    def _set_outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))

    def forward(self, x1, x2, last_dim_is_batch=False, diag=False, **params):
        orig_output = self.base_kernel.forward(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        
        outputscales = self.outputscale.squeeze(-1)
        if self.likelihood is not None:
            outputscales = outputscales*self.likelihood.noise
        if last_dim_is_batch:
            outputscales = outputscales.unsqueeze(-1)
        if diag:
            outputscales = outputscales.unsqueeze(-1)
            return gpytorch.delazify(orig_output) * outputscales
        else:
            outputscales = outputscales.view(*outputscales.shape, 1, 1)
            # We had to apply delazify because we get a dismatch 
            # between lazy tensor dimension and true tensor dimension
            return lazify(gpytorch.delazify(orig_output).mul(outputscales))

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)




class AdditiveStructureKernel(gpytorch.kernels.AdditiveStructureKernel):
    """This function is a slightly modified version of the gpytorch AdditiveStructureKernel in order to be used With
    the ScaleAdditiveStructureKernel class. Modifications are in the forward function. Given some base kernel k and inputs
    X, Y of size D,  we compute the Gram matrix of the base kernel of Main Effect/Interaction effect. For Main
    effect : R(X,Y) = \sum_i K(X_i, Y_i). For interaction effects  of first order, R(X,Y)=\sum_{i<j} K(X_i,
    Y_i) K(X_j, Y_j).
    The base kernel can be a unique kernel or tuple of kernel from gpytorch.kernels.Kernel.
    In the latter case, the kernel K used to compute the Gram matrix is the sum of Kernels in the tuple
    Args:
            base_kernel (gpytorch.kernels, tuple of gpytorch.kernels): base kernel
            num_dims (int): number of dimensions
            active_dims (int, optional): the active dimension of the data. Defaults to None.
            batch_shape (torch.Size, optional): batch shape. Defaults to torch.Size([]).
    """
    
    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if base kernel is stationary.
        """
        return self.base_kernel[0].is_stationary

    def __init__(self, base_kernel, num_dims, active_dims=None, batch_shape=torch.Size([])):
        if not isinstance(base_kernel, tuple):
            if issubclass(type(base_kernel), gpytorch.kernels.Kernel):
                base_kernel = (base_kernel,)
            else:
                raise ValueError(
                    "Base kernel should be a gpytorch.kernels.Kernel or a tuple of gpytorch.kernels.Kernel")
        super(AdditiveStructureKernel, self).__init__(base_kernel, num_dims, active_dims=active_dims)
        self.batch_shape = batch_shape
        self.base_kernel = base_kernel

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("AdditiveStructureKernel does not accept the last_dim_is_batch argument.")
        D = x1.size(-1)
        if self.num_dims == D:  # Main effect
            res = gpytorch.lazy.ZeroLazyTensor(1)
            for k in range(len(self.base_kernel)):
                res = res + self.base_kernel[k](x1, x2, diag=diag, last_dim_is_batch=True, **params)
        else:
            if self.num_dims == D * (D - 1) / 2:  # First order interaction effect
                res = gpytorch.lazy.ZeroLazyTensor(1)
                index = torch.tensor([[i, j] for i in np.arange(0, D - 1)
                                    for j in np.arange(i + 1, D)], dtype=torch.long)
                for k in range(len(self.base_kernel)):
                    res0 = self.base_kernel[k](x1, x2, diag=diag, last_dim_is_batch=True, **params)
                    if diag:
                        res = res + res0[..., index[:, 0], :] * res0[..., index[:, 1], :]
                    else:
                        res = res + res0[..., index[:, 0], :, :] * res0[..., index[:, 1], :, :]
            else:  # Higher order interaction effect
                res = None
                # To be continued
        res = res.sum(-2 if diag else -3)
        return res

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return self.base_kernel[0].prediction_strategy(train_inputs, train_prior_dist, train_labels, likelihood)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel[0].num_outputs_per_input(x1, x2)



class ScaleAdditiveStructureKernel(gpytorch.kernels.AdditiveStructureKernel):
    """For some kernel k and inputs X, Y of size D,  we compute the Gram matrix of the base kernel of Main
    Effect/Interaction effect. For Main effect : R(X,Y) = \sum_i \theta_i K(X_i, Y_i). For interaction effects  of
    first order, R(X,Y)=\sum_{i<j} \theta_ij K(X_i, Y_i) K(X_j, Y_j) The base kernel can be a unique kernel or tuple
    of kernel from gpytorch.kernels.Kernel. In the latter case, the kernel K used to compute the Gram matrix is the
    sum of Kernels in the tuple base_kernel is can be either a unique kernel or tuple of kernels to be added the
    output scale \theta_k is  parameterized on a log scale to constrain it to be positive

    Args:
        base_kernel (gpytorch.kernels, tuple of gpytorch.kernels): base kernel
        num_dims (int): number of dimensions
        active_dims (int, optional): the active dimension of the data. Defaults to None.
        outputscale_prior (gpytorch.priors, optional): prior for outputscale. Defaults to None.
        outputscale_constraint (gpytorch.constraint, optional): constraint for outputsale. Defaults to None.
        batch_shape (torch.Size, optional): batch shape. Defaults to torch.Size([]).
        normalization_coef (float or tensor, optional): the weighting coefficient. Defaults to None.
    """
    

    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if base kernel is stationary.
        """
        return self.base_kernel[0].is_stationary

    def __init__(self, base_kernel, num_dims, active_dims=None, outputscale_prior=None, outputscale_constraint=None,
                batch_shape=torch.Size([]), normalization_coef=None):
        if not isinstance(base_kernel, tuple):
            if issubclass(type(base_kernel), gpytorch.kernels.Kernel):
                base_kernel = (base_kernel,)
            else:
                raise ValueError(
                    "Base kernel should be a gpytorch.kernels.Kernel or a tuple of gpytorch.kernels.Kernel")
        super(ScaleAdditiveStructureKernel, self).__init__(base_kernel, num_dims, active_dims=active_dims)
        self.batch_shape = batch_shape
        self.base_kernel = base_kernel
        self.normalization_coef = normalization_coef
        if outputscale_constraint is None:
            outputscale_constraint = Positive()
        outputscale = torch.zeros(*self.batch_shape, num_dims) if len(self.batch_shape) else torch.zeros(num_dims)
        self.register_parameter(name="raw_outputscale", parameter=torch.nn.Parameter(outputscale))

        if outputscale_prior is not None:
            self.register_prior(
                "outputscale_prior", outputscale_prior, lambda m: m.outputscale, lambda m, v: m._set_outputscale(v)
            )

        self.register_constraint("raw_outputscale", outputscale_constraint)

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, value):
        self._set_outputscale(value)

    def _set_outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("ScaleAdditiveStructureKernel does not accept the last_dim_is_batch argument.")
        D = x1.size(-1)
        if self.num_dims == D:  # Main effect
            res = gpytorch.lazy.ZeroLazyTensor(1)
            for k in range(len(self.base_kernel)):
                res = res + self.base_kernel[k](x1, x2, diag=diag, last_dim_is_batch=True, **params)
        else:
            if self.num_dims == D * (D - 1) / 2:  # First order interaction effect
                res = gpytorch.lazy.ZeroLazyTensor(1)
                index = torch.tensor([[i, j] for i in np.arange(0, D - 1)
                                    for j in np.arange(i + 1, D)], dtype=torch.long)
                for k in range(len(self.base_kernel)):
                    res0 = self.base_kernel[k](x1, x2, diag=diag, last_dim_is_batch=True, **params)
                    if diag:
                        res = res + res0[..., index[:, 0], :] * res0[..., index[:, 1], :]
                    else:
                        res = res + res0[..., index[:, 0], :, :] * res0[..., index[:, 1], :, :]
            else:  # Higher order interaction effect
                res = None
                # To be continued
        outputscales = self.outputscale
        if self.normalization_coef is not None:
            outputscales = outputscales*self.normalization_coef
        outputscales = outputscales.view(*outputscales.shape, 1, 1)
        res = res.mul(outputscales)
        res = res.sum(-2 if diag else -3)
        return res

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel[0].num_outputs_per_input(x1, x2)

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return self.base_kernel[0].prediction_strategy(train_inputs, train_prior_dist, train_labels, likelihood)




def univ_main_effect(poly_order, c=None, **kwargs):
    """This function computes the covariance module of a main effect component for univariate data. 

    Args:
        poly_order (int): polynomial order
        c (float or tensor, optional): a constant to weight the non-stationary kernel. Defaults to None.

    Returns:
        covar_module
    """
    if poly_order == 1:
        covar_module = StatBernPoly(poly_order=1, **kwargs) + VarBernPoly(poly_order=1, c=c, **kwargs)
    elif poly_order == 2:
        covar_module = StatBernPoly(poly_order=2, **kwargs) + VarBernPoly(poly_order=2, c=c, **kwargs) + VarBernPoly(
            poly_order=1, c=c, **kwargs)
    else:
        covar_module = None
    return covar_module


def univ_first_order_interaction_effect(poly_order, c=None, correction=False, **kwargs):
    """This function computes the covariance module of the first order interaction effect component

    Args:
        poly_order (int): polynomial order
        c (float or tensor, optional): a constant to wight the non-stationary kernel. Defaults to None.
        correction (bool, optional): if true apply the correction as in [Reich, 2009]. Defaults to False.

    Returns:
        covar module
    """
    if correction is True and c is not None and c >= 1:
        covar_module = univ_main_effect(poly_order, active_dims=torch.tensor([0]), c=1, **kwargs) * \
                       univ_main_effect(poly_order, active_dims=torch.tensor([1]), c=1, **kwargs) - \
                       StatBernPoly(poly_order=2, active_dims=torch.tensor([0]), c=c - 1, **kwargs) * \
                       StatBernPoly(poly_order=2, active_dims=torch.tensor([1]), **kwargs)
    else:
        covar_module = univ_main_effect(poly_order, active_dims=torch.tensor([0]), **kwargs) * \
                       univ_main_effect(poly_order, active_dims=torch.tensor([1]), **kwargs)
    return covar_module


def main_effect(num_dims, poly_order=1):
    """This function gives the main effect covariance computed on a multidimensional data

    Args:
        num_dims (int): number of dimensions
        poly_order (int, optional): polynomial order. Defaults to 1.

    Returns:
        covar module
    """
    base_kernel = univ_main_effect(poly_order)
    covar_module = ScaleAdditiveStructureKernel(base_kernel, num_dims)
    return covar_module
