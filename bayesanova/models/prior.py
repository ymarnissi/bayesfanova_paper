from abc import ABC

import gpytorch
import torch
from pyro.distributions.transforms.normalize import Normalize
from torch.distributions import constraints
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.exponential import Exponential
from torch.distributions.gamma import Gamma
from torch.distributions.half_cauchy import HalfCauchy
from torch.distributions.transformed_distribution import \
    TransformedDistribution
from torch.distributions.transforms import PowerTransform
from torch.nn import Module as TModule


class InverseGamma(TransformedDistribution):
    """Create an Inverse Gamma parametrized such that X~Gamma(a,b)
        then 1/X ~InvGamma(a,b)
        Args:
            concentration (float or torch.tensor): concentration parameter
            rate (float or torch.tensor): rate parameter
            
    """

    arg_constraints = {'concentration': constraints.positive, 'rate': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, concentration, rate, validate_args=None):
        base_dist = Gamma(concentration, rate, validate_args=False)
        super(InverseGamma, self).__init__(base_dist, PowerTransform(-1), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(InverseGamma, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(InverseGamma, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def concentration(self):
        return self.base_dist.concentration

    @property
    def rate(self):
        return self.base_dist.rate

    @property
    def mean(self):
        return self.concentration/(self.rate-1)

    @property
    def variance(self):
        return torch.pow(self.concentration, torch.tensor(2))/(torch.pow((self.rate-2), torch.tensor(2))*(self.rate-2))


class InverseGammaPrior(gpytorch.priors.prior.Prior, InverseGamma):
    """Inverse Gamma Prior to be used with gpytorch parameterized by concentration and rate

    pdf(x) = beta^alpha / Gamma(alpha) * x^(-alpha + 1) * exp(-beta / x)

    were alpha > 0 and beta > 0 are the concentration and rate parameters, respectively.
    Args:
            concentration (float or torch.tensor): concentration parameter
            rate (float or torch.tensor): rate parameter
    """

    def __init__(self, concentration, rate, validate_args=False, transform=None):
        TModule.__init__(self)
        InverseGamma.__init__(self, concentration=concentration, rate=rate, validate_args=validate_args)
        #_bufferize_attributes(self, ("concentration", "rate"))
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return InverseGammaPrior(self.concentration.expand(batch_shape), self.rate.expand(batch_shape))



class HalfCauchyPrior(gpytorch.priors.prior.Prior, HalfCauchy):
    """Half Cauchy Prior to be used with gpytorch parameterized by the scale
    p(x)=1/pi*gamma*(1+x^2/gamma^2)
    scale = gamma

    Args:
        scale (float or tensor): scale parameter
    """

    def __init__(self, scale, validate_args=False, transform=None):
        TModule.__init__(self)
        HalfCauchy.__init__(self, scale=scale, validate_args=validate_args)
        #_bufferize_attributes(self, ("concentration", "rate"))
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return HalfCauchyPrior(self.scale.expand(batch_shape))




class HalfHalfCauchy(TransformedDistribution):
    """Create a HalfHalfCauchy X~HHC then \sqrt(x) ~HC

    Args:
            scale (float or tensor): scale of the halfcauchy
    """
    arg_constraints = {'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, scale, validate_args=None):
        base_dist = HalfCauchy(scale, validate_args=False)
        super(HalfHalfCauchy, self).__init__(base_dist, PowerTransform(2),
                                        validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(HalfHalfCauchy, _instance)
        batch_shape = torch.Size(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(HalfHalfCauchy, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def scale(self):
        return self.base_dist.scale


class HalfHalfCauchyPrior(gpytorch.priors.prior.Prior, HalfHalfCauchy):
    """"Half Half Cauchy Prior to be used in Gpytorch parameterized by the scale

    Args:
            scale (float or tensor): scale of the halfcauchy
    """

    def __init__(self, scale, validate_args=False, transform=None):
        TModule.__init__(self)
        HalfHalfCauchy.__init__(self, scale=scale, validate_args=validate_args)
        #_bufferize_attributes(self, ("concentration", "rate"))
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return HalfHalfCauchyPrior(self.scale.expand(batch_shape))



class StudentMixingPrior(InverseGammaPrior):
    """Student (or t-distribution) hierarchical model parametrized by 
    p(fj) \propto (1+1/nu * fj.T*inv(Rj*gscale)*f_j) ^(nu+d)/2
    nu is the freedom parameter 
    gscale is the scale
    
    In hierchical modeling, fj ~N(0, s*Rj*gscale ) where s ~InvGamma(nu/2, nu/2*gscale)

    Args:
            freedom (tensor or float): degre of freedom
            scale (tensor or float): scale parameter. Defaults to torch.tensor([1]).
    """
    def __init__(self, freedom,  scale=torch.tensor([1]), validate_args=False, transform=None):
        TModule.__init__(self)
        self.freedom = freedom
        self.gscale = scale
        InverseGammaPrior.__init__(self, concentration=freedom/2, rate=freedom/2*self.gscale, validate_args=validate_args, transform=transform)




class LaplaceMixingPrior(gpytorch.priors.prior.Prior, Exponential):
    """Laplace hierarchical model parametrized by 
    p(fj) = exp(-1/sqrt(lambda)*1/sqrt(gscale)*\sqrt(fj.T*inv(Rj)*f_j) then
    p(gamma_j) = Exp(1/(2*gscale)) and p(x|gamma_j) ~N(0, lambda*gamma_j*R_j)
    gscale can be estimated as ||f_j||^2

    Args:
        scale (tensor or float): scale parameter. Defaults to torch.tensor([1]).
    """
    def __init__(self, scale=torch.tensor([1.0]), validate_args=False, transform=None):
        TModule.__init__(self)
        self.gscale = scale
        Exponential.__init__(self, 1/(2*self.gscale), validate_args=validate_args)
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return LaplaceMixingPrior(self.gscale.expand(batch_shape))
    
    

class HoreshoeMixingPrior(gpytorch.priors.prior.Prior, HalfHalfCauchy):
    """Horeshoe hierchical model parametrized by 
    p(gamma_j) = HalfCauchy(gscale) and p(x|gamma_j) ~N(0, lambda*gamma_j*R_j)
    
    There is no explicit formula for the marginalized density.
        Args:
            scale (tensor or float): scale parameter. Defaults to torch.tensor([1]).
    """
    def __init__(self, scale=torch.tensor([1.0]), validate_args=False, transform=None):
        TModule.__init__(self)
        self.gscale = scale
        HalfHalfCauchy.__init__(self, self.gscale, validate_args=validate_args)
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return HoreshoeMixingPrior(self.gscale.expand(batch_shape))




class DirichletMixingPrior(gpytorch.priors.prior.Prior, Dirichlet, ABC):
    """Diricklet hierchical model parametrized by 
    p(gamma) = Dirichlet(gscale) and p(fj|gamma_j) ~N(0, lambda*gamma_j*R_j)
    The scale parameters are then dependent.  
    Support = simplex
    
    There is no explicit formula for the marginalized density.
        Args:
            scale (tensor or float): scale parameter. Defaults to torch.tensor([1]).
    """
    support = constraints.simplex
    _validate_args = True
    def __init__(self, scale, validate_args=False, transform=Normalize(p=1)):
        TModule.__init__(self)
        self.gscale = scale
        Dirichlet.__init__(self, concentration=scale, validate_args=validate_args)
        #_bufferize_attributes(self, ("concentration", "rate"))
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return DirichletMixingPrior(self.gscale.expand(batch_shape))




# TODO: Add explicitly constraints on dirichlet outputscale
class SimplexConstraintTransform(torch.nn.Module):
    """This is a transform to be called as gyptorch.constraints
    It defines a transform into a simplex i.e sum(paramerter value) = 1
    parameter should be positive
    
    This can be used to define the transform method For Dirichlet 
    and to normalize the outputscales for any other prior


    Args:
        torch (_type_): _description_
    """
    def __init__(self) -> None:
        super().__init__()
        # TO DO : define a transform into a simplex 
    
    def __call__(self, x):
        raise NotImplementedError
    
    
        
    


    


        
    