import gpytorch
import torch


class JointLogLikelihood(gpytorch.Module):
    """This module computes the joint log Gaussian likelihood
        of the GP model when applied to data.  i.e., given a latent f(X) with \math:`f \sim
        \mathcal{GP}(u, K)` with parameters \theta , y \sim N(\mu, \sigma^2), and data :math:`\mathbf X, \mathbf y
        this module computes
        \begin{equation*}
            \mathcal{L} = log p(y|f(X),\sigma^2) + log p(\theta, u, \sigma^2)+log(p(f(X))
        \end{equation*}
        Example for likelihood Gaussian
        log p(y|f(X),\sigma^2) = - 1/(2*\sigma^2)*||y_f(X) ||^2
                                                    - 1/(2*\sigma^2) * trace(R_XX-R_XZ R_ZZ^-1 R_ZX)
                                                    -1/2 log p(f(X))

    Args:
            likelihood (gpytorch likelihood): the likelihood
            model (gpytorch model): the gaussian process model
    """

    def __init__(self, likelihood, model):
        super(JointLogLikelihood, self).__init__()
        self.likelihood = likelihood
        self.model = model

    def _add_other_terms(self, res, params):
        # Add additional terms (SGPR / learned inducing points, heteroskedastic likelihood models)
        for added_loss_term in self.model.added_loss_terms():
            res = res.add(added_loss_term.loss(*params))

        # Add log probs of priors on the (functions of) parameters
        for name, module, prior, closure, _ in self.named_priors():
            res.add_(prior.log_prob(closure(module)).sum())

        return res

    def forward(self, f, target, *params):
        # Get the conditional liklihood p(y|f)
        if not isinstance(f, torch.Tensor):
            raise RuntimeError("JointLogLikelihood can only operate on tensors")

        output = self.likelihood(f, *params)
        res = output.log_prob(target)

        res = self._add_other_terms(res, params)

        # Scale by the amount of data we have
        num_data = output.event_shape.numel()
        return res.div_(num_data)

    def pyro_factor(self, output, target, *params):
        import pyro

        ell = target.size(-1) * self(output, target, *params)
        pyro.factor("gp_ell", ell)
        return ell


class ExpectedLogLikelihood(gpytorch.Module):
    """This module computes the expected log Gaussian likelihood
        of the GP model when applied to data.  i.e., given a GP :math:`f \sim
        \mathcal{GP}(u, K)` with parameters \theta , y \sim N(\mu, \sigma^2), and data :math:`\mathbf X, \mathbf y
        and inducing points \mathbf Z where u=f(Z)
        this module computes: 
        \begin{equation*}
            \mathcal{L} = E_p(f|u,\theta) [log p(y|u,\sigma^2)] + log p(\theta, u, \sigma^2)
        \end{equation*}
        Example for likelihood Gaussian:
        E_p(f|u,\theta) [log p(y|u,\sigma^2)] = - 1/(2*\sigma^2)*||y_\mu-R_XZ R_ZZ^-1 R_ZX u ||^2
                                                    - 1/(2*\sigma^2) * trace(R_XX-R_XZ R_ZZ^-1 R_ZX)

    Args:
            likelihood (gpytorch likelihood): the likelihood
            model (gpytorch model): the gaussian process model
    """

    def __init__(self, likelihood, model):
        super(ExpectedLogLikelihood, self).__init__()
        self.likelihood = likelihood
        self.model = model

    def _add_other_terms(self, res, params):
        # Add additional terms (SGPR / learned inducing points, heteroskedastic likelihood models)
        for added_loss_term in self.model.added_loss_terms():
            res = res.add(added_loss_term.loss(*params))

        # Add log probs of priors on the (functions of) parameters
        for name, module, prior, closure, _ in self.named_priors():
            res.add_(prior.log_prob(closure(module)).sum())

        return res

    def forward(self, function_dist, target, *params):
        # Get the expected log prob of the marginal distribution
        res = self.likelihood.expected_log_prob(target, function_dist).sum()

        res = self._add_other_terms(res, params)

        # Scale by the amount of data we have
        num_data = function_dist.event_shape.numel()
        return res.div_(num_data)

    def pyro_factor(self, output, target, *params):
        import pyro

        ell = target.size(-1) * self(output, target, *params)
        pyro.factor("gp_ell", ell)
        return ell