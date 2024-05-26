import math
import numbers
#import cvxopt
import numpy as np
import torch
from sklearn.linear_model import Lasso

from bayesanova.anova_estimators.ss import SS
from bayesanova.anova_estimators.utils import k_fold, get_adaptive_scale_from_covariance_trace
from bayesanova.models import sobolev_kernel
from bayesanova.models.utils import first_order_InteractionEffect_to_index 

'''  This version does not include residual components !'''
'''  This version includes only lasso solver !'''

class COSSO(torch.nn.Module):
    """Component Selection Smoothing Spline method

    Args:
        ndims (int): dimension size 
        model_order (int) : model order
        poly_order (int) : polynomial order 
    """
    def __init__(self, ndims, model_order=1, poly_order=1, weight=None):
        super(COSSO, self).__init__()

        base_kernel = sobolev_kernel.univ_main_effect(poly_order, c=None) 
    
        self.base_kernel = base_kernel
        self.model_order = model_order
        self.poly_order = poly_order
        self.ndims = ndims
        # setting weights
        if weight is None:
            if self.model_order==1:
                self.weight = (torch.ones(ndims), )
            else:
                self.weight = (torch.ones(ndims), torch.ones(ndims * (ndims - 1) // 2))
        else:
            self.weight = weight
        
        if self.weight[0] is None:
            self.weight[0] = torch.ones(ndims)
        if (self.model_order==2) and (self.weight[1] is None):
            self.weight[1] = torch.ones(ndims * (ndims - 1) // 2)
            
        
        if self.model_order == 1:
            covar_module = sobolev_kernel.ScaleAdditiveStructureKernel(base_kernel, ndims)
            covar_module.outputscale = torch.ones(ndims, 1)
            self.covar_module = covar_module
        else:
            self.main_covar_module = sobolev_kernel.ScaleAdditiveStructureKernel(base_kernel, ndims)
            self.interaction_covar_module = sobolev_kernel.ScaleAdditiveStructureKernel(base_kernel,
                                                                                        ndims * (ndims - 1) // 2)
            self.main_covar_module.outputscale = torch.ones(ndims, 1)
            self.interaction_covar_module.outputscale = torch.ones(ndims * (ndims - 1) // 2, 1)
            self.covar_module = self.main_covar_module + self.interaction_covar_module
        self.train_x = None
        self.train_y = None
        self.coef = None
        self.ss_weight = None
        
        
    def initialize(self):
        """This method initialize to COSSO model to clear cache
        """
        self.coef = None
        self.train_x = None
        self.train_y = None
        self.ss_weight = None
        
        if self.model_order==1:
            self.covar_module.outputscale = torch.ones(self.ndims, 1)
        else:
            self.main_covar_module.outputscale = torch.ones(self.ndims, 1)
            self.interaction_covar_module.outputscale = torch.ones(self.ndims*(self.ndims-1)//2, 1)
        
            

    def train_step(self, train_x, train_y, alpha=torch.tensor((0.01)), M=0.5, GCV=False, 
                tol=0.0001, solver='lasso', max_iter=100):
        """This method trains the COSSO model using train data (train_x, train_y)
        given a scalar smoothing parameter alpha and a selection regularization M

        Args:
            train_x (tensor): train inputs
            train_y (tensor): train outcome
            alpha (float, tensor): smoothing parameter. Defaults to torch.tensor((0.01)).
            GCV (bool, optional): either use GCV or not. Defaults to False.
            M (float, optional): regularization parameter for selection parameter. Defaults to 0.5.
            tol (float, optional): tolorance for solver. Defaults to 0.0001.
            solver (str, optional): solver for selection parameter. Defaults to 'lasso'.
            max_iter (int, optional) : maximum number of iteration in solver

        Returns:
            tensor: the estimated train_y
        """
        N = train_y.shape[0]
        M = M.item()
        # Initialize the covariance 
        self.initialize()
        # Train the SS estimator
        ss_estimator = SS(ndims=self.ndims, model_order=self.model_order, poly_order=self.poly_order)
        ss_estimator.covar_module = self.covar_module
        ss_estimator.train_step(train_x, train_y, alpha=alpha, GCV=False)
        self.coef = ss_estimator.coef
        # Solve the Lasso problem on selection parameter 
        z = train_y - 1 / 2 * N * alpha * self.coef[:-1] - self.coef[-1]
        G0 = self.get_components(train_x)
        if not(isinstance(G0, tuple)):
            G0 = (G0,)
        G = [i.T for i in G0]
        G = torch.cat(G, dim=1)
        # If weight is not None, this turns to the ACOSSO solution
        # Instead of changing the solver, we use the same solver 
        # but we weight to selection variables
        weight = torch.cat([i for i in self.weight], dim=0)
        G = torch.matmul(G, torch.diag(weight))
        
        if solver == 'lasso':
            self.ss_weight = Lasso(alpha=M/2, fit_intercept=False,
                                positive=True, normalize=False,
                                tol=tol, max_iter=max_iter, selection='random').fit(G.detach().numpy(),
                                                                            z.detach().numpy()).coef_
        else:
            raise NotImplementedError('Only Lasso is implemented for the moment')
        
        if self.model_order== 1:
            self.covar_module.outputscale = self.ss_weight*self.weight[0].detach().numpy()
        else:
            self.main_covar_module.outputscale = self.ss_weight[:self.ndims]*self.weight[0].detach().numpy()
            self.interaction_covar_module.outputscale = self.ss_weight[self.ndims:]*self.weight[1].detach().numpy()
        
        # Train again the SS estimator (weighted SS)    
        ss_estimator.covar_module = self.covar_module
        output = ss_estimator.train_step(train_x, train_y, alpha=alpha, GCV=GCV)
        self.coef = ss_estimator.coef
        return output

    def __train_test_step(self, train_x, train_y, test_x, test_y, alpha, M, loss, 
                            tol=0.0001, solver='lasso', max_iter=100):
        """This is a train + test steps

        Args:
            train_x (tensor): train inputs
            train_y (tensor): train outcome
            test_x (tensor): test inputs
            test_y (tensor): test outcome
            l (float, tensor): smoothing parameter. Defaults to torch.tensor((0.01)).
            loss (function): this is the loss to optimize the smoothing parameter.
                            Defaults to None.
            tol (float, optional): tolorance for solver. Defaults to 0.0001.
            solver (str, optional): solver for selection parameter. Defaults to 'lasso'.
            max_iter (int, optional) : maximum number of iteration in solver

        Returns:
            tensor : loss value in test data
        """
        self.initialize()
        try:
            self.train_step(train_x, train_y, alpha=alpha, GCV=False, M=M, tol=tol, 
                            solver=solver, max_iter=max_iter)
            output = self.__predict_step(train_x, test_x)
            loss_val = loss(output , test_y) 
        except:
            loss_val = torch.tensor([math.inf])
        return loss_val
    
    
    def train(self, train_x, train_y, GCV=False, M=torch.tensor((0.5)), alpha=torch.tensor((0.01)),
            folds=4, loss=None, solver='lasso', tol=0.0001, max_iter=100, random_state=12345): 
        """This is a general training method of the COSSO model using train data (train_x, train_y)
        given either a scalar regualarization parameter M or a set of smoothing parameters to choose
        the optimal value with cross-validation 

        Args:
            train_x (tensor): train inputs
            train_y (tensor): train outcome
            GCV (bool, optional): either use GCV or not. Defaults to False.
            folds (int, optional): number of folds in cross-validation. Defaults to 4.
            M (float, tensor): regularization parameter. Defaults to torch.tensor((0.5)).
            alpha (float, tensor): smoothing parameter. Defaults to torch.tensor((0.01)).
            loss (function, optional):  this is the loss to optimize the smoothing parameter.
                                        Defaults to None. If None then quadratic.
            tol (float, optional): tolorance for solver. Defaults to 0.0001.
            solver (str, optional): solver for selection parameter. Defaults to 'lasso'.

        Returns:
            float or tensor, float or tensor: optimal alpha, optimal loss
        """
        
        loss_min = None
        
        if not(torch.is_tensor(M)):
            M = torch.tensor((M))
        
        if M.ndimension()>0:
            # First define the loss if not defined
            if loss is None:
                loss = lambda output, test_y : torch.mean((output - test_y) ** 2)
            # Define the training strategy  
            train_stratgey = lambda  x1, y1, x2, y2, l: self.__train_test_step(x1, y1, x2, y2, M=l,
                                                                            alpha=alpha, loss=loss, 
                                                                            tol=tol, solver=solver, 
                                                                            max_iter=max_iter)
            # Find the optimal regularization parameter
            lhat, loss_min = k_fold(train_x, train_y, M, train_stratgey, folds=folds, 
                                    random_state=random_state)
            
        else:
            lhat = M
            
        self.initialize()
        # Optimize COSSO with respect to train data with the optimal regularization parameter    
        self.train_step(train_x, train_y, alpha=alpha, M=lhat, GCV=GCV, tol=tol, solver=solver, 
                        max_iter=max_iter)   
        
        self.train_x = train_x
        self.train_y = train_y
        
        return lhat, loss_min
    
    
    def __predict_step(self, train_x, x):
        """This is the prediction method that compute the predicted outcome for new data x from train data x

        Args:
            train_x (tensor): train data
            x (tensor): new data


        Returns:
            tensor: predicted value
        """
        if self.coef is None:
            raise ValueError("COSSO model should be trained before prediction")
        R_lazy = self.covar_module(x, train_x)
        y = R_lazy.matmul(self.coef[:-1]) + self.coef[-1]
        return y
        

    def get_components(self, x=None):
        """This methods returns the estimated components for data x

        Args:
            x (tensor, optional): data. Defaults to None.

        Returns:
            tensor, tuple: estimated components 
        """
        components = None
        if x is None:
            x = self.train_x
        if self.model_order == 1:
            theta = self.covar_module.outputscale.unsqueeze(-1).unsqueeze(-1)
            R = self.base_kernel(x, self.train_x, last_dim_is_batch=True)
            components = ((R.mul(theta)).matmul(self.coef[:-1]),)
        else:
            theta = self.main_covar_module.outputscale.unsqueeze(-1).unsqueeze(-1)
            R = self.base_kernel(x, self.train_x, last_dim_is_batch=True)
            components_main = (R.mul(theta)).matmul(self.coef[:-1])

            theta = self.interaction_covar_module.outputscale.unsqueeze(-1).unsqueeze(-1)
            index = torch.tensor([[i, j] for i in np.arange(0, self.ndims - 1)
                                for j in np.arange(i + 1, self.ndims)], dtype=torch.long)
            res0 = self.base_kernel(x, self.train_x, last_dim_is_batch=True)
            R = res0[index[:, 0], :, :] * res0[index[:, 1], :, :]
            components_inter = (R.mul(theta)).matmul(self.coef[:-1])
            components = (components_main, components_inter)
        return components
    
    @property
    def selection_parameter(self):
        """The selection parameter is just the outputscale for covar_module.  

        Returns:
            tensor or tuple: containing the selection parameters
        """
        N = self.train_x.shape[-1]
        out = (None,)
        adaptive_scale = get_adaptive_scale_from_covariance_trace(self.train_x, self.poly_order,
                                                model_order=self.model_order,
                                                c=None, correction=False)        
        if self.model_order == 1:
            return ((self.main_covar_module.outputscale * adaptive_scale[0]).detach().numpy(),)
        else:
            return ((self.main_covar_module.outputscale * adaptive_scale[0]).detach().numpy(), 
                    (self.interaction_covar_module.outputscale * adaptive_scale[1]).detach().numpy())
    
    
    
    def predict(self, x, kind='latent', component_index=None):
        """This is the prediction method that compute the predicted outcome for new data x.

        Args:
            x (tensor): _description_
            kind (str, optional): main, intraction, latent or outcome. Defaults to 'latent'.
            component_index (int or list, optional): target component. Defaults to None.


        Returns:
            tensor: predicted value
        """
        if kind=='latent':
            return self.__predict_step(self.train_x, x)
        elif kind=='outcome':
            return self.__predict_step(self.train_x, x)
        else:
            components = self.get_components(x)
            if kind=='main':
                if component_index is None:
                    component_index = torch.arange(0, self.ndims)    
                if isinstance(component_index, numbers.Number):
                    component_index = torch.tensor([component_index])
                return dict(zip([str(int(i)) for i in component_index], [components[0][k, :] for k in range(self.ndims)]))

            elif kind=='interaction': 
                if component_index is None:
                    component_index = torch.tensor([[i, j] for i in np.arange(0, self.ndims - 1)
                                    for j in np.arange(i + 1,self. ndims)], dtype=torch.long)
                k = [first_order_InteractionEffect_to_index(self.ndims, list(j)) for j in component_index.numpy()]
    
                return dict(zip([str(component_index[i].numpy()) for i in k],
                                [components[1][i, :] for i in k]))
    @property
    def constant_mean(self):
        """This gives the estimated constant mean

        Args:
            x (tensor): constant mean
        """
        return self.coef[-1]
        


























