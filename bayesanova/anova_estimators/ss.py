import numbers
import math 

import numpy as np
import torch

from bayesanova.anova_estimators.utils import k_fold
from bayesanova.models import sobolev_kernel
from bayesanova.models.utils import first_order_InteractionEffect_to_index

'''  This version does not include residual components ! '''

class SS(torch.nn.Module):
    """Smoothing spline Anova method

    Args:
        ndims (int): dimension size 
        model_order (int) : model order
        poly_order (int) : polynomial order
    """
    def __init__(self,  ndims, model_order=1, poly_order=1, base_kernel=None, weight=None):
        super(SS, self).__init__()
        
        base_kernel = sobolev_kernel.univ_main_effect(poly_order, c=None) 
      
    
        self.base_kernel = base_kernel
        self.model_order = model_order
        
        self.base_kernel_main = base_kernel
        self.ndims = ndims
        
        if weight is None:
            if self.model_order==1:
                self.weight = (None, )
            else:
                self.weight = (None, None)
        else:
            self.weight = weight
        
        if self.model_order==1:
            covar_module = sobolev_kernel.ScaleAdditiveStructureKernel(base_kernel, ndims)
            covar_module.outputscale = torch.ones(ndims, 1)
            if self.weight[0] is not None:
                covar_module.outputscale = torch.reshape(self.weight[0], (ndims, 1))
            self.covar_module = covar_module
            
        elif self.model_order==2:
            self.main_covar_module = sobolev_kernel.ScaleAdditiveStructureKernel(base_kernel, ndims)
            self.interaction_covar_module = sobolev_kernel.ScaleAdditiveStructureKernel(base_kernel, ndims * (ndims - 1) // 2)
            self.main_covar_module.outputscale = torch.ones(ndims, 1)
            self.interaction_covar_module.outputscale = torch.ones(ndims*(ndims-1)//2, 1)
            if self.weight[0] is not None:
                self.main_covar_module.outputscale = torch.reshape(self.weight[0], (ndims, 1))
            if self.weight[1] is not None:
                self.interaction_covar_module.outputscale = torch.reshape(self.weight[1], (ndims*(ndims-1)//2, 1))
            self.covar_module = self.main_covar_module + self.interaction_covar_module
        self.train_x = None
        self.train_y = None
        self.coef = None


    def initialize(self):
        """This method initialize to SS model to clear cache
        """
        self.coef = None
        self.train_x = None
        self.train_y = None
        if self.model_order==1:
            if self.weight[0] is None:
                self.covar_module.outputscale = torch.ones(self.ndims, 1)
            else:
                self.covar_module.outputscale = torch.reshape(self.weight[0], (self.ndims, 1))
                
        elif self.model_order==2:
            if self.weight[0] is None:
                self.main_covar_module.outputscale = torch.ones(self.ndims, 1)
            else:
                self.main_covar_module.outputscale = torch.reshape(self.weight[0], (self.ndims, 1))
                
            if self.weight [1] is None:
                self.interaction_covar_module.outputscale = torch.ones(self.ndims*(self.ndims-1)//2, 1)
            else:
                self.interaction_covar_module.outputscale = torch.reshape(self.weight[1],
                                                                        (self.ndims*(self.ndims-1)//2, 1))
        
    def train_step(self, train_x, train_y, alpha=torch.tensor((0.01)), GCV=False):
        """This method trains the SS model using train data (train_x, train_y)
        given a scalar smoothing parameter k 

        Args:
            train_x (tensor): train inputs
            train_y (tensor): train outcome
            alpha (float, tensor): smoothing parameter. Defaults to torch.tensor((0.01)).
            GCV (bool, optional): either use GCV or not. Defaults to False.

        Returns:
            tensor: the estimated train_y
        """
        N = train_y.shape[0]
        R_lazy = self.covar_module(train_x).add_jitter()
        
        d = 1/N*R_lazy.matmul(torch.ones(N,1).to(R_lazy.dtype))
        
        R_lazy_square = R_lazy.t().matmul(R_lazy)
        Q = 1/N*R_lazy_square + R_lazy.mul(alpha)
        Q = Q.add_jitter()
        
        A = Q.cat_rows(cross_mat=d.T, new_mat = torch.ones(1,1), generate_roots=False).add_jitter()
        
        z = 1/N*torch.cat((R_lazy.t().matmul(train_y), torch.tensor([train_y.sum()])))
        self.coef = A.inv_matmul(z)
        yhat = R_lazy.matmul(self.coef[:-1])+self.coef[-1]
        output = yhat
        if GCV:
            K = torch.cat((R_lazy.evaluate(), torch.ones(1,N)))
            trace = A.inv_quad(K)
            loss = (torch.norm(train_y - yhat) /((N-trace)/N))**2
            output = yhat, loss
        return output
    
    def __train_test_step(self, train_x, train_y, test_x, test_y, alpha, loss):
        """This is a train + test steps

        Args:
            train_x (tensor): train inputs
            train_y (tensor): train outcome
            test_x (tensor): test inputs
            test_y (tensor): test outcome
            l (float, tensor): smoothing parameter. Defaults to torch.tensor((0.01)).
            loss (function): this is the loss to optimize the smoothing parameter.
                            Defaults to None.

        Returns:
            tensor : loss value in test data
        """
        self.initialize()
        try:
            self.train_step(train_x, train_y, alpha=alpha, GCV=False)
            output = self.__predict_step(train_x, test_x)
            loss_val = loss(output , test_y) 
        except:
            loss_val = torch.tensor([math.inf])
        return loss_val
    
    def train(self, train_x, train_y, GCV=False, alpha=torch.tensor((0.01)), 
            folds=4, loss=None, random_state=123456): 
        """This is a general training method of the SS model using train data (train_x, train_y)
        given either a scalar smoothing parameter k or a set of smoothing parameters to choose
        the optimal value with cross-validation 

        Args:
            train_x (tensor): train inputs
            train_y (tensor): train outcome
            GCV (bool, optional): either use GCV or not. Defaults to False.
            folds (int, optional): number of folds in cross-validation. Defaults to 4.
            alpha (float, tensor): smoothing parameter. Defaults to torch.tensor((0.01)).
            loss (function, optional):  this is the loss to optimize the smoothing parameter.
                                        Defaults to None.

        Returns:
            float or tensor, float or tensor: optimal alpha, optimal loss
        """
        self.initialize()
        loss_min = None
        
        if not(torch.is_tensor(alpha)):
            alpha = torch.tensor((alpha))
    
        
        if alpha.ndimension()>0:
            # First define the loss if not defined
            if loss is None:
                loss = lambda output, test_y : torch.mean((output - test_y) ** 2)
            # Define the training strategy  
            train_stratgey = lambda  x1, y1, x2, y2, l: self.__train_test_step(x1, y1, x2, y2, l, loss)
            # Find the optimal smoothing parameter
            lhat, loss_min = k_fold(train_x, train_y, alpha, train_stratgey, folds=folds,
                                    random_state=random_state)
            
        else:
            lhat = alpha
        
        self.initialize()
        # Optimize SS with respect to train data with the optimal smoothing parameter    
        self.train_step(train_x, train_y, alpha=lhat, GCV=GCV)   
        
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
            raise ValueError("SS model should be trained before prediction")
        R_lazy = self.covar_module(x, train_x)
        y = R_lazy.matmul(self.coef[:-1])+self.coef[-1]
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
        if self.model_order==1:
            R = self.base_kernel(x, self.train_x, last_dim_is_batch=True)
            components = (R.matmul(self.coef[:-1]),)
        if self.model_order==2:
            R = self.base_kernel(x, self.train_x, last_dim_is_batch=True)
            components_main = R.matmul(self.coef[:-1])
            index = torch.tensor([[i, j] for i in np.arange(0, self.ndims- 1)
                                for j in np.arange(i + 1, self.ndims)], dtype=torch.long)
            res0= self.base_kernel(x, self.train_x, last_dim_is_batch=True)
            R = res0[index[:, 0], :, :] * res0[index[:, 1], :, :]
            components_inter = R.matmul(self.coef[:-1])
            components = (components_main, components_inter)
        return components
    
    
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
                                    for j in np.arange(i + 1, self. ndims)], dtype=torch.long)
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
        


























