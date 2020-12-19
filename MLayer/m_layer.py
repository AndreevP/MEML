import torch
import torch.nn as nn
from torch.distributions import Normal, Uniform
from torch.nn import ParameterList, Parameter
import numpy as np
from matrix_exponentials import SecondGreatLimitExpm

class MLayer(nn.Module):
    '''
    TODO: add docs
    '''

    @property
    def matrix_init(self):
        return self._matrix_init
    
    @matrix_init.setter
    def matrix_init(self, val):
        if isinstance(val, str):
            if val == 'normal':
                self._matrix_init = Normal(0., 1.)
            elif val == 'uniform':
                # see https://arxiv.org/pdf/2008.03936.pdf
                self._matrix_init = Uniform(-0.05, 0.05) 
            else:
                raise Exception(
                    "matrix_init with {}".format(val))
            return
        if isinstance(val, torch.distributions):
            self._matrix_init = val
            return

        raise Exception('wrong initialization')

    def __init__(
        self, input_dim, m_dim, matrix_init='normal', 
        with_bias=False, expm=SecondGreatLimitExpm(6), device='cuda'):
        '''
        :Parameters:
        input_dim: tuple: shape of input tensor
        m_dim: int : matrix dimension
        matrix_init: str or torch.distribution : parameters initializer
        with_bias : bool : whether to use bias (when constructing matrix from input)
        expm : callable : method to compute matrix exponential
        device : str : torch device
        '''
        super().__init__()
        self.input_dim = input_dim
        self.m_dim = m_dim
        self.matrix_init = matrix_init
        self.bias_init = Uniform(0., 1.)
        self.with_bias = with_bias
        self.expm = expm
        self.device = device

        self._rep_to_exp_tensor = Parameter(
            self.matrix_init.sample(
                (np.prod(self.input_dim), self.m_dim, self.m_dim)).to(self.device), 
                requires_grad=True)
        
        if self.with_bias:
            self._matrix_bias = Parameter(
                self.bias_init.sample((1, self.m_dim, self.m_dim)).to(self.device), 
                requires_grad=True)
    
    def forward(self, x):
        b_sz = x.size(0)
        x = x.view(b_sz, -1)
        mat = torch.einsum('amn,...a->...mn', self._rep_to_exp_tensor, x)
        if self.with_bias:
            mat = mat + self._matrix_bias
        return self.expm(mat)

if __name__ == "__main__":
    a = torch.rand((3, 4, 4))
    mlayer = MLayer((4, 4), 2, with_bias=True, device='cpu')
    res = mlayer(a)
    print(res.shape)
