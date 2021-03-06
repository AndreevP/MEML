import torch
import torch.nn as nn
from m_layer import MLayer
from torch.nn.functional import sigmoid
from torch.distributions import Normal

class SpiralMLayer(nn.Module):

    def __init__(self, dim_repr, dim_matrix, expm, l2_reg=1e-3, device='cuda'):
        super().__init__()
        self.device = device
        self.dim_repr = dim_repr
        self.dim_matrix = dim_matrix
        self.lin1 = nn.Linear(2, self.dim_repr)
        self.m_layer = MLayer(
            self.dim_repr, self.dim_matrix, matrix_init='uniform', expm=expm, with_bias=True, device=device)
        self.flatten = nn.Flatten()
        self.lin2 = nn.Linear(self.dim_matrix**2, 1)
        self.loss_fn = nn.BCELoss()
        self.l2_reg = l2_reg
        self.to(self.device)
    
    def forward(self, x):
        x_m = self.m_layer(self.lin1(x))
        x_l2_norms = torch.norm(x_m, dim=(1, 2))**2
        return torch.sigmoid(self.lin2(self.flatten(x_m))), x_l2_norms
    
    def loss(self, x, y):
        x_out, x_l2_norms = self(x)
        return self.loss_fn(x_out.squeeze(), y.squeeze()) + self.l2_reg * torch.mean(x_l2_norms), x_out

class SpiralDNN(nn.Module):

    def __init__(self, hid_dim, device='cuda', activation=nn.ReLU()):
        super().__init__()
        self.device = device
        self.hid_dim = hid_dim
        self.net = nn.Sequential(
            nn.Linear(2, self.hid_dim),
            activation,
            nn.Linear(self.hid_dim, self.hid_dim),
            activation,
            nn.Linear(self.hid_dim, 1)
        )
        self.loss_fn = nn.BCELoss()
        self.to(self.device)
    
    def forward(self, x):
        return torch.sigmoid(self.net(x))
    
    def loss(self, x, y):
        x_out = self(x)
        return self.loss_fn(x_out.squeeze(), y.squeeze()), x_out

class PeriodicMLayer(nn.Module):

    def _init_U(self):
        shape = self.m_layer._rep_to_exp_tensor.shape
        data = Normal(0., self.init_scale).sample(shape).float()
        for i in range(shape[1]):
            data[:, i, i] -= self.init_diag
        self.m_layer._rep_to_exp_tensor.data = data.to(self.m_layer._rep_to_exp_tensor.data)

    def __init__(self, dim_repr, dim_matrix, init_scale, init_diag, expm, device='cuda'):
        super().__init__()
        self.device = device
        self.dim_repr = dim_repr
        self.dim_matrix = dim_matrix
        self.init_scale = init_scale
        self.init_diag = init_diag
        # self.bn = nn.BatchNorm1d(1)
        self.lin1 = nn.Linear(1, self.dim_repr)
        self.m_layer = MLayer(
            self.dim_repr, self.dim_matrix, matrix_init='uniform', expm=expm, with_bias=False, device=device)
        self.flatten = nn.Flatten()
        self.lin2 = nn.Linear(self.dim_matrix**2, 1)
        self.loss_fn = nn.MSELoss()
        self._init_U()
        self.to(self.device)
    
    def forward(self, x):
        # x_1 = self.lin1(self.bn(x))
        x_1 = self.lin1(x)
        # print(x_1)
        x_m = self.m_layer(x_1)
        # print(x_m)
        res = self.lin2(self.flatten(x_m))
        # print(res)
        return res
    
    def loss(self, x, y):
        x_out = self(x)
        return self.loss_fn(x_out, y)
