import torch
from time import perf_counter
import sys
import math
import numpy as np

class Expm:

    def __init__(self):
        self.n_calls = 0
        self.time = 0
    
    def __call__(self, A):
        s_time = perf_counter()
        res = self.compute_expm(A)
        el_time = perf_counter() - s_time
        self.n_calls += 1
        self.time += el_time
        return res
    
    def release(self):
        return self.time, self.n_calls
    
    def reset(self):
        self.n_calls = 0
        self.time = 0
    
    def compute_expm(self, A):
        raise NotImplementedError()
        

class SecondGreatLimitExpm(Expm):

    def __init__(self, n_matmuls):
        super().__init__()
        self.n_matmuls = n_matmuls
    
    def compute_expm(self, A):
        y = A * 0.5**self.n_matmuls + torch.eye(A.shape[-1]).to(A)
        for k in range(self.n_matmuls):
            y = y @ y
        return y

class PadeExpm(Expm):

    def __init__(self, n_matmuls):
        super().__init__()
        self.n_matmuls = n_matmuls
        assert n_matmuls >= 5
        self.b = [math.factorial(2 * 5 - j) * math.factorial(5) \
            / math.factorial(2 * 5) / math.factorial(5 - j) \
            / math.factorial(j) for j in range(6)]
    
    def compute_expm(self, A):
        if (A.max().item() == 0):
            return A + torch.eye(A.shape[-1]).to(A)
        k = min(int(np.ceil(np.log2(np.max([torch.norm(A, p=1, dim=-1).max().item(), 0.5]))) + 1), self.n_matmuls-5)
        A = A / 2**k
        A2 = A @ A
        A4 = A2 @ A2
        U = A @ (self.b[5] * A4 + self.b[3] * A2 + self.b[1] * torch.eye(A.shape[-1]).to(A))
        V = self.b[4] * A4 + self.b[2] * A2 + self.b[0] * torch.eye(A.shape[-1]).to(A)
        E, _ = torch.solve(U + V, -U + V)
        for i in range(k):
            E = E @ E
        return E

class Pade3Expm(Expm):

    def __init__(self, n_matmuls):
        super().__init__()
        self.n_matmuls = n_matmuls
        assert n_matmuls >= 4
        self.b = [math.factorial(2 * 3 - j) * math.factorial(3) \
            / math.factorial(2 * 3) / math.factorial(3 - j) \
            / math.factorial(j) for j in range(4)] 
        
    def compute_expm(self, A):
        if (A.max().item() == 0):
            return A + torch.eye(A.shape[-1]).to(A)
        k = min(int(np.ceil(np.log2(np.max([torch.norm(A, p=1, dim=-1).max().item(), 0.5]))) + 1), self.n_matmuls-4)
        A = A / 2**k
        A2 = A @ A
        U = A @ (self.b[3] * A2 + self.b[1] * torch.eye(A.shape[-1]).to(A))
        V =  self.b[2] * A2 + self.b[0] * torch.eye(A.shape[-1]).to(A)
        #E, _ = torch.solve(U + V, -U + V)
        E = torch.inverse(-U + V) @ (U + V)
        for i in range(k):
            E = E @ E
        return E

class TaylorExpm(Expm):

    def __init__(self, n_matmuls):
        super().__init__()
        self.n_matmuls = n_matmuls
        assert n_matmuls >= 4
    
    def compute_expm(self, x):
        if (x.max().item() == 0):
            return x + torch.eye(x.shape[-1]).to(x)
        scale = min(int(np.ceil(np.log2(np.max([torch.norm(x, p=1, dim=-1).max().item(), 0.5]))) + 1), self.n_matmuls-3)
        x = x / (2 ** scale)
        s = torch.eye(x.size(-1), device=x.device)
        t = x
        k = 2
        # while torch.norm(t, p=1, dim=-1).max().item() > eps:
        for i in range(self.n_matmuls - scale):
            s = s + t
            t = torch.matmul(x, t) / k
            k = k + 1
        for i in range(scale):
            s = torch.matmul(s, s)
        return s

if __name__ == "__main__":
    A = torch.rand((3, 3))
    pexpm = PadeExpm(5)
    pexpm(A)

