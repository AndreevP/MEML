import numpy as np
import torch
import math

eps = 1e-8


def squeeze2d(x, factor=2):
    n, c, h, w = x.size()
    assert h % factor == 0 and w % factor == 0
    x = x.view(-1, c, h // factor, factor, w // factor, factor)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
    x = x.view(-1, factor * factor * c, h // factor, w // factor)
    return x


def unsqueeze2d(x, factor=2):
    n, c, h, w = x.size()
    number = factor ** 2
    assert c >= number and c % number == 0
    x = x.view(-1, factor, factor, c // number, h, w)
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
    x = x.view(-1, c // number, h * factor, w * factor)
    return x


def split2d(x, channels):
    z1 = x[:, :channels]
    z2 = x[:, channels:]
    return z1, z2


def unsplit2d(x):
    return torch.cat(x, dim=1)


def preprocess(image, bits, noise=None):
    bins = 2. ** bits
    image = image.mul(255)
    if bits < 8:
        image = torch.floor(image.div(256. / bins))
    if noise is not None:
        image = image + noise
    image = image.div(bins)
    image = (image - 0.5).div(0.5)
    return image


def postprocess(image, bits):
    bins = 2. ** bits
    image = image.mul(0.5) + 0.5
    image = image.mul(bins)
    image = torch.floor(image) * (256. / bins)
    image = image.clamp(0, 255).div(255)
    return image


b = [math.factorial(2 * 5 - j) * math.factorial(5) \
     / math.factorial(2 * 5) / math.factorial(5 - j) \
     / math.factorial(j) for j in range(6)]

#K_def = 4 #number of matmuls

def pade(A, K):
    if (A.max().item() == 0):
        return A + torch.eye(A.shape[-1]).to(A)
    k = min(int(np.ceil(np.log2(np.max([torch.norm(A, p=1, dim=-1).max().item(), 0.5]))) + 1), K-5)
    A = A / 2**k
    A2 = A @ A
    A4 = A2 @ A2
    U = A @ (b[5] * A4 + b[3] * A2 + b[1] * torch.eye(A.shape[-1]).to(A))
    V = b[4] * A4 + b[2] * A2 + b[0] * torch.eye(A.shape[-1]).to(A)
    #E, _ = torch.solve(U + V, -U + V)
    E = torch.inverse(-U + V) @ (U + V)
    for i in range(k):
        E = E @ E
    return E

B = [math.factorial(2 * 3 - j) * math.factorial(3) \
     / math.factorial(2 * 3) / math.factorial(3 - j) \
     / math.factorial(j) for j in range(4)]
def pade3(A, K):
    if (A.max().item() == 0):
        return A + torch.eye(A.shape[-1]).to(A)
    k = min(int(np.ceil(np.log2(np.max([torch.norm(A, p=1, dim=-1).max().item(), 0.5]))) + 1), K-4)
    A = A / 2**k
    A2 = A @ A
    U = A @ (B[3] * A2 + B[1] * torch.eye(A.shape[-1]).to(A))
    V =  B[2] * A2 + B[0] * torch.eye(A.shape[-1]).to(A)
    #E, _ = torch.solve(U + V, -U + V)
    E = torch.inverse(-U + V) @ (U + V)
    for i in range(k):
        E = E @ E
    return E


x3 = 2./3
sq = math.sqrt(177)
y = [1, 1, (857 - 58 * sq)/630]
x = [0, x3 * (1. + sq)/88, (1 + sq) /352 * x3, x3, (-271 + 29 *sq)/315/x3,
    11* (-1 + sq)/1260 /x3, 11 * (-9 + sq)/5040/x3, (89 - sq)/5040/(x3**2)]

def optimized_taylor(A, K):
    if (A.max().item() == 0):
        return A + torch.eye(A.shape[-1]).to(A)  
    k = min(int(np.ceil(np.log2(np.max([torch.norm(A, p=1, dim=-1).max().item(), 0.5]))) + 1), K-3)
    A = A / 2**k
    A2 = A @ A
    A4 = A2 @ (x[1] * A + x[2] * A2)
    A8 = (x3 * A2 + A4) @ (x[4] * torch.eye(A.shape[-1]).to(A) + x[5] * A + x[6] * A2 + x[7] * A4)
    E = y[0] * torch.eye(A.shape[-1]).to(A) + y[1] * A + y[2] * A2 + A8
    for i in range(k):
        E = E @ E
    return E  


def second_limit(A, K):
    if (A.max().item() == 0):
        return A + torch.eye(A.shape[-1]).to(A)
    y = A / 2**K + torch.eye(A.shape[-1]).to(A)
    for k in range(K):
        y = y @ y
    return y


def default(x, K):
    if (x.max().item() == 0):
        return x + torch.eye(x.shape[-1]).to(x)
    scale = min(int(np.ceil(np.log2(np.max([torch.norm(x, p=1, dim=-1).max().item(), 0.5]))) + 1), K-3)
    x = x / (2 ** scale)
    s = torch.eye(x.size(-1), device=x.device)
    t = x
    k = 2
   # while torch.norm(t, p=1, dim=-1).max().item() > eps:
    for i in range(K - scale):
        s = s + t
        t = torch.matmul(x, t) / k
        k = k + 1
        if torch.norm(t, p=1, dim=-1).max().item() < eps:
            break
    for i in range(scale):
        s = torch.matmul(s, s)
    return s

def default_full(x, K=0):
    if (x.max().item() == 0):
        return x + torch.eye(x.shape[-1]).to(x)
    scale = int(np.ceil(np.log2(np.max([torch.norm(x, p=1, dim=-1).max().item(), 0.5]))) + 1)
    x = x / (2 ** scale)
    s = torch.eye(x.size(-1), device=x.device)
    t = x
    k = 2
    while torch.norm(t, p=1, dim=-1).max().item() > eps:
        s = s + t
        t = torch.matmul(x, t) / k
        k = k + 1
    for i in range(scale):
        s = torch.matmul(s, s)
    return s


D = {"default" : default, "optimized_taylor" : optimized_taylor, "pade": pade, "second_limit" : second_limit,
    "default_full" : default_full, "pade3" : pade3}

def expm(x, method="default 0"):
    """
    compute the matrix exponential: \sum_{k=0}^{\infty}\frac{x^{k}}{k!}
    """
    i = method.find(" ")
    K = int(method[i+1:])
    if K == 0:
        K = 1000
    method = method[:i]
    shape = x.shape
    x = x.reshape((-1, shape[-2], shape[-1]))
    E = D[method](x, K)
    E = E.reshape(shape)
    return E


def series(x, method="default 0"):
    """
    compute the matrix series: \sum_{k=0}^{\infty}\frac{x^{k}}{(k+1)!}
    """
    i = method.find(" ")
    K = int(method[i+1:])
    if K == 0:
        K = 1000
    method = method[:i]
    if method == "default":
        s = torch.eye(x.size(-1), device=x.device)
        t = x / 2
        k = 3

        while torch.norm(t, p=1, dim=-1).max().item() > eps:
            s = s + t
            t = torch.matmul(x, t) / k
            k = k + 1
    else:
        s = torch.inverse(x) @ D[method](x, K) - torch.eye(s.shape[-1])
    return s
