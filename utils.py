# Code modified to handle non-square images

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from scipy.interpolate import griddata
import cv2
from typing import Optional, Union
import ot as pot
from functools import partial
import math
from LSIM.distance_model import DistanceModel



def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def plot_field(X, vmin=-2, vmax=2, figsize=(5,5), colorbar=True, cmap="twilight"):
    X = X.cpu() if type(X) is torch.Tensor else X
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    img = axes.imshow(X, vmin=vmin, vmax=vmax, cmap=cmap)
    if colorbar: fig.colorbar(img, ax=axes)
    plt.show()


def plot_two(X, Y, vmin=-2, vmax=2, colorbar=False, figsize=(10, 5)):
    X = X.cpu() if type(X) is torch.Tensor else X
    Y = Y.cpu() if type(Y) is torch.Tensor else Y

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    im0 = axes[0].imshow(X, vmin=vmin, vmax=vmax)
    im1 = axes[1].imshow(Y, vmin=vmin, vmax=vmax)
    if colorbar:
        fig.colorbar(im0, ax=axes[0])
        fig.colorbar(im1, ax=axes[1])
    plt.show()


def plot_three(X, Y, Z, vmin=-2, vmax=2, colorbar=False, figsize=(15, 5)):
    X = X.cpu() if type(X) is torch.Tensor else X
    Y = Y.cpu() if type(Y) is torch.Tensor else Y
    Z = Z.cpu() if type(Z) is torch.Tensor else Z

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    im0 = axes[0].imshow(X, vmin=vmin, vmax=vmax)
    im1 = axes[1].imshow(Y, vmin=vmin, vmax=vmax)
    im2 = axes[2].imshow(Z, vmin=vmin, vmax=vmax)
   
    if colorbar:
        fig.colorbar(im0, ax=axes[0])
        fig.colorbar(im1, ax=axes[1])
        fig.colorbar(im2, ax=axes[2])
    
    plt.show()


def plot_four(X, Y, W, Z, vmin=-4, vmax=4, colorbar=False, axes_on=False):
    X = X.cpu() if type(X) is torch.Tensor else X
    Y = Y.cpu() if type(Y) is torch.Tensor else Y
    W = W.cpu() if type(W) is torch.Tensor else W
    Z = Z.cpu() if type(Z) is torch.Tensor else Z

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    im0 = axes[0].imshow(X, cmap="twilight", vmin=vmin, vmax=vmax)
    im1 = axes[1].imshow(Y, cmap="twilight", vmin=vmin, vmax=vmax)
    im2 = axes[2].imshow(W, cmap="twilight", vmin=vmin, vmax=vmax)
    im2 = axes[3].imshow(Z, cmap="twilight", vmin=vmin, vmax=vmax)
   
    if colorbar:
        fig.colorbar(im0, ax=axes[0])
        fig.colorbar(im1, ax=axes[1])
        fig.colorbar(im2, ax=axes[2])
        fig.colorbar(im2, ax=axes[3])

    if not axes_on:
        for ax in axes.flat:
            ax.axis('off')
    
    plt.show()


def plot_grid(X_list, Y_list, Z_list, W_list, vmin=-10, vmax=10, colorbar=False, L=3, title=None):
    to_cpu = lambda x : x.cpu() if type(x) is torch.Tensor else x

    N = len(X_list)
    fig, axes = plt.subplots(N, 4, figsize=(L * 4, L * N))
    if title: fig.suptitle(title)

    for i in range(N):
        im0 = axes[i, 0].imshow(to_cpu(X_list[i,0]), cmap="twilight", vmin=vmin, vmax=vmax)
        im1 = axes[i, 1].imshow(to_cpu(Y_list[i,0]), cmap="twilight", vmin=vmin, vmax=vmax)
        im2 = axes[i, 2].imshow(to_cpu(Z_list[i,0]), cmap="twilight", vmin=vmin, vmax=vmax)
        im3 = axes[i, 3].imshow(to_cpu(W_list[i,0]), cmap="twilight", vmin=vmin, vmax=vmax)

        cbar = fig.colorbar(im3, ax=axes[i, 3], fraction=0.03, pad=0.03)


    for ax in axes.flatten(): ax.axis("off")
    
    if title: 
        plt.tight_layout(rect=[0, 0, 1, 0.98])
    else: 
        plt.tight_layout()
    
    plt.show()


def l1_loss_fn(x, y):
    return torch.mean(torch.abs(x - y))


def l2_loss_fn(x, y):
    return ((x - y)**2).mean((-1, -2)).sqrt().mean()


class StdScaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def scale(self):
        return self.std
    

def interpolate_points(image, perc=0, ids=None, method="nearest"):
    Nx, Ny = image.shape  # Allow non-square images

    if ids is None:
        sampled_ids = random.sample(range(Nx * Ny), int(Nx * Ny * perc))
    else:
        sampled_ids = ids

    vals = np.tile(image.reshape(Nx * Ny)[sampled_ids], 9)

    ids = [[(x // Ny), (x % Ny)] for x in sampled_ids] + \
          [[(x // Ny), Ny + (x % Ny)] for x in sampled_ids] + \
          [[(x // Ny), 2*Ny + (x % Ny)] for x in sampled_ids] + \
          [[Nx + (x // Ny), (x % Ny)] for x in sampled_ids] + \
          [[Nx + (x // Ny), Ny + (x % Ny)] for x in sampled_ids] + \
          [[Nx + (x // Ny), 2*Ny + (x % Ny)] for x in sampled_ids] + \
          [[2*Nx + (x // Ny), (x % Ny)] for x in sampled_ids] + \
          [[2*Nx + (x // Ny), Ny + (x % Ny)] for x in sampled_ids] + \
          [[2*Nx + (x // Ny), 2*Ny + (x % Ny)] for x in sampled_ids]

    grid_x, grid_y = np.mgrid[0:Nx*3, 0:Ny*3]
    grid_z = griddata(ids, vals, (grid_x, grid_y), method=method, fill_value=0)  # 'linear', 'cubic', 'nearest'

    return torch.tensor(grid_z[Nx:Nx*2, Ny:Ny*2])


def interpolate_dataset(dataset, perc, method="nearest"):
    X_vals      = dataset.cpu().clone() if type(dataset) is torch.Tensor else dataset.copy()
    n_samples   = dataset.shape[0] 
    Ny          = dataset.shape[-1]
    Nx          = dataset.shape[-2]
    n_points    = int(Nx * Ny * perc)
    sampled_ids = np.zeros((n_samples, n_points),dtype=np.int32)

    for i in range(n_samples):
        if i%100 == 0: print(f"Interpolating sample {i} of {n_samples}")
        sampled_ids[i] = np.array(random.sample(range(Nx * Ny), n_points))

        X_vals[i, 0] = interpolate_points(X_vals[i, 0], perc=perc, ids=sampled_ids[i], method=method)
        X_vals[i, 1] = interpolate_points(X_vals[i, 1], perc=perc, ids=sampled_ids[i], method=method)
        X_vals[i, 2] = interpolate_points(X_vals[i, 2], perc=perc, ids=sampled_ids[i], method=method)

    return X_vals, sampled_ids


def regular_sparse_data(dataset, N_grid=8, method="nearest"):

    X_vals      = dataset.cpu().clone() if type(dataset) is torch.Tensor else dataset.copy()
    n_samples   = dataset.shape[0] 
    N           = dataset.shape[-1]
    sampled_ids = np.zeros((N, N), dtype=np.int32)
    
    for i in range(N // (2*N_grid), N, N // N_grid):
        for j in range(N // (2*N_grid), N, N // N_grid):
            sampled_ids[i,j] = 1

    sampled_ids = np.where(sampled_ids.flatten()==1)[0]

    for i in range(n_samples):
        X_vals[i, 0] = interpolate_points(X_vals[i, 0], ids=sampled_ids, method=method)
        X_vals[i, 1] = interpolate_points(X_vals[i, 1], ids=sampled_ids, method=method)
        X_vals[i, 2] = interpolate_points(X_vals[i, 2], ids=sampled_ids, method=method)

    return torch.tensor(X_vals), sampled_ids


def downscale_data(high_res, scale_factor):
    channels = len(high_res.shape) == 4 

    if channels:
        N, C, Lx, Ly = high_res.shape
        high_res = high_res.reshape(N * C, Lx, Ly)
    else:
        N, Lx, Ly = high_res.shape

    _high_res = high_res.numpy() if isinstance(high_res, torch.Tensor) else high_res

    Lx_small = int(Lx / scale_factor)
    Ly_small = int(Ly / scale_factor)
    NN = _high_res.shape[0]

    X_small = np.zeros((NN, Lx_small, Ly_small))
    X_upscaled = np.zeros((NN, Lx, Ly))

    for i in range(NN):
        X_small[i] = cv2.resize(_high_res[i], (Ly_small, Lx_small), interpolation=cv2.INTER_CUBIC)
        X_upscaled[i] = cv2.resize(X_small[i], (Ly, Lx), interpolation=cv2.INTER_CUBIC)

    if channels:
        X_upscaled = X_upscaled.reshape(N, C, Lx, Ly)

    return torch.Tensor(X_upscaled)


def diffuse_mask(value_ids, A=1, sig=0.044, search_dist=-1, N=256, Nx=256, Ny=256, tol=1e-6):
    L = 2 * np.pi
    X = np.linspace(0, L, Nx)
    Y = np.linspace(0, L, Ny)
    dx = L / Nx
    dy = L / Ny
    grid = np.zeros((Nx, Ny))

    grid[0,  :] = 1
    grid[-1, :] = 1
    grid[:,  0] = 1
    grid[:, -1] = 1

    def gauss(x0, y0, x, y):
        return A * np.exp(-((x0 - x)**2+(y0 - y)**2)/(2*sig**2))

    if search_dist < 0:
        min_search_steps = 0
        while gauss(0, 0, dx*min_search_steps, 0) > tol:
            min_search_steps += 1
        search_dist = min_search_steps

    """
    for sid in value_ids:
        i = sid // N
        j = sid % N
        x0 = X[i]
        y0 = Y[j]

        for ii in range(-search_dist, search_dist):
            for jj in range(-search_dist, search_dist):
                x = x0 + ii * dx
                y = y0 + jj * dy

                gi = (i + ii) % N
                gj = (j + jj) % N

                grid[gi, gj] = max(grid[gi,gj], gauss(x0, y0, x, y))

    return grid
    """

    gaussian = np.zeros((search_dist*2 + 1, search_dist*2 + 1))
    x0 = y0 = search_dist * dx
    for i in range(len(gaussian)):
        for j in range(len(gaussian)):
            gaussian[i,j] = gauss(x0, y0, i*dx, j*dx)

    for sid in value_ids:
        i = sid // Ny  # Compute row index correctly for Nx × Ny grid
        j = sid % Ny   # Compute column index correctly

        ilb = max(0, i - search_dist)
        iub = min(Nx, i + search_dist + 1)
        jlb = max(0, j - search_dist)
        jub = min(Ny, j + search_dist + 1)

        S = search_dist * 2 + 1

        gilb = max(0, search_dist - i)
        giub = S - max(0, i + search_dist - (Nx - 1))
        gjlb = max(0, search_dist - j)
        gjub = S - max(0, j + search_dist - (Ny - 1))

        grid[ilb:iub, jlb:jub] = np.fmax(gaussian[gilb:giub, gjlb:gjub], grid[ilb:iub, jlb:jub])

    return grid


# https://github.com/atong01/conditional-flow-matching/blob/c25e1918a80dfacbe9475c055d61ac997f28262a/torchcfm/optimal_transport.py#L218
def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    """Compute the Wasserstein (1 or 2) distance (wrt Euclidean cost) between a source and a target
    distributions.

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the source minibatch
    method : str (default : None)
        Use exact Wasserstein or an entropic regularization
    reg : float (default : 0.05)
        Entropic regularization coefficients
    power : int (default : 2)
        power of the Wasserstein distance (1 or 2)
    Returns
    -------
    ret : float
        Wasserstein distance
    """
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=int(1e7))
    if power == 2:
        ret = math.sqrt(ret)
    return ret


def fix_randomness(seed=1234):
    torch.torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)



lsim_model = DistanceModel(baseType="lsim", isTrain=False, useGPU=False)
lsim_model.load("LSIM/LSiM.pth")

def LSiM_distance(A, B):
    # https://github.com/tum-pbs/LSIM
    # Expected input sizes: [1, 3, 256, 256], [3, 256, 256]  or [256,256]
    assert len(A.shape) == len(B.shape)
    global lsim_model
    
    if len(A.shape) == 4:
        A = A[0]
        B = B[0]

    if A.shape[0] == 3:
        return np.mean([
            LSiM_distance(A[0], B[0]),
            LSiM_distance(A[1], B[1]),
            LSiM_distance(A[2], B[2])
        ])

    if len(A.shape) == 2:
        A = A.unsqueeze(-1)
    
    if len(B.shape) == 2:
        B = B.unsqueeze(-1)
        
    A = A.cpu() if type(A) is torch.Tensor else A
    B = B.cpu() if type(B) is torch.Tensor else B

    dist = lsim_model.computeDistance(A, B)

    return dist[0]


def spectral_kinetic_energy(sample, n_bins=200, N=256, L=2*torch.pi):
    dx = L / N

    kx = torch.fft.fftfreq(N, dx).reshape(N, 1) * 2 * torch.pi * 1j
    ky = torch.fft.fftfreq(N, dx).reshape(1, N) * 2 * torch.pi * 1j    
    lap = (kx ** 2 + ky ** 2)
    lap[..., 0, 0] = 1.0

    w_h = torch.fft.fft2(sample, dim=[-2, -1])
    psi_h = -w_h / lap

    u_h = psi_h * ky
    v_h = -psi_h * kx

    # kx = torch.fft.fftfreq(N, dx) * L # TODO: 
    # ky = torch.fft.fftfreq(N, dx) * L

    dx = L / N
    E_k = 0.5 * (torch.abs(u_h)**2 + torch.abs(v_h)**2)

    kx = torch.fft.fftfreq(N, dx)
    ky = torch.fft.fftfreq(N, dx)

    kx, ky = torch.meshgrid(kx, ky, indexing="ij")
    k = torch.sqrt(kx**2 + ky**2)  # Radial wavenumber
    k_max = torch.max(k)

    bins = torch.arange(0, k_max + k_max / n_bins, k_max / n_bins)
    E_k_shell = torch.zeros(len(bins))

    for i in range(len(bins) - 1):
        modes = torch.abs(E_k[(k > bins[i]) * (k <= bins[i+1])])
        if len(modes) > 0:
            E_k_shell[i] = torch.sum(modes) / len(modes)

    bins = [b for i, b in enumerate(bins) if E_k_shell[i] > 0]
    E_k_shell = [e for e in E_k_shell if e > 0]
    
    return bins, E_k_shell


def spectral_loss(real, pred):
    return np.sqrt(np.sum((np.log(real) - np.log(pred))**2))