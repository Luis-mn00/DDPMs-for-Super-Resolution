# Code modified to handle non-square grids

import torch
import numpy as np

def solve_poisson(vort, Lx=2*np.pi, Ly=2*np.pi):
    batch, Nx, Ny = vort.shape
    device = vort.device
    dx, dy = Lx / Nx, Ly / Ny

    vort_h = torch.fft.fft2(vort)
    # Create the wavevector grid in Fourier space
    kx = torch.fft.fftfreq(Nx, dx).reshape(1, Nx, 1).to(device) * 2 * torch.pi * 1j
    ky = torch.fft.fftfreq(Ny, dy).reshape(1, 1, Ny).to(device) * 2 * torch.pi * 1j    
    
    # Negative Laplacian in Fourier space
    lap = (kx ** 2 + ky ** 2)
    lap[..., 0, 0] = 1.0

    psi_h = -vort_h / lap

    u_h = psi_h * ky
    v_h = -psi_h * kx
    wx_h = kx * vort_h
    wy_h = ky * vort_h
    wlap_h = lap * vort_h

    u = torch.fft.ifft2(u_h).real
    v = torch.fft.ifft2(v_h).real
    wx = torch.fft.ifft2(wx_h).real
    wy = torch.fft.ifft2(wy_h).real
    wlap = torch.fft.ifft2(wlap_h).real
    advection = u * wx + v * wy

    return wlap, advection

def residual_fourier(w, Re=1000, dt=1/32, Lx=2*np.pi, Ly=2*np.pi):
    wlap, advection = solve_poisson(w[:, 1], Lx=Lx, Ly=Ly)
    dwdt = (w[:, 2] - w[:, 0]) / (2 * dt)

    x = torch.linspace(0, Lx, w.shape[-2], device=w.device)
    y = torch.linspace(0, Ly, w.shape[-1], device=w.device)
    _, Y = torch.meshgrid(x, y, indexing="ij")
    force = -4 * torch.cos(4 * Y) - 0.1 * w[:, 1]

    res = dwdt + advection - (wlap / Re) - force
    return res, [dwdt, advection, wlap]

class ResidualOp:
    def __init__(self, device="cpu", Nx=256, Ny=256, Re=1000, dt=1/32, Lx=2*np.pi, Ly=2*np.pi):
        self.Re = Re
        self.dt = dt
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.dx = Lx / Nx
        self.dy = Ly / Ny

        x = torch.linspace(0, Lx, Nx+1, device=device)[:-1]
        y = torch.linspace(0, Ly, Ny+1, device=device)[:-1]
        _, Y = torch.meshgrid(x, y, indexing="ij")
        self.const_force = (-4 * torch.cos(4 * Y)).reshape(1, 1, Nx, Ny)

        # Create the wavevector grid in Fourier space
        self.kx = torch.fft.fftfreq(Nx, self.dx).reshape(1, 1, Nx, 1).to(device) * 2 * torch.pi * 1j
        self.ky = torch.fft.fftfreq(Ny, self.dy).reshape(1, 1, 1, Ny).to(device) * 2 * torch.pi * 1j    
        
        # Negative Laplacian in Fourier space
        self.lap = (self.kx ** 2 + self.ky ** 2).to(device)
        self.lap[..., 0, 0] = 1.0
    
    def __call__(self, w):
        w_h = torch.fft.fft2(w[:, 1:2], dim=[2, 3])
        psi_h = -w_h / self.lap

        u_h = psi_h * self.ky
        v_h = -psi_h * self.kx
        wx_h = self.kx * w_h
        wy_h = self.ky * w_h
        wlap_h = self.lap * w_h

        u = torch.fft.ifft2(u_h, dim=[2, 3]).real
        v = torch.fft.ifft2(v_h, dim=[2, 3]).real
        wx = torch.fft.ifft2(wx_h, dim=[2, 3]).real
        wy = torch.fft.ifft2(wy_h, dim=[2, 3]).real
    
        wlap = torch.fft.ifft2(wlap_h, dim=[2, 3]).real
        dwdt = (w[:, 2:3] - w[:, 0:1]) / (2 * self.dt)
        advection = u * wx + v * wy
        force = self.const_force - 0.1 * w[:, 1:2]
    
        res = dwdt + advection - (wlap / self.Re) - force
        return res, [dwdt, advection, wlap]
