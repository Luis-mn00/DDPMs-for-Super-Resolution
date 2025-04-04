import numpy as np
import torch
import random
import os
from utils import StdScaler


class WakeFlowDataset:
    def __init__(self, filename, norm=True, trim_start=0):
        raw_data = np.load(filename)
        self.data = torch.Tensor(raw_data["data"])
        self.constants = raw_data["constants"].squeeze(1)
        self.obstacle_mask = torch.Tensor(raw_data["obstacle_mask"])

        self.nexp, self.ntimesteps, self.ch, self.ny, self.nx = self.data.shape

        self.x_scaler = StdScaler(0.4416, 0.2182)
        self.y_scaler = StdScaler(-0.0002, 0.1836)
        self.p_scaler = StdScaler(0.0007, 0.0047)

        self.Re_scaler = StdScaler(550, 250)
        self.constants = self.Re_scaler(self.constants)

        self.norm = norm
        self.trim_start = trim_start

    def vorticity(self, X):
        u, v = X[0], X[1]

        dx = 1 / self.nx
        dy = 2 / self.ny

        # First order
        # dudy = (np.roll(u, 1, axis=2) - u) / dy
        # dvdx = (np.roll(v, 1, axis=1) - v) / dx

        # Second order
        # f'(x) = f(x + h) - f(x - h) / 2*dx
        # next_u = np.roll(u, 1, axis=2)
        # next_u[:,:, 0] = 0
        # prev_u = np.roll(u, -1, axis=2)
        # prev_u[:,:,-1] = 0
        # dudy = (next_u - prev_u) / (2 * dy)
        
        # next_v = np.roll(v, 1, axis=1)
        # next_v[:,:, 0] = 0
        # prev_v = np.roll(v, -1, axis=1)
        # prev_v[:,:,-1] = 0
        # dvdx = (next_v - prev_v) / (2 * dx)

        # Fourth order
        # f'(x) = -f(x + 2h) + 8f(x + h) - 8f(x - h) + f(x - 2h) / 12*dx
        next1_u = np.roll(u, 1, axis=2)
        next1_u[:,:, 0] = 0
        next2_u = np.roll(u, 2, axis=2)
        next2_u[:,:, 0:2] = 0
        prev1_u = np.roll(u, -1, axis=2)
        prev1_u[:,:,-1] = 0
        prev2_u = np.roll(u, -2, axis=2)
        prev2_u[:,:,-3:-1] = 0
        dudy = (-next2_u + 8*next1_u - 8*prev1_u + prev2_u) / (12 * dy)
        
        next1_v = np.roll(v, 1, axis=1)
        next1_v[:,:, 0] = 0
        next2_v = np.roll(v, 2, axis=1)
        next2_v[:,:, 0:2] = 0
        prev1_v = np.roll(v, -1, axis=1)
        prev1_v[:,:,-1] = 0
        prev2_v = np.roll(v, -2, axis=1)
        prev2_v[:,:,-3:-1] = 0
        dvdx = (-next2_v + 8*next1_v - 8*prev1_v + prev2_v) / (12 * dx)

        return dudy - dvdx


    def scale(self, X):
        res = X.clone()
        res[0] = self.x_scaler(X[0])
        res[1] = self.y_scaler(X[1])
        res[2] = self.p_scaler(X[2])
        return res


    def scale_inverse(self, X):
        res = X.clone()
        if len(res.shape) == 4: 
            res[:, 0] = self.x_scaler.inverse(X[:, 0])
            res[:, 1] = self.y_scaler.inverse(X[:, 1])
            res[:, 2] = self.p_scaler.inverse(X[:, 2])

        else:
            res[0] = self.x_scaler.inverse(X[0])
            res[1] = self.y_scaler.inverse(X[1])
            res[2] = self.p_scaler.inverse(X[2])
        
        return res


    def __len__(self):
        return (self.ntimesteps - self.trim_start) * self.nexp


    def __getitem__(self, idx):
        _idx = idx[0] if type(idx) is tuple else idx
        restid = idx[1:] if type(idx) is tuple and len(idx) > 1 else None

        if _idx < 0 or _idx >= self.__len__():
            raise IndexError("Wrong index")

        iexp = _idx // (self.ntimesteps - self.trim_start)
        isample = (_idx % (self.ntimesteps - self.trim_start)) + self.trim_start
        sample = self.data[iexp, isample].clone() # (3, 128, 64)
        
        Re = self.constants[iexp]
        Re_channel = torch.full((1, 128, 64), Re)
        sample = torch.cat((sample, Re_channel), dim=0)

        if self.norm:
            sample = self.scale(sample)

        if restid:
            return torch.Tensor(sample[restid])
        else: 
            return torch.Tensor(sample)

        

class KolmogorovFlowDataset:
    def __init__(self, norm=True, shuffle=True, n_concurrent=3, shu=True, seed=1234, size=-1, data_folder="data"):
        # Setting random seed for reproducibility
        torch.torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        if shu: # Re=1000, 256x256
            filename = os.path.join(data_folder, "kf_2d_re1000_256_40seed.npy")
            self.raw_data = torch.Tensor(np.load(filename))
            self.scaler = StdScaler(0.0, 4.7852)

        else: # Re=100, 160x160
            filename = os.path.join(data_folder, "2d_phy_kolm_flow__num_spatial_dims=2_train.npy")
            self.raw_data = torch.Tensor(np.load(filename)).squeeze(2)
            self.scaler = StdScaler(0.0, 3.5213)

        self.nexp, self.ntimesteps, self.ny, self.nx = self.raw_data.shape
        self.norm = norm
        self.n_concurrent = n_concurrent

        samples_xexp = self.ntimesteps // n_concurrent
        samples_extra = self.ntimesteps % n_concurrent

        if samples_extra > 0: 
            self.raw_data = self.raw_data[:, :-samples_extra]

        self.data = self.raw_data.reshape(self.nexp, samples_xexp, n_concurrent, self.ny, self.nx)
        self.data = self.data.reshape(self.nexp * samples_xexp, n_concurrent, self.ny, self.nx)

        if shuffle:
            indices = torch.randperm(self.data.size(0))
            self.data = self.data[indices]

        if norm:
            self.data = self.scaler(self.data)

        self.size = self.data.shape[0] if size < 0 else size
        self.data = self.data[:self.size]


    def __len__(self):
        return self.size
    

    def __getitem__(self, idx):
        return self.data[idx].clone()


    def to_numpy(self):
        return self.data.numpy()
    

    def split(self, perc=0.75):
        # Return train and test sets
        N = int(self.__len__() * perc)
        return self.data[:N].clone(), self.data[N:].clone()

class LucaFlowDataset:
    def __init__(self, norm=True, shuffle=True, seed=1234, size=-1, data_folder="data"):
        torch.torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        filename = os.path.join(data_folder, "Ret180_192x65x192_fluct_c3_zy8_0.npz")
        file = np.load(filename)
        print(file["samples"].shape)
        self.raw_data = torch.Tensor(file["samples"])
        self.scaler = StdScaler(file["mean"], file["scale"])

        self.nsamples, self.n_concurrent, self.nx, self.ny = self.raw_data.shape
        self.raw_data = self.raw_data[:, :, :, :self.ny-1]
        self.norm = norm

        if size > 0:
            self.data = self.raw_data[:size]
        else:
            self.data = self.raw_data

        if shuffle:
            indices = torch.randperm(self.data.size(0))
            self.data = self.data[indices]

        if norm:
            self.data = self.scaler(self.data)

        self.size = self.data.shape[0] if size < 0 else size


    def __len__(self):
        return self.size
    

    def __getitem__(self, idx):
        return self.data[idx].clone()


    def to_numpy(self):
        return self.data.numpy()
    

    def split(self, perc=0.75):
        # Return train and test sets
        N = int(self.__len__() * perc)
        return self.data[:N].clone(), self.data[N:].clone()
