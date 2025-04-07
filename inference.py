import yaml
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
from sklearn.metrics import mean_squared_error

from utils import *
from diffusion import Diffusion
from models.networks import Model
from models.networks_baseline import Model_base
from datasets import KolmogorovFlowDataset, LucaFlowDataset
from models.network_swinir import SwinIR

from experiments import *
from pod import *
from wavelet_dict import *

class SuperResolution:
    def __init__(self, config_file, scale_factor, perc, model_num, model_epoch, seed, dataset_name, device_id=0):
        # Get current time to measure running time
        self.start_time = datetime.now()

        # Set random seed
        self.seed = seed
        fix_randomness(self.seed)

        # Load configuration
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        self.config = dict2namespace(self.config)

        # Set up device
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        print("-- Performing super resolution --")
        print("Device:", self.device)
        print("PID:", os.getpid())
        self.config.device = self.device
        self.config.seed = self.seed

        # Initialize model & diffusion
        self.scale_factor = scale_factor
        self.perc = perc
        self.model_num = model_num
        self.model_epoch = model_epoch
        self.diffusion = Diffusion(self.config)
        self.model = Model(self.config)
        checkpoint = torch.load(f"runs/{str(self.model_num).zfill(3)}/checkpoint{self.model_epoch}.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model.to(self.device)
        self.model.eval();

        # Load dataset (default is KolmogorovFlowDataset)
        self.dataset_name = dataset_name
        if self.dataset_name == "kolmogorov":
            self.dataset = KolmogorovFlowDataset(seed=self.seed)
        if self.dataset_name == "luca":
            self.dataset = LucaFlowDataset(seed=self.seed)
        print(f"Shape of the raw data: ", self.dataset.raw_data.shape)
        print(f"Shape of the data: ", self.dataset.data.shape)

    def set_parameters(self, **kwargs):
        """Allows modification of class attributes after initialization."""
        reload_model = False
        reload_dataset = False
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if key in ["model_num", "model_epoch"]:
                    reload_model = True
                if key == "seed":
                    fix_randomness(value)
                    reload_dataset = True
                if key == "dataset_name":
                    reload_dataset = True
            else:
                print(f"Warning: Attribute {key} does not exist.")
        
        if reload_model:
            print("Reloading model...")
            checkpoint_path = f"runs/{str(self.model_num).zfill(3)}/checkpoint{self.model_epoch}.pt"
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.model.to(self.device)
            self.model.eval()
        
        if reload_dataset:
            print("Reloading dataset...")
            if self.dataset_name == "kolmogorov":
                self.dataset = KolmogorovFlowDataset(seed=self.seed)
            if self.dataset_name == "luca":
                self.dataset = LucaFlowDataset(seed=self.seed)
            print(f"Shape of the raw data: ", self.dataset.raw_data.shape)
            print(f"Shape of the data: ", self.dataset.data.shape)
            
    
    def get_parameters(self):
        """Returns a dictionary with the current attributes and their values."""
        return {
            "scale_factor": self.scale_factor,
            "perc": self.perc,
            "model_num": self.model_num,
            "model_epoch": self.model_epoch,
            "seed": self.seed,
            "dataset": self.dataset,
            "device": self.device
        }

    def generate_sample(self, nsamples):
        noise = torch.randn((nsamples, 3, 256, 256))
        with torch.no_grad():
            for i in range(nsamples):
                print("Sample", i)
                x_noise = noise[i].unsqueeze(0).to(self.device)
                y_pred = self.diffusion.ddim(x_noise, self.model, 1000, 100)

                plt.figure(figsize=(10, 10))

                plt.title("Random Sample")
                plt.imshow(y_pred[0][1], cmap="inferno")
                plt.show()
                plt.axis('off')

    def save_plot(self, low_res, high_res, gt, filename="super_res_output.png"):
        plt.figure(figsize=(10, 10))

        plt.subplot(1, 3, 1)
        plt.title("Low-Resolution Input")
        plt.imshow(low_res, cmap="inferno")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Super-Resolved Output")
        plt.imshow(high_res, cmap="inferno")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Ground truth")
        plt.imshow(gt, cmap="inferno")
        plt.axis('off')
        #plt.colorbar(label="Value Range")

        # Save the plot as a PNG file
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Plot saved")

    def run_sr_diff(self, sample=None, run=None, time=1, mask_steps=100, mask_start=1000, w_mask=1, sig=0.008, dist=10):
        if sample is not None:
            if sample >= 0 and sample < self.dataset.data.shape[0]:
                dataset = self.dataset[sample].unsqueeze(0)
            else:
                print("Wrong sample selection")
        else:
            if time > 0 and time < self.dataset.raw_data.shape[1]-1 and run >= 0 and run < self.dataset.raw_data.shape[0]:
                dataset = self.dataset.raw_data[run][time-1:time+2].unsqueeze(0)
            else:
                print("Wrong time selection")
        Nx = dataset.shape[-2]
        Ny = dataset.shape[-1]
        X_upscaled = downscale_data(dataset, self.scale_factor)

        # Precompute diffuse masks
        diffuse_masks = torch.zeros(1, 3, Nx, Ny)
        ids = random.sample(range(Nx*Ny), int(Nx*Ny*w_mask))
        mask = diffuse_mask(ids, A=1, sig=sig, search_dist=dist, Nx=Nx, Ny=Ny)
        diffuse_masks[0] = torch.tensor(mask, dtype=torch.float).unsqueeze(0).repeat(3, 1, 1)

        x = X_upscaled.to(self.device)
        y = dataset.to(self.device)
        noise = torch.randn(x.shape, device=x.device)
        diff_mask = diffuse_masks[0].unsqueeze(0).to(self.device)

        mask_pred, _ = self.diffusion.ddim_mask(noise, self.model, x, mask_start, mask_steps, w_mask=0, diff_mask=diff_mask)

        self.save_plot(x[0][1].cpu().detach().numpy(), mask_pred[0][1].cpu().detach().numpy(), y[0][1].cpu().detach().numpy())

    def run_sparse_diff(self, interp, sample=None, run=None, time=1, mask_steps=100, mask_start=1000, w_mask=0, sig=0.038):
        if sample is not None:
            if sample >= 0 and sample < self.dataset.data.shape[0]:
                dataset = self.dataset[sample].unsqueeze(0)
            else:
                print("Wrong sample selection")
        else:
            if time > 0 and time < self.dataset.raw_data.shape[1]-1 and run >= 0 and run < self.dataset.raw_data.shape[0]:
                dataset = self.dataset.raw_data[run][time-1:time+2].unsqueeze(0)
            else:
                print("Wrong time selection")
        Nx = dataset.shape[-2]
        Ny = dataset.shape[-1]
        X_vals, sampled_ids = interpolate_dataset(dataset, self.perc, method=interp)

        # Precompute diffuse masks
        diffuse_masks = torch.zeros(len(sampled_ids), 3, Nx, Ny).to(self.device)
        for i in range(len(sampled_ids)):
            ids = list(sampled_ids[i]) + random.sample(range(Nx*Ny), int(Nx*Ny*w_mask))
            mask = diffuse_mask(ids, A=1, sig=sig, Nx=Nx, Ny=Ny)
            diffuse_masks[i] = torch.tensor(mask, dtype=torch.float).unsqueeze(0).repeat(3, 1, 1)

        x = X_vals.to(self.device)
        y = dataset.to(self.device)
        noise = torch.randn(x.shape, device=x.device)

        mask_pred, _ = self.diffusion.ddim_mask(noise.clone(), self.model, x.clone(), mask_start, mask_steps, w_mask=0, diff_mask=diffuse_masks[0].unsqueeze(0))

        # Compute MSE and MAE
        mask_pred_np = mask_pred.cpu().detach().numpy()
        y_np = y.cpu().detach().numpy()
        mse_error = np.mean((mask_pred_np - y_np) ** 2)
        print(f"Mean Squared Error between mask_pred and y: {mse_error}")

        self.save_plot(x[0][1].cpu().detach().numpy(), mask_pred[0][1].cpu().detach().numpy(), y[0][1].cpu().detach().numpy())
        
    def run_sparse_pod(self, sample=None, run=None, time=1, num_modes=200, perc=5):
        if sample is not None:
            if sample >= 0 and sample < self.dataset.data.shape[0]:
                snapshot = self.dataset[sample].unsqueeze(0)
            else:
                print("Wrong sample selection")
        else:
            if time > 0 and time < self.dataset.raw_data.shape[1]-1 and run >= 0 and run < self.dataset.raw_data.shape[0]:
                snapshot = self.dataset.raw_data[run][time-1:time+2].unsqueeze(0)
            else:
                print("Wrong time selection")
        
        ntrain = 3000 
        dataset = self.dataset.data
        print(dataset.shape)
        Nx, Ny = dataset.shape[2], dataset.shape[3]
        data_matrix = reshape_data(dataset, ntrain).T
        U, S, Vt = compute_pod(data_matrix)
        print("POD computed")
        
        C, indices = create_selection_matrix(Nx, Ny, perc/100)
        snapshot = snapshot[:, 1, :, :].squeeze(0)
        interpolated_snapshot = direct_interpolation(snapshot, indices, Nx, Ny)
        reconstruct_snapshot = perform_superresolution(U, num_modes, C, snapshot, Nx, Ny)
        save_plot(interpolated_snapshot, reconstruct_snapshot, snapshot)
        
    def load_dataset_unet(self, dataset_path):
        low_res_images = np.load(os.path.join(dataset_path, 'low_res_images.npy'))
        high_res_images = np.load(os.path.join(dataset_path, 'high_res_images.npy'))
        return low_res_images, high_res_images

    def run_sparse_unet(self, sample=None, perc=5):
        if perc == 5:
            X, Y = self.load_dataset_unet("dataset")
        if perc == 1:
            X, Y = self.load_dataset_unet("dataset_verysparse")
        train_X, val_X, test_X = X[:3000], X[3000:3240], X[3240:] 
        train_Y, val_Y, test_Y = Y[:3000], Y[3000:3240], Y[3240:]
        
        # Model config (example)
        class Config_base:
            class model:
                ch = 64
                out_ch = 3 
                ch_mult = [1, 1, 1, 2]
                num_res_blocks = 1
                attn_resolutions = [16, ]
                dropout = 0.0
                in_channels = 3  
                padding_mode = 'reflect'
                resamp_with_conv = True
            
            class data:
                image_size = 256

        # Initialize model
        config = Config_base()
        model = Model_base(config)
        if perc == 5:
            model.load_state_dict(torch.load("checkpoints_unet/epoch_701.pth"))
        if perc == 1:
            model.load_state_dict(torch.load("checkpoints_unet_verysparse/epoch_991.pth"))
        model.eval()
        model.to(self.device)
        
        # Select image
        low_res_input, high_res_gt = test_X[sample], test_Y[sample]
        low_res_tensor = torch.tensor(low_res_input, dtype=torch.float32).unsqueeze(0).to(self.device)  # Shape: (1, C, H, W)

        # Forward pass through the model
        with torch.no_grad():
            predicted_high_res = model(low_res_tensor)

        # Remove batch dimension and convert back to numpy
        predicted_high_res = predicted_high_res.squeeze(0).cpu().numpy()
        low_res_input = low_res_input  # Already in numpy
        high_res_gt = high_res_gt  # Already in numpy

        # Plot results with a shared colorbar
        low_res_img = low_res_input[1, :, :]
        high_res_gt_img = high_res_gt[1, :, :]
        predicted_img = predicted_high_res[1, :, :]
        
        save_plot(low_res_img, predicted_img, high_res_gt_img)
        
    def run_sparse_swin(self, sample=None, perc=5):
        if perc == 5:
            X, Y = self.load_dataset_unet("dataset")
        if perc == 1:
            X, Y = self.load_dataset_unet("dataset_verysparse")
        train_X, val_X, test_X = X[:3000], X[3000:3240], X[3240:] 
        train_Y, val_Y, test_Y = Y[:3000], Y[3000:3240], Y[3240:]
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        upscale = 1
        window_size = 8
        model = SwinIR(
            upscale=upscale,
            img_size=(256, 256),
            window_size=window_size,
            img_range=1.,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler=None
        )
        model = model.to(device)
        model.load_state_dict(torch.load("checkpoints_swin_5/swinir_epoch_10.pth"))
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print("Model initialized")
        
        # Select image
        low_res_input, high_res_gt = test_X[sample], test_Y[sample]
        low_res_tensor = torch.tensor(low_res_input, dtype=torch.float32).unsqueeze(0).to(self.device)  # Shape: (1, C, H, W)

        # Forward pass through the model
        with torch.no_grad():
            predicted_high_res = model(low_res_tensor)

        # Remove batch dimension and convert back to numpy
        predicted_high_res = predicted_high_res.squeeze(0).cpu().numpy()
        low_res_input = low_res_input  # Already in numpy
        high_res_gt = high_res_gt  # Already in numpy

        # Plot results with a shared colorbar
        low_res_img = low_res_input[1, :, :]
        high_res_gt_img = high_res_gt[1, :, :]
        predicted_img = predicted_high_res[1, :, :]
        
        save_plot(low_res_img, predicted_img, high_res_gt_img)
        
    def perform_experiment(self, mask, type, param, num_modes=None, nsamples=10):
        if num_modes is None:
            if mask == "std" and type == "sr":
                if param == 4:
                    mask_experiment([self.model_num], nsamples=nsamples, model_epoch=self.model_epoch, cuda=0, scale_factor=4, art_steps=36, art_start=160, art_K=3, mask_steps=100, mask_start=1000, w_mask=1)
                if param == 8:
                    mask_experiment([self.model_num], nsamples=nsamples, model_epoch=self.model_epoch, cuda=0, scale_factor=8, art_steps=36, art_start=320, art_K=1, mask_steps=80, mask_start=800, w_mask=0.5)
            
            if mask == "diff" and type == "sr":
                if param == 4:
                    diffuse_mask_experiment([self.model_num], nsamples=nsamples, model_epoch=self.model_epoch, scale_factor=4, cuda=0, art_steps=36, art_start=160, art_K=3, mask_steps=100, mask_start=1000, w_mask=1, sig=0.008, dist=10)
                if param == 8:
                    diffuse_mask_experiment([self.model_num], nsamples=nsamples, model_epoch=self.model_epoch, scale_factor=4, cuda=0, art_steps=36, art_start=320, art_K=1, mask_steps=80, mask_start=800, w_mask=0.5, sig=0.008, dist=10)
                    
            if mask == "std" and type == "sparse":
                if param == 5:
                    mask_sparse_experiment([self.model_num], nsamples=nsamples, model_epoch=self.model_epoch, perc=5, art_steps=36, art_start=160, art_K=3, mask_steps=100, mask_start=1000, w_mask=0.15)
                if param == 1:
                    mask_sparse_experiment([self.model_num], nsamples=nsamples, model_epoch=self.model_epoch, perc=1.5625, art_steps=36, art_start=400, art_K=1, mask_steps=100, mask_start=1000, w_mask=0.15)
                    
            if mask == "diff" and type == "sparse":
                if param == 5:
                    diffuse_mask_sparse_experiment([self.model_num], nsamples=nsamples, model_epoch=self.model_epoch, cuda=0, perc=5, art_steps=36, art_start=160, art_K=3, mask_steps=100, mask_start=1000, w_mask=0, sigs=0.038, interp="nearest")
                if param == 1:
                    diffuse_mask_sparse_experiment([self.model_num], nsamples=nsamples, model_epoch=self.model_epoch, cuda=0, perc=1.5625, art_steps=36, art_start=400, art_K=1, mask_steps=100, mask_start=1000, w_mask=0, sigs=0.052, interp="nearest")
                    
            if mask == "unet":
                if param == 5:
                    X, Y = self.load_dataset_unet("dataset")
                if param == 1:
                    X, Y = self.load_dataset_unet("dataset_verysparse")
                    
                train_X, val_X, test_X = X[:3000], X[3000:3240], X[3240:] 
                train_Y, val_Y, test_Y = Y[:3000], Y[3000:3240], Y[3240:]
                
                num_samples = 10
                mse_list = []
                random_indices = random.sample(range(len(test_X)), num_samples)

                # Create a batch of low-resolution images
                low_res_batch = torch.tensor([test_X[idx] for idx in random_indices], dtype=torch.float32).to(self.device)  # Shape: (num_samples, C, H, W)
                high_res_batch = np.array([test_Y[idx] for idx in random_indices])  # Ground truth batch
                
                # Model config (example)
                class Config_base:
                    class model:
                        ch = 64
                        out_ch = 3 
                        ch_mult = [1, 1, 1, 2]
                        num_res_blocks = 1
                        attn_resolutions = [16, ]
                        dropout = 0.0
                        in_channels = 3  
                        padding_mode = 'reflect'
                        resamp_with_conv = True
                    
                    class data:
                        image_size = 256

                # Initialize model
                config = Config_base()
                model = Model_base(config)
                if param == 5:
                    model.load_state_dict(torch.load("checkpoints_unet/epoch_701.pth"))
                if param == 1:
                    model.load_state_dict(torch.load("checkpoints_unet_verysparse/epoch_991.pth"))
                model.eval()
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)

                start_time = time.time()
                with torch.no_grad():
                    predicted_high_res_batch = model(low_res_batch)  # Shape: (num_samples, C, H, W)
                end_time = time.time()

                predicted_high_res_batch = predicted_high_res_batch.cpu().numpy()
                high_res_batch = high_res_batch  

                for i in range(num_samples):
                    mse = mean_squared_error(high_res_batch[i].flatten(), predicted_high_res_batch[i].flatten())
                    mse_list.append(mse)

                avg_mse = np.mean(mse_list)
                std_mse = np.std(mse_list)
                time_spent = end_time - start_time

                # Print results
                print(f"MSE over {num_samples} random test images: {avg_mse:.4f} +/- {std_mse:.4f}")
                print(f"Time spent applying the model: {time_spent:.4f} seconds")
                
            if mask == "swin":
                if param == 5:
                    X, Y = self.load_dataset_unet("dataset")
                if param == 1:
                    X, Y = self.load_dataset_unet("dataset_verysparse")
                    
                train_X, val_X, test_X = X[:3000], X[3000:3240], X[3240:] 
                train_Y, val_Y, test_Y = Y[:3000], Y[3000:3240], Y[3240:]
                
                num_samples = 10
                mse_list = []
                random_indices = random.sample(range(len(test_X)), num_samples)

                # Create a batch of low-resolution images
                low_res_batch = torch.tensor([test_X[idx] for idx in random_indices], dtype=torch.float32).to(self.device)  # Shape: (num_samples, C, H, W)
                high_res_batch = np.array([test_Y[idx] for idx in random_indices])  # Ground truth batch
                
                # Load model
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                upscale = 1
                window_size = 8
                model = SwinIR(
                    upscale=upscale,
                    img_size=(256, 256),
                    window_size=window_size,
                    img_range=1.,
                    depths=[6, 6, 6, 6],
                    embed_dim=60,
                    num_heads=[6, 6, 6, 6],
                    mlp_ratio=2,
                    upsampler=None
                )
                model = model.to(device)
                model.load_state_dict(torch.load("checkpoints_swin_5/swinir_epoch_10.pth"))
                model.eval()
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                print("Model initialized")
                
                start_time = time.time()
                with torch.no_grad():
                    predicted_high_res_batch = model(low_res_batch)  # Shape: (num_samples, C, H, W)
                end_time = time.time()

                predicted_high_res_batch = predicted_high_res_batch.cpu().numpy()
                high_res_batch = high_res_batch  

                for i in range(num_samples):
                    mse = mean_squared_error(high_res_batch[i].flatten(), predicted_high_res_batch[i].flatten())
                    mse_list.append(mse)

                avg_mse = np.mean(mse_list)
                std_mse = np.std(mse_list)
                time_spent = end_time - start_time

                # Print results
                print(f"MSE over {num_samples} random test images: {avg_mse:.4f} +/- {std_mse:.4f}")
                print(f"Time spent applying the model: {time_spent:.4f} seconds")
                
        if num_modes is not None and mask == "pod":
            ntrain = 3000 
            dataset = self.dataset.data
            print(dataset.shape)
            Nx, Ny = dataset.shape[2], dataset.shape[3]
            data_matrix = reshape_data(dataset, ntrain).T
            U, S, Vt = compute_pod(data_matrix)
            print("POD computed")
            
            C, indices = create_selection_matrix(Nx, Ny, param/100)
            
            snapshots_test = dataset[-nsamples:, 1, :, :]
            print(snapshots_test.shape)

            start_time = time.time()
            reconstructed_snapshots = reconstruct_all_snapshots_gappy(U, num_modes, C, snapshots_test, Nx, Ny)
            stop_time = time.time()
            print(f"Time taken for gappy POD: {stop_time - start_time} seconds")

            mse_errors = [mean_squared_error(snapshots_test[i].flatten(), reconstructed_snapshots[i].flatten()) for i in range(len(snapshots_test))]

            average_mse = np.mean(mse_errors)
            std_mse = np.std(mse_errors)
            print(f"MSE over test snapshots (Gappy POD): {average_mse:.4f} +/- {std_mse:.4f}")
            
        if num_modes is not None and mask == "wavelet":
            dataset = KolmogorovFlowDataset(shu=True, shuffle=True)
            dataset = dataset.data  # Shape: (4240, 3, 256, 256): first 3000 for training, last 1240 for testing
            nsamples = 3000

            Nx, Ny = dataset.shape[2], dataset.shape[3]
            data_matrix = reshape_data_wavelet(dataset, nsamples)

            # Step 1: Create wavelet dictionary
            dict_model = learn_wavelet_dictionary(data_matrix, n_components=100)

            # Step 2: Sample a sparse CFD field
            N = 10
            mse_values = []

            for i in range(N):
                test_sample = dataset[nsamples + i, 1, :, :]  # Select sample i from the dataset
                sparse_field, mask = sample_sparse_field(test_sample, perc=1.5625)

                # Step 3: Reconstruct the field
                reconstructed_field = wavelet_reconstruct(sparse_field, mask, dict_model)

                # Step 4: Compute MSE for this sample
                mse = mean_squared_error(test_sample.flatten(), reconstructed_field.flatten())
                mse_values.append(mse)

            # Step 5: Calculate mean and std of MSE
            mean_mse = np.mean(mse_values)
            std_mse = np.std(mse_values)

            print(f"Mean MSE: {mean_mse:.4f}")
            print(f"Standard Deviation of MSE: {std_mse:.4f}")
                
                

# ====================== MAIN FUNCTION ======================
if __name__ == "__main__":
    super_res = SuperResolution(
        config_file=os.path.join('configs', "kmflow_re1000_rs256.yml"), 
        scale_factor=8, 
        perc = 1.5625/100,
        model_num=56, 
        model_epoch=1000, 
        seed=1234, 
        dataset_name="luca"
    )

    super_res.set_parameters(scale_factor=4, perc=5/100, model_num=54, dataset_name="kolmogorov")
    print("Current parameters:")
    print(super_res.get_parameters())

    #super_res.generate_sample(3)
    #super_res.run_sr_diff(sample=0)
    #super_res.run_sparse_diff("nearest", sample=0)
    #super_res.run_sparse_pod(sample=0, num_modes=200, perc=5)
    #super_res.run_sparse_unet(10, 5)
    
    super_res.perform_experiment("unet", "sparse", 5)