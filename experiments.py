import yaml
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
import time

from utils import *
from diffusion import Diffusion
from models.networks import Model
from datasets import KolmogorovFlowDataset
from residuals import ResidualOp

def residual_of_generated(models, nsamples=10, model_epoch=200, cuda=1):
    print("Loading config")
    with open(os.path.join('configs', "kmflow_re1000_rs256.yml"), 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    device = torch.device(f'cuda:{cuda}') if torch.cuda.is_available() else torch.device('cpu')
    seed = 1234
    print("Device", device)
    print("PID", os.getpid())
    print("Testing models:", models)

    config.device = device
    config.seed = seed
    residual_op = ResidualOp(device)


    for model_num in models:
        print("-" * 30)
        print("Testing model", model_num)
        model = Model(config)
        diffusion = Diffusion(config)
        cp = torch.load(f"runs/{str(model_num).zfill(3)}/checkpoint{model_epoch}.pt", map_location=device)
        model.load_state_dict(cp["model_state_dict"])
        model.to(device)
        model.eval();

        dataset = KolmogorovFlowDataset(shu=True)

        # Average residual of model samples
        l1_loss = np.zeros(nsamples) 
        rmse_loss = np.zeros(nsamples)
        sample_size = 256

        for i in range(nsamples):
            print("Batch", i)
            noise = torch.randn((1, 3, sample_size, sample_size), device=device)
            y_pred = diffusion.ddpm(noise, model, 1000, plot_prog=False)
            res, _ = residual_op(dataset.scaler.inverse(y_pred))
            l1_loss[i] = torch.mean(abs(res))
            rmse_loss[i] = torch.sqrt(torch.mean(res**2))

        # l1_loss /= nbatches
        # rmse_loss /= nbatches
        print("Statics out of", nsamples)
        print(f"L1 residual: {np.mean(l1_loss):.2f} +/- {np.std(l1_loss):.2f}") 
        print(f"L2 residual: {np.mean(rmse_loss):.2f} +/- {np.std(rmse_loss):.2f}") 


def test_MSE(models, nsamples=10, model_epoch=200, cuda=1):
    print("-- MSE TEST --")
    print("Loading config")
    with open(os.path.join('configs', "kmflow_re1000_rs256.yml"), 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    device = torch.device(f'cuda:{cuda}') if torch.cuda.is_available() else torch.device('cpu')
    seed = 1234
    print("Device", device)
    print("PID", os.getpid())
    print("Testing models:", models)
    print("Epochs:", model_epoch)

    config.device = device
    config.seed = seed

    for model_num in models:
        print("-" * 30)
        print("Testing model", model_num)
        model = Model(config)
        diffusion = Diffusion(config)

        model.load_state_dict(torch.load(f"runs/{str(model_num).zfill(3)}/model{model_epoch}.pickle", map_location=device))
        model.to(device)
        model.eval();

        dataset = KolmogorovFlowDataset(shu=True)

        # Average residual of model samples
        l1_loss = np.zeros(nsamples) 
        l2_loss = np.zeros(nsamples)

        for i in range(nsamples):
            index = i % 1000
            y = dataset[-index].unsqueeze(0).to(device)
            t = torch.randint(0, diffusion.num_timesteps, size=(1,), device=y.device)
            x_t, noise = diffusion.forward(y, t)
            e_pred = model(x_t, t)
            l2_loss[i] = (noise - e_pred).square().mean()

        print("Statics out of", nsamples)
        print(f"L2 residual: {np.mean(l2_loss):.4f} +/- {np.std(l2_loss):.4f}") 


def mask_experiment(
        models, nsamples=10, model_epoch=200, scale_factor=4, cuda=1,
        art_steps=30, art_start=160, art_K=3,
        mask_steps=100, mask_start=1000, w_mask=1, 
    ):
    
    with open(os.path.join('configs', "kmflow_re1000_rs256.yml"), 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    device = torch.device(f'cuda:{cuda}') if torch.cuda.is_available() else torch.device('cpu')
    seed = 1234
    fix_randomness(seed)
    print("Device", device)
    print("PID", os.getpid())
    print("N samples:", nsamples)

    config.device = device
    config.seed = seed

    for model_num in models:
        print("-" * 30)
        print("Testing model", model_num)
        model = Model(config)
        diffusion = Diffusion(config)
        
        cp = torch.load(f"runs/{str(model_num).zfill(3)}/checkpoint{model_epoch}.pt", map_location=device)
        model.load_state_dict(cp["model_state_dict"])
        model.to(device)
        model.eval();

        dataset = KolmogorovFlowDataset(shu=True)
        test_dataset = dataset[-1000:] 
        X_upscaled = downscale_data(test_dataset, scale_factor)
       
        residual_op = ResidualOp(device)

        mask_samples    = []
        article_samples = []
        ref_samples     = []
        l1_res_mask     = []
        l1_res_article  = []

        for i in range(nsamples):
            x     = X_upscaled[i].unsqueeze(0).to(device)
            y     = test_dataset[i].unsqueeze(0).to(device)
            noise = torch.randn(x.shape, device=x.device)

            mask_pred, _ = diffusion.ddim_mask(noise, model, x, mask_start, mask_steps, w_mask=w_mask)
            article_pred = diffusion.ddim_article(x, model, art_start, art_steps, K=art_K)

            ref_samples.append(y[0].cpu())    
            mask_samples.append(mask_pred[0].cpu())
            article_samples.append(article_pred[0].cpu())
            
            res, _ = residual_op(dataset.scaler.inverse(mask_pred))
            l1_res_mask.append(torch.mean(abs(res)).item())

            res, _ = residual_op(dataset.scaler.inverse(article_pred))
            l1_res_article.append(torch.mean(abs(res)).item())

        l2_losses = np.mean((np.array(ref_samples) - np.array(mask_samples))**2, axis=(-1,-2,-3))
        print("Mask method:")
        print(f"\tPixel-wise L2 error: {np.mean(l2_losses):.4f} +/- {np.std(l2_losses):.4f}")
        print(f"\tResidual L1 error: {np.mean(l1_res_mask):.4f} +/- {np.std(l1_res_mask):.4f}") 

        l2_losses = np.mean((np.array(ref_samples) - np.array(article_samples))**2, axis=(-1,-2,-3))
        print("Article method:")
        print(f"\tPixel-wise L2 error: {np.mean(l2_losses):.4f} +/- {np.std(l2_losses):.4f}")
        print(f"\tResidual L1 error: {np.mean(l1_res_article):.4f} +/- {np.std(l1_res_article):.4f}") 


def diffuse_mask_experiment(
        models, nsamples=10, model_epoch=200, scale_factor=4, cuda=1,
        art_steps=30, art_start=160, art_K=3,
        mask_steps=100, mask_start=1000, w_mask=1,
        sig=0.008, dist=10
    ):
    
    with open(os.path.join('configs', "kmflow_re1000_rs256.yml"), 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    device = torch.device(f'cuda:{cuda}') if torch.cuda.is_available() else torch.device('cpu')
    seed = 1234
    fix_randomness(seed)
    print("Device", device)
    print("PID", os.getpid())
    print("N samples:", nsamples)

    config.device = device
    config.seed = seed

    dataset = KolmogorovFlowDataset(shu=True)
    test_dataset = dataset[-1000:] 
    X_upscaled = downscale_data(test_dataset[0:nsamples], scale_factor)

    # Precompute diffuse masks
    diffuse_masks = torch.zeros(nsamples, 3, 256, 256)
    for i in range(nsamples):
        ids = random.sample(range(256**2), int(256**2*w_mask))
        mask = diffuse_mask(ids, A=1, sig=sig, search_dist=dist)
        diffuse_masks[i] = torch.tensor(mask, dtype=torch.float).unsqueeze(0).repeat(3, 1, 1)
    
    for model_num in models:
        print("-" * 30)
        print("Testing model", model_num)
        residual_op = ResidualOp(device)

        model = Model(config)
        diffusion = Diffusion(config)

        cp = torch.load(f"runs/{str(model_num).zfill(3)}/checkpoint{model_epoch}.pt", map_location=device)
        model.load_state_dict(cp["model_state_dict"])
        model.to(device)
        model.eval();
       
        mask_samples    = []
        article_samples = []
        ref_samples     = []
        l1_res_mask     = []
        l1_res_article  = []

        for i in range(nsamples):
            x     = X_upscaled[i].unsqueeze(0).to(device)
            y     = test_dataset[i].unsqueeze(0).to(device)
            noise = torch.randn(x.shape, device=x.device)
            diff_mask = diffuse_masks[i].unsqueeze(0).to(device)

            mask_pred, _ = diffusion.ddim_mask(noise, model, x, mask_start, mask_steps, w_mask=0, diff_mask=diff_mask)
            article_pred = diffusion.ddim_article(x, model, art_start, art_steps, K=art_K)

            ref_samples.append(y[0].cpu())    
            mask_samples.append(mask_pred[0].cpu())
            article_samples.append(article_pred[0].cpu())
            
            res, _ = residual_op(dataset.scaler.inverse(mask_pred))
            l1_res_mask.append(torch.mean(abs(res)).item())

            res, _ = residual_op(dataset.scaler.inverse(article_pred))
            l1_res_article.append(torch.mean(abs(res)).item())

        l2_losses = np.mean((np.array(ref_samples) - np.array(mask_samples))**2, axis=(-1,-2,-3))
        print("Mask method:")
        print(f"\tPixel-wise L2 error: {np.mean(l2_losses):.4f} +/- {np.std(l2_losses):.4f}")
        print(f"\tResidual L1 error: {np.mean(l1_res_mask):.4f} +/- {np.std(l1_res_mask):.4f}") 

        l2_losses = np.mean((np.array(ref_samples) - np.array(article_samples))**2, axis=(-1,-2,-3))
        print("Article method:")
        print(f"\tPixel-wise L2 error: {np.mean(l2_losses):.4f} +/- {np.std(l2_losses):.4f}")
        print(f"\tResidual L1 error: {np.mean(l1_res_article):.4f} +/- {np.std(l1_res_article):.4f}") 


def mask_sparse_experiment(
        models, nsamples=10, model_epoch=200, perc=5, cuda=0,
        art_steps=30, art_start=160, art_K=3,
        mask_steps=100, mask_start=1000, w_mask=1
    ):
    
    with open(os.path.join('configs', "kmflow_re1000_rs256.yml"), 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    device = torch.device(f'cuda:{cuda}') if torch.cuda.is_available() else torch.device('cpu')
    seed = 123
    fix_randomness(seed)
    print("Device", device)
    print("PID", os.getpid())
    print("N samples:", nsamples)
    print("Params:", art_steps, art_start, art_K, mask_steps, mask_start, w_mask)
    config.device = device
    config.seed = seed

    for model_num in models:
        print("-" * 30)
        print("Testing model", model_num)
        model = Model(config)
        diffusion = Diffusion(config)

        cp = torch.load(f"runs/{str(model_num).zfill(3)}/checkpoint{model_epoch}.pt", map_location=device)
        model.load_state_dict(cp["model_state_dict"])
        model.to(device)
        model.eval();
        
        residual_op = ResidualOp(device)

        dataset = KolmogorovFlowDataset(shu=True)
        test_dataset = dataset[-1000:] 
        X_vals, sampled_ids = interpolate_dataset(test_dataset[0:nsamples], perc/100)
        losses_a = []
        losses_m = []
        residuals_a = []
        residuals_m = []

        total_time = 0

        for i in range(nsamples):
            x     = X_vals[i].unsqueeze(0).to(device)
            y     = test_dataset[i].unsqueeze(0).to(device)
            noise = torch.randn(x.shape, device=x.device)

            mask = torch.zeros(256,256).flatten()
            mask[sampled_ids[i]] = 1
            mask = mask.reshape(256,256).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)

            x_pred_a    = diffusion.ddim_article(x.clone(), model, art_start, art_steps, K=art_K)
            
            time_start = time.time()
            x_pred_m, _ = diffusion.ddim_mask(noise.clone(), model, x.clone(), mask_start, mask_steps, w_mask=w_mask, _mask=mask)
            time_end = time.time()
            total_time += time_end - time_start

            losses_a.append(l2_loss_fn(x_pred_a[0], y).item())
            losses_m.append(l2_loss_fn(x_pred_m[0], y).item())
            
            residuals_a.append(torch.mean(abs(residual_op(dataset.scaler.inverse(x_pred_a))[0])).item())
            residuals_m.append(torch.mean(abs(residual_op(dataset.scaler.inverse(x_pred_m))[0])).item())
    
        print("Total mask time:", total_time)
    
        print("Mask method:")
        print(f"\tPixel-wise L2 error: {np.mean(losses_m):.4f} +/- {np.std(losses_m):.4f}")
        print(f"\tResidual L1 error: {np.mean(residuals_m):.4f} +/- {np.std(residuals_m):.4f}") 

        print("Article method:")
        print(f"\tPixel-wise L2 error: {np.mean(losses_a):.4f} +/- {np.std(losses_a):.4f}")
        print(f"\tResidual L1 error: {np.mean(residuals_a):.4f} +/- {np.std(residuals_a):.4f}")


def diffuse_mask_sparse_experiment(
        models, nsamples=10, model_epoch=200, perc=5, cuda=1, interp="nearest",
        art_steps=30, art_start=160, art_K=3,
        mask_steps=100, mask_start=1000, w_mask=1, 
        sigs=0.044
    ):
    
    with open(os.path.join('configs', "kmflow_re1000_rs256.yml"), 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    device = torch.device(f'cuda:{cuda}') if torch.cuda.is_available() else torch.device('cpu')
    seed = 1234
    fix_randomness(seed)
    print("Device", device)
    print("PID", os.getpid())
    print("N samples:", nsamples)
    print("Params:", art_steps, art_start, art_K, mask_steps, mask_start, sigs)
    config.device = device
    config.seed = seed

    dataset = KolmogorovFlowDataset(shu=True)
    test_dataset = dataset[-1000:] 

    print(f"Precomputing interpolated samples with {interp} method")
    # Generate sparse mask
    X_vals, sampled_ids = interpolate_dataset(test_dataset[0:nsamples], perc/100, method=interp)

    # interp=[linear, cubic, nearest]
    for sig in np.array([sigs]).flatten():
        # print("-" * 30)
        # print("# Testing mask_steps:", m_steps)
        # print("# Testing sig:", sig)
        # print("-" * 30)
        
        # print("Precomputing diffuse masks")
        # Precompute diffuse masks
        diffuse_masks = torch.zeros(len(sampled_ids), 3, 256, 256).to(device)
        for i in range(len(sampled_ids)):
            ids = list(sampled_ids[i]) + random.sample(range(256**2), int(256**2*w_mask))
            mask = diffuse_mask(ids, A=1, sig=sig)
            diffuse_masks[i] = torch.tensor(mask, dtype=torch.float).unsqueeze(0).repeat(3, 1, 1)

        for model_num in models:
            # print("-" * 30)
            # print("Testing model", model_num)
            model = Model(config)
            diffusion = Diffusion(config)

            cp = torch.load(f"runs/{str(model_num).zfill(3)}/checkpoint{model_epoch}.pt", map_location=device)
            model.load_state_dict(cp["model_state_dict"])
            model.to(device)
            model.eval();

            residual_op = ResidualOp(device)

            l2_norm = lambda x: torch.sqrt(torch.mean(x**2)).item()


            for m_steps in np.array([mask_steps]).flatten():
                losses_a = []
                losses_m = []
                losses_mm = []
                losses_i = []
                residuals_a = []
                residuals_m = []
                residuals_mm = []
                residuals_i = []
                lsim_a = []
                lsim_m = []
                lsim_mm = []
                lsim_i = []
                
                time_article = 0
                time_mask = 0

                for i in range(nsamples):
                    x     = X_vals[i].unsqueeze(0).to(device)
                    y     = test_dataset[i].unsqueeze(0).to(device)
                    noise = torch.randn(x.shape, device=x.device)

                    # --------------------------------------------------
                    # Old mask method
                    # _mask = torch.zeros(256,256).flatten()
                    # _mask[sampled_ids[i]] = 1
                    # _mask = _mask.reshape(256,256).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
                    # _mask[:, :, 0 , :] = 1
                    # _mask[:, :, -1, :] = 1
                    # _mask[:, :, : , 0] = 1
                    # _mask[:, :, :, -1] = 1
                    # x_pred_mm, _ = diffusion.ddim_mask(noise.clone(), model, x.clone(), 1000, 100, w_mask=0.2, _mask=_mask)
                    # --------------------------------------------------
                    # Measure time for x_pred_a
                    start_time = time.time()
                    x_pred_a = diffusion.ddim_article(x.clone(), model, art_start, art_steps, K=art_K)
                    end_time = time.time()
                    time_article += end_time - start_time

                    # Measure time for x_pred_m
                    start_time = time.time()
                    x_pred_m, _ = diffusion.ddim_mask(noise.clone(), model, x.clone(), mask_start, m_steps, diff_mask=diffuse_masks[i].unsqueeze(0))
                    end_time = time.time()
                    time_mask += end_time - start_time

                    losses_a.append(l2_loss_fn(x_pred_a[0], y).item())
                    losses_m.append(l2_loss_fn(x_pred_m[0], y).item())
                    # losses_mm.append(l2_loss_fn(x_pred_mm[0], y).item())
                    # losses_i.append(l2_loss_fn(x[0], y).item())
                    
                    residuals_a.append(l2_norm(residual_op(dataset.scaler.inverse(x_pred_a))[0]))
                    residuals_m.append(l2_norm(residual_op(dataset.scaler.inverse(x_pred_m))[0]))
                    # residuals_mm.append(l2_norm(residual_op(dataset.scaler.inverse(x_pred_mm))[0]))
                    # residuals_i.append(l2_norm(residual_op(dataset.scaler.inverse(x))[0]))
            
                    lsim_a.append(LSiM_distance(y, x_pred_a))
                    lsim_m.append(LSiM_distance(y, x_pred_m))
                    # lsim_mm.append(LSiM_distance(y, x_pred_mm))
                    # lsim_i.append(LSiM_distance(y, x))

                print(f"Time article: {time_article:.4f}")
                print(f"Time mask: {time_mask:.4f}")

                # print("Interpolation:")
                # print(f"\tPixel-wise L2 error: {np.mean(losses_i):.4f} +/- {np.std(losses_i):.4f}")
                # print(f"\tResidual L2 norm: {np.mean(residuals_i):.4f} +/- {np.std(residuals_i):.4f}") 
                # print(f"\tMean LSiM: {np.mean(lsim_i):.4f} +/- {np.std(lsim_i):.4f}")

                print("Article method:")
                print(f"\tPixel-wise L2 error: {np.mean(losses_a):.4f} +/- {np.std(losses_a):.4f}")
                print(f"\tResidual L2 norm: {np.mean(residuals_a):.4f} +/- {np.std(residuals_a):.4f}") 
                print(f"\tMean LSiM: {np.mean(lsim_a):.4f} +/- {np.std(lsim_a):.4f}")

                # print("Old Mask method:")
                # print(f"\tPixel-wise L2 error: {np.mean(losses_mm):.4f} +/- {np.std(losses_mm):.4f}")
                # print(f"\tResidual L2 norm: {np.mean(residuals_mm):.4f} +/- {np.std(residuals_mm):.4f}") 
                # print(f"\tMean LSiM: {np.mean(lsim_mm):.4f} +/- {np.std(lsim_mm):.4f}")
    
                print("Mask method:")
                print(f"\tPixel-wise L2 error: {np.mean(losses_m):.4f} +/- {np.std(losses_m):.4f}")
                print(f"\tResidual L2 norm: {np.mean(residuals_m):.4f} +/- {np.std(residuals_m):.4f}") 
                print(f"\tMean LSiM: {np.mean(lsim_m):.4f} +/- {np.std(lsim_m):.4f}")

                # print(f"{m_steps} {np.mean(losses_m):.4f} {np.mean(residuals_m):.4f} {np.mean(lsim_m):.4f}")

def save_plot(low_res, high_res, gt, filename="super_res_output.png"):
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

    # Save the plot as a PNG file
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved")

def test_wasserstein(models, nsamples=10, ntest=10, model_epoch=200, cuda=1):
    with open(os.path.join('configs', "kmflow_re1000_rs256.yml"), 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    device = torch.device(f'cuda:{cuda}') if torch.cuda.is_available() else torch.device('cpu')
    seed = 1234
    
    print("-- Wasserstein TEST --")
    print("Device", device)
    print("PID", os.getpid())
    print("Testing models:", models)
    print("Epochs:", model_epoch)

    config.device = device
    config.seed = seed
    residual_op = ResidualOp(device)
    
    print("Loading dataset")
    dataset = KolmogorovFlowDataset()
    dataset_test = dataset.scaler.inverse(dataset[-ntest:]).to(device)
    for model_num in models:
        print("-" * 30)
        print("Testing model", model_num)
        model = Model(config)
        diffusion = Diffusion(config)

        cp = torch.load(f"runs/{str(model_num).zfill(3)}/checkpoint{model_epoch}.pt", map_location=device)
        model.load_state_dict(cp["model_state_dict"])
        model.to(device)
        model.eval();

        sample_size = 256
        gen_samples = torch.zeros(nsamples, 3, sample_size, sample_size, device=device)
        res_samples = torch.zeros(nsamples)
        for i in range(nsamples):
            if i % 10 == 0: print("Generating sample:", i)

            noise          = torch.randn((1, 3, sample_size, sample_size), device=device)
            gen_samples[i] = diffusion.ddpm(noise, model, 1000)        
            res, _         = residual_op(dataset.scaler.inverse(gen_samples[i]).unsqueeze(0))
            res_samples[i] = torch.sqrt(torch.mean(res**2))

        print("Computing metric")
        metric = wasserstein(dataset_test, dataset.scaler.inverse(gen_samples))
        print("Wasserstein distance:", metric)
        print(f"L2 norm Residuals: {torch.mean(res_samples).item():.4f} +/- {torch.std(res_samples).item():.4f}")


def gen_samples(models, n_samples, cuda=0, model_epoch=1000):
    with open(os.path.join('configs', "kmflow_re1000_rs256.yml"), 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    device = torch.device(f'cuda:{cuda}') if torch.cuda.is_available() else torch.device('cpu')
    
    print("-- Generating samples --")
    print("Device", device)
    print("PID", os.getpid())

    config.device = device
    diffusion = Diffusion(config)

    for model_num in models:
        print("-"*30)
        print(f"Generating {n_samples} samples for model {model_num}:")    

        model = Model(config)
        diffusion = Diffusion(config)

        cp = torch.load(f"runs/{str(model_num).zfill(3)}/checkpoint{model_epoch}.pt", map_location=device)
        model.load_state_dict(cp["model_state_dict"])
        model.to(device)
        model.eval()

        samples = torch.zeros((n_samples, 3, 256, 256))
        for i in range(n_samples):
            if i % 10: print("Generating sample", i)

            noise = torch.randn((1, 3, 256, 256), device=device)
            samples[i] = diffusion.ddpm(noise, model, 1000)
        
        filename = f"samples_{str(model_num).zfill(3)}_n{n_samples}.pt"
        print("Storing samples at", filename)        
        torch.save(samples, filename)


if __name__ == "__main__":

    start_time = datetime.now()

    # residual_of_generated([173], model_epoch=200, cuda=1)
    # residual_of_generated([181], model_epoch=1000, cuda=0)
    # test_MSE([108, 111, 114], model_epoch=400, nsamples=1000, cuda=1)
    # test_wasserstein([169, 180, 181,167,182,171,176,183,000, 172,175,179], model_epoch=1000, nsamples=1000, ntest=1000, cuda=1)

    # mask_experiment(
    #     # [119,109,110,112,113,115,116,118], 
    #     # [89, 101, 121],
    #     # [97,108,111,114,117,120,122,123],
    #     [121],
    #     nsamples=10, model_epoch=1000, cuda=0,
        
    #     ## Task 4:
    #     scale_factor=4, 
    #     art_steps=36, art_start=160, art_K=3,
    #     mask_steps=100, mask_start=1000, w_mask=1,

    #     ## Task 8:
    #     # scale_factor=8, 
    #     # art_steps=36, art_start=320, art_K=1,
    #     # mask_steps=80, mask_start=800, w_mask=0.5, 
    # )


    mask_sparse_experiment(
        # [119,109,110,112,113,115,116,118], 
        [54],
        nsamples=10, model_epoch=1000,
        
        perc = 1.5625,
        #perc=5,
        art_steps=36, art_start=400, art_K=1,
        #art_steps=36, art_start=160, art_K=3,
        mask_steps=100, mask_start=1000, w_mask=0.15,
    )


    # Task 5%:    art_steps=36, art_start=160, art_K=3,
    # Task 1.56%: art_steps=36, art_start=400, art_K=1,
    
    #diffuse_mask_experiment(
    #    [15], 
    #    nsamples=1, model_epoch=550, scale_factor=4, cuda=0,
    #    art_steps=30, art_start=160, art_K=3,
    #    mask_steps=100, mask_start=1000, w_mask=1,
    #    sig=0.008, dist=10
    #)

    #diffuse_mask_sparse_experiment(
    #    # [169, 167, 171, 172, 175],
    #    [54],
    #    nsamples=10, model_epoch=1000, cuda=0,
    #    perc=5,
    #    #perc=1.5625,
    #    art_steps=36, art_start=160, art_K=3,
    #    #art_steps=36, art_start=400, art_K=1,
    #    mask_steps=[100], mask_start=1000, w_mask=0,
    #    sigs=[0.038], interp="nearest"
    #   # sigs=[0.052], interp="nearest"
    #)



    execution_time = datetime.now() - start_time
    print(f"Execution time: {execution_time.total_seconds()} seconds")
