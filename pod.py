import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import torch

def reshape_data(data, nsamples):
    batch_size, time, Nx, Ny = data.shape
    dataset = data[:, 1, :, :]
    dataset = dataset.reshape(batch_size, Nx * Ny)  
    dataset = dataset[:nsamples]  
    print(f"Data matrix shape for SVD: {dataset.shape}")  
    return dataset

# Perform POD using Singular Value Decomposition (SVD)
def compute_pod(data_matrix):
    U, S, Vt = np.linalg.svd(data_matrix, full_matrices=False)
    return U, S, Vt

# Reconstruct a snapshot using a given number of modes
def reconstruct_snapshot(U, S, Vt, num_modes, snapshot_idx, Nx, Ny):
    # Reconstruct the entire snapshot matrix using the first 'num_modes' modes
    snapshot_flat = U[:, :num_modes] @ np.diag(S[:num_modes]) @ Vt[:num_modes, :]
    
    # Extract the specific snapshot for snapshot_idx and reshape it
    snapshot = snapshot_flat[:, snapshot_idx].reshape(Nx, Ny) 
    return snapshot

def plot_singular_values(S):
    plt.figure(figsize=(8, 5))
    plt.semilogy(S, 'bo-', label="Singular values")
    plt.xlabel("Mode index")
    plt.ylabel("Singular Value (log scale)")
    plt.title("POD Singular Values")
    plt.legend()
    plt.grid()
    plt.show()

def plot_snapshots(original, reconstructed, title1="Original Snapshot", title2="Reconstructed Snapshot"):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap="viridis")
    plt.title(title1)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed, cmap="viridis")
    plt.title(title2)
    plt.colorbar()

    plt.show()

# Function to create the selection matrix C
def create_selection_matrix(Nx, Ny, percentage=0.2):
    total_points = Nx * Ny
    selected_points = int(percentage * total_points)
    selected_indices = np.random.choice(total_points, selected_points, replace=False)
    
    C = np.zeros((selected_points, total_points))
    C[np.arange(selected_points), selected_indices] = 1
    
    return C, selected_indices

def direct_interpolation(snapshot, indices, Nx, Ny):
    x_selected, y_selected = np.unravel_index(indices, (Nx, Ny))
    sparse_values = snapshot.flatten()[indices]
    x_full, y_full = np.meshgrid(np.arange(Nx), np.arange(Ny))
    interpolated_snapshot = griddata((x_selected, y_selected), sparse_values, (x_full, y_full), method='nearest')

    return interpolated_snapshot

def perform_superresolution(U, num_modes, C, snapshot, Nx, Ny):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    Ur = U[:, :num_modes]
    A = torch.tensor(C @ Ur, device=device, dtype=torch.float32)
    b = torch.tensor(C @ np.array(snapshot.flatten()), device=device, dtype=torch.float32)
    
    # Solve least squares using torch
    x = torch.linalg.lstsq(A, b).solution
    
    # Compute reconstructed snapshot
    reconstructed_snapshot = (Ur @ x.cpu().numpy()).reshape(Nx, Ny)

    return reconstructed_snapshot

def plot_sparse_reconstruction(snapshot, interpolated_snapshot, reconstructed_snapshot):
    # Plot original snapshot, interpolated snapshot, and reconstructed snapshot
    plt.figure(figsize=(15, 5))

    # Plot the original snapshot
    plt.subplot(1, 3, 1)
    plt.imshow(snapshot, cmap="inferno")
    plt.title("Original Snapshot")

    # Plot the interpolated snapshot
    plt.subplot(1, 3, 2)
    plt.imshow(interpolated_snapshot.T, cmap="inferno")
    plt.title("Interpolated Snapshot")

    # Plot the reconstructed snapshot
    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed_snapshot, cmap="inferno")
    plt.title("Reconstructed Snapshot")

    # Show the plots
    plt.tight_layout()
    plt.show()

def reconstruct_all_snapshots_gappy(U, num_modes, C, test_snapshots, Nx, Ny):
    reconstructed_snapshots = []
    index = 1
    for snapshot in test_snapshots:
        print(f"Reconstructing snapshot {index}")
        index += 1
        reconstructed_snapshot = perform_superresolution(U, num_modes, C, snapshot, Nx, Ny)
        reconstructed_snapshots.append(reconstructed_snapshot)
    
    return np.array(reconstructed_snapshots)

def compute_fft_error(original, reconstructed):
    # Calculate the error
    error = original - reconstructed
    
    # Compute the 2D FFT of the error
    fft_error = np.fft.fft2(error)
    
    # Compute the magnitude spectrum
    magnitude_spectrum = np.abs(fft_error)
    
    # Compute radial frequency bins
    Nx, Ny = error.shape
    kx = np.fft.fftfreq(Nx).reshape(-1, 1)  # Frequency values along x
    ky = np.fft.fftfreq(Ny).reshape(1, -1)  # Frequency values along y
    k = np.sqrt(kx**2 + ky**2)  # Compute the radial frequency

    # Flatten arrays for binning
    k_flat = k.flatten()
    mag_flat = magnitude_spectrum.flatten()

    # Sort by frequency
    sorted_indices = np.argsort(k_flat)
    k_sorted = k_flat[sorted_indices]
    mag_sorted = mag_flat[sorted_indices]

    # Compute mean magnitude in frequency bins
    num_bins = 100  # Number of frequency bins
    k_bins = np.linspace(k_sorted.min(), k_sorted.max(), num_bins)
    k_means = 0.5 * (k_bins[:-1] + k_bins[1:])
    mag_means = np.histogram(k_sorted, bins=k_bins, weights=mag_sorted)[0] / np.histogram(k_sorted, bins=k_bins)[0]

    # Plot the error magnitude vs. frequency
    plt.figure(figsize=(8, 5))
    plt.plot(k_means, mag_means, 'b-', label="FFT Error Magnitude")
    plt.xlabel("Frequency")
    plt.ylabel("Error Magnitude")
    plt.title("Error Spectrum in Frequency Domain")
    plt.legend()
    plt.grid()
    plt.show()
