import numpy as np
import pywt
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit

def reshape_data_wavelet(data, nsamples):
    batch_size, time, Nx, Ny = data.shape
    dataset = data[:, 1, :, :]
    dataset = dataset.reshape(batch_size, Nx, Ny)  
    dataset = dataset[:nsamples]  
    print(f"Data matrix shape: {dataset.shape}")  
    return dataset

def wavelet_decompose(data, wavelet='db4', level=1):
    coeffs = pywt.wavedec2(data, wavelet, level=level)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    return coeff_arr, coeff_slices

def learn_wavelet_dictionary(full_dataset, n_components):
    """
    Train a dictionary from wavelet-transformed CFD snapshots.
    """
    wavelet_coeffs = []
    for snapshot in full_dataset:
        coeff_arr, _ = wavelet_decompose(snapshot)
        wavelet_coeffs.append(coeff_arr.flatten())

    wavelet_coeffs = np.array(wavelet_coeffs)

    dict_learner = DictionaryLearning(n_components=n_components, transform_algorithm='lasso_lars')
    dict_learner.fit(wavelet_coeffs)

    return dict_learner

def sample_sparse_field(field, perc=5):
    """
    Sample only a fraction of the field.
    """
    mask = np.random.rand(*field.shape) < (perc / 100)
    sparse_field = field * mask
    return sparse_field, mask

def wavelet_reconstruct(sparse_field, mask, dict_model, wavelet='db4', level=1):
    """
    Reconstruct a field from sparse samples using the wavelet dictionary.
    """
    # Decompose the sparse field into wavelet coefficients WITHOUT applying the mask first
    coeffs = pywt.wavedec2(sparse_field, wavelet, level=level)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    # Ensure the mask is transformed the same way
    mask_coeffs = pywt.wavedec2(mask.astype(float), wavelet, level=level)
    mask_arr, _ = pywt.coeffs_to_array(mask_coeffs)

    # Identify missing coefficients based on the mask
    missing_coeffs_mask = mask_arr == 0

    # Use sparse coding to reconstruct the missing wavelet coefficients
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=10)
    omp.fit(dict_model.components_.T, coeff_arr.flatten())
    reconstructed_coeffs = omp.predict(dict_model.components_.T).reshape(coeff_arr.shape)

    # Replace only the missing coefficients with the reconstructed values
    reconstructed_coeffs[missing_coeffs_mask] = coeff_arr[missing_coeffs_mask]

    # Convert back to spatial domain
    reconstructed_coeffs = pywt.array_to_coeffs(reconstructed_coeffs, coeff_slices, output_format='wavedec2')
    reconstructed_field = pywt.waverec2(reconstructed_coeffs, wavelet)

    return reconstructed_field
