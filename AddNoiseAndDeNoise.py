import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
import pywt

# 1. Create a sample signal (e.g., a simple sine wave)
def create_signal(length):
    return np.sin(np.linspace(0, 4 * np.pi, length))

# 2. Add Gaussian Noise
def add_gaussian_noise(signal, noise_level=0.5):
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise

# 3. Denoising (a simple averaging filter)
def denoise_signal(noisy_signal, window_size=5):
    denoised_signal = np.convolve(noisy_signal, np.ones(window_size)/window_size, mode='same')
    return denoised_signal

# 4. Median Filtering
def denoise_signal_median(noisy_signal, kernel_size=3):
    """
    Denoise the signal using a median filter.
    :param noisy_signal: The noisy input signal.
    :param kernel_size: Size of the median filter kernel (must be odd).
    :return: Denoised signal.
    """
    return medfilt(noisy_signal, kernel_size)

# 5. Gaussian Filtering
def denoise_signal_gaussian(noisy_signal, sigma=1.0):
    """
    Denoise the signal using a Gaussian filter.
    :param noisy_signal: The noisy input signal.
    :param sigma: Standard deviation for Gaussian kernel.
    :return: Denoised signal.
    """
    return gaussian_filter(noisy_signal, sigma=sigma)

# 6. Wavelet Denoising

def denoise_signal_wavelet(noisy_signal, wavelet='db1', level=1):
    
    #Denoise the signal using wavelet thresholding.
    #:param noisy_signal: The noisy input signal.
    #:param wavelet: Type of wavelet to use (e.g., 'db1', 'haar').
    #:param level: Decomposition level.
    #:return: Denoised signal.
    
    coeffs = pywt.wavedec(noisy_signal, wavelet, level=level)
    threshold = np.sqrt(2 * np.log(len(noisy_signal))) * np.median(np.abs(coeffs[-1])) / 0.6745
    denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(denoised_coeffs, wavelet)


# --- Main ---
signal_length = 100
original_signal = create_signal(signal_length)
noisy_signal = add_gaussian_noise(original_signal, noise_level=0.8) # Increased noise level
denoised_signal = denoise_signal(noisy_signal, window_size=7) # Increased window size

# Example usage of the new denoising methods
denoised_signal_median = denoise_signal_median(noisy_signal, kernel_size=5)
denoised_signal_gaussian = denoise_signal_gaussian(noisy_signal, sigma=1.0)
denoised_signal_wavelet = denoise_signal_wavelet(noisy_signal, wavelet='db6', level=3)

# --- Plotting ---
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(original_signal, label='Original Signal')
plt.plot(noisy_signal, label='Noisy Signal', alpha=0.7)
plt.title('Original and Noisy Signal')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(original_signal, label='Original Signal')
plt.plot(denoised_signal, label='Averaging Filter', alpha=0.7)
plt.title('Denoised Signal (Averaging Filter)')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(original_signal, label='Original Signal')
plt.plot(denoised_signal_median, label='Median Filter', alpha=0.7)
plt.title('Denoised Signal (Median Filter)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(original_signal, label='Original Signal')
plt.plot(denoised_signal_gaussian, label='Gaussian Filter', alpha=0.7)
plt.plot(denoised_signal_wavelet, label='Wavelet Denoising', alpha=0.7)
plt.title('Denoised Signal (Gaussian Filter)')
plt.legend()

plt.tight_layout()
plt.show()