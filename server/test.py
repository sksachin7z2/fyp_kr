# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import wavfile

# # Load an audio file (replace 'your_audio_file.wav' with the actual file)
# fs, audio_signal = wavfile.read('')

# # Perform FFT to obtain frequency spectrum
# frequencies = np.fft.fftfreq(len(audio_signal), 1/fs)
# magnitude_spectrum = np.abs(np.fft.fft(audio_signal))

# # Plot the frequency spectrum
# plt.plot(frequencies, magnitude_spectrum)
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.title('Frequency Spectrum of Audio Signal')
# plt.show()

# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import numpy as np

# # Load audio file
# audio_file_path = "1001_AH01MENC_5.wav"
# y, sr = librosa.load(audio_file_path)

# # Compute STFT
# D = librosa.feature.melspectrogram(y,n_fft=)

# # Display the STFT
# librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
# plt.title('STFT of Audio Signal')
# plt.colorbar(format='%+2.0f dB')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters for the sine wave
# amplitude = 2.0        # Amplitude of the sine wave
# frequency = 5.0        # Frequency of the sine wave in Hertz
# duration = 1.0         # Duration of the signal in seconds
# sampling_rate = 100.0  # Sampling rate in Hertz

# # Generate a sine wave
# t = np.arange(0, duration, 1/sampling_rate)
# sinewave = amplitude * np.sin(2 * np.pi * frequency * t)

# # Plot the generated sine wave
# plt.figure(figsize=(12, 4))
# plt.subplot(2, 1, 1)
# plt.plot(t, sinewave, label='Generated Sine Wave')
# plt.title('Generated Sine Wave')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.legend()

# # Calculate the DFT
# dft_result = np.fft.fft(sinewave)
# frequencies = np.fft.fftfreq(len(dft_result), d=1/sampling_rate)

# # Plot the magnitude spectrum (DFT)
# plt.subplot(2, 1, 2)
# plt.plot(frequencies, np.abs(dft_result), label='DFT Magnitude Spectrum')
# plt.title('DFT Magnitude Spectrum')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.legend()

# plt.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters for the signal
# duration = 5.0         # Duration of the signal in seconds
# sampling_rate = 100.0   # Sampling rate in Hertz

# # Generate the signal e^(-t) * sin(t)
# t = np.arange(0, duration, 1/sampling_rate)
# signal = np.exp(-t) * np.sin(t)

# # Plot the generated signal
# plt.figure(figsize=(12, 4))
# plt.subplot(2, 1, 1)
# plt.plot(t, signal, label=r'$e^{-t} \sin(t)$')
# plt.title('Generated Signal: $e^{-t} \sin(t)$')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.legend()

# # Calculate the DFT
# dft_result = np.fft.fft(signal)
# frequencies = np.fft.fftfreq(len(dft_result), d=1/sampling_rate)

# # Plot the magnitude spectrum (DFT)
# plt.subplot(2, 1, 2)
# plt.plot(frequencies, np.abs(dft_result), label='DFT Magnitude Spectrum')
# plt.title('DFT Magnitude Spectrum')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.legend()

# plt.tight_layout()
# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt

# def relu(x):
#     return np.maximum(0, x)

# # Generate a range of values
# x_values = np.linspace(-5, 5, 100)

# # Apply ReLU to the values
# y_values = relu(x_values)

# # Plot the ReLU function
# plt.plot(x_values, y_values, label='ReLU')
# plt.xlabel('Input')
# plt.ylabel('Output')
# plt.title('ReLU Activation Function')
# plt.legend()
# plt.grid(True)
# plt.show()

# import numpy as np
# from scipy.fftpack import fftfreq, fft, ifft

# def mel_filter_bank(num_filters, fft_size, sample_rate, low_freq, high_freq):
#     """
#     Generate a Mel filter bank.

#     Parameters:
#     - num_filters: Number of Mel filters.
#     - fft_size: Size of the FFT.
#     - sample_rate: Sampling rate of the audio.
#     - low_freq: Lower limit of the Mel filter bank.
#     - high_freq: Upper limit of the Mel filter bank.

#     Returns:
#     - filter_bank: Mel filter bank matrix.
#     """

#     # Convert frequency limits to Mel scale
#     mel_low = 1127 * np.log(1 + low_freq / 700)
#     mel_high = 1127 * np.log(1 + high_freq / 700)

#     # Generate equally spaced points in Mel scale
#     mel_points = np.linspace(mel_low, mel_high, num_filters + 2)

#     # Convert Mel points back to Hz scale
#     hz_points = 700 * (np.exp(mel_points / 1127) - 1)

#     # Convert Hz points to FFT bin indices
#     bin_indices = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)

#     filter_bank = np.zeros((num_filters, int(fft_size / 2 + 1)))

#     for i in range(1, num_filters + 1):
#         # Triangular filter shape
#         filter_bank[i - 1, bin_indices[i - 1]:bin_indices[i]] = (bin_indices[i] - np.arange(bin_indices[i - 1], bin_indices[i])) / (bin_indices[i] - bin_indices[i - 1])
#         filter_bank[i - 1, bin_indices[i]:bin_indices[i + 1]] = (np.arange(bin_indices[i] + 1, bin_indices[i + 1] + 1) - bin_indices[i]) / (bin_indices[i + 1] - bin_indices[i])

#     return filter_bank

# # Example usage:
# num_filters = 26
# fft_size = 512
# sample_rate = 16000
# low_freq = 0
# high_freq = 8000

# mel_filterbank = mel_filter_bank(num_filters, fft_size, sample_rate, low_freq, high_freq)

# # Display the Mel filter bank
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 4))
# plt.imshow(mel_filterbank, cmap='viridis', aspect='auto', origin='lower')
# plt.title('Mel Filter Bank')
# plt.xlabel('FFT Bins')
# plt.ylabel('Mel Filters')
# plt.colorbar(format='%+2.0f dB')
# plt.show()

# import numpy as np

# # Generate a random 3x3 matrix
# random_matrix = np.random.rand(3, 3)

# print("Random Matrix:")
# print(random_matrix)

# # Calculate the Frobenius norm (magnitude) of the matrix
# matrix_magnitude = np.linalg.norm(random_matrix, 'fro')

# print("\nMatrix Magnitude (Frobenius Norm):", matrix_magnitude)

# import numpy as np
# import matplotlib.pyplot as plt

# def hann_window(N):
#     n = np.arange(N)
#     return 0.54 - 0.54 * np.cos(2 * np.pi * n / (N - 1))

# # Values of N to visualize
# N_values = [16, 32, 64, 128]

# # Plotting the Hann window for different N values
# plt.figure(figsize=(10, 6))
# for N in N_values:
#     window = hann_window(N)
#     plt.plot(window, label=f'N = {N}')

# plt.title('Hann Window for Different N Values')
# plt.xlabel('Sample Index')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.grid(True)
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# def hamming_window(N, alpha):
#     n = np.arange(N)
#     return alpha - (1 - alpha) * np.cos(2 * np.pi * n / (N - 1))

# def hann_window(N):
#     n = np.arange(N)
#     return 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))

# # Values of alpha to visualize the transition from Hamming to Hann
# alpha_values = np.linspace(0, 1, 100)

# # Plotting the transition from Hamming to Hann window
# plt.figure(figsize=(10, 6))
# for alpha in alpha_values:
#     window = hamming_window(N=64, alpha=alpha)
#     plt.plot(window, label=f'Alpha = {alpha:.2f}')

# # Plot the Hann window for comparison
# plt.plot(hann_window(N=64), label='Hann Window', linestyle='--', linewidth=2)

# plt.title('Transition from Hamming to Hann Window')
# plt.xlabel('Sample Index')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.grid(True)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

def hamming_window(N, alpha=0.54):
    n = np.arange(N)
    return alpha - (1 - alpha) * np.cos(2 * np.pi * n / (N - 1))

def hann_window(N):
    n = np.arange(N)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))

# Values of N to visualize
N_values = [16, 32, 64, 128]

# Plotting the Hamming and Hann windows for different N values
plt.figure(figsize=(10, 6))
for N in N_values:
    hamming = hamming_window(N)
    hann = hann_window(N)
    plt.plot(hamming, label=f'Hamming, N = {N}')
    plt.plot(hann, label=f'Hann, N = {N}', linestyle='dashed')

plt.title('Comparison of Hamming and Hann Windows for Different N Values')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
