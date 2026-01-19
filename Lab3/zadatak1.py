import numpy as np
import matplotlib.pyplot as plt

def function(tt):
    f = 8000
    t = np.arange(tt)
    return np.cos(np.pi * 2 * 800 * t / f) + np.cos(np.pi * 2 * 888.8 * t / f) + np.cos(np.pi * 2 * 1600 * t / f)

funcs = [function, np.blackman, np.hamming]

for f in funcs:
    fig, [(graph1, graph2), (graph3, graph4), (graph5, graph6), (graph7, graph8), (graph9, grapha)] = plt.subplots(nrows = 5, ncols = 2)

    graph1.stem(np.arange(64),f(64))
    graph2.plot(np.arange(64),np.abs(np.fft.fft(f(64),64)))
    graph3.plot(np.arange(1024),np.abs(np.fft.fft(f(64),1024)))
    graph4.plot(np.arange(2048),np.abs(np.fft.fft(f(64),2048)))
    graph5.plot(np.arange(256),np.abs(np.fft.fft(f(256))))
    graph6.plot(np.arange(512),np.abs(np.fft.fft(f(512))))
    graph7.plot(np.arange(1024),np.abs(np.fft.fft(f(256),1024)))
    graph8.plot(np.arange(2048),np.abs(np.fft.fft(f(256),2048)))
    graph9.plot(np.arange(1024),np.abs(np.fft.fft(f(512),1024)))
    grapha.plot(np.arange(2048),np.abs(np.fft.fft(f(512),2048)))

    plt.show()

    """
    import numpy as np

def stft_from_scratch(x, window, hop_length, nfft=None):

    Compute the Short-Time Fourier Transform (STFT) of a signal from scratch.

    Args:
        x (np.ndarray): Input signal (1D array).
        window (np.ndarray): Window function (1D array).
        hop_length (int): Number of samples between successive frames.
        nfft (int, optional): Length of the FFT. If None, uses window length.

    Returns:
        np.ndarray: The STFT matrix (rows are frequencies, columns are time frames).
 
    N = len(window)
    # Use window length for FFT if nfft is not specified
    if nfft is None:
        nfft = N
    
    # Calculate the number of frames
    L = len(x)
    M = int(np.floor((L - N) / hop_length)) + 1
    
    # Initialize the STFT matrix with complex zeros
    X = np.zeros((nfft // 2 + 1, M), dtype=complex) # For real input, we only need positive frequencies

    for m in range(M):
        # Extract the m-th segment of the signal
        start_index = m * hop_length
        end_index = start_index + N
        x_segment = x[start_index:end_index]
        
        # Apply the window function
        x_windowed = x_segment * window
        
        # Compute the FFT of the windowed segment
        # Using rfft for real input, which returns the one-sided spectrum
        X_win = np.fft.rfft(x_windowed, n=nfft) 
        
        # Store the result in the STFT matrix
        X[:, m] = X_win
        
    return X

# --- Example Usage ---

# 1. Generate a test signal (e.g., two sine waves with different frequencies)
fs = 1000  # Sampling frequency in Hz
t = np.arange(0, 5, 1/fs) # 5 seconds duration
# Signal changes frequency after 2.5 seconds
x = np.sin(2 * np.pi * 50 * t) # 50 Hz sine wave for the first part
x[int(len(t)/2):] += np.sin(2 * np.pi * 120 * t[int(len(t)/2):]) # Add 120 Hz sine wave later

# 2. Define STFT parameters
N_window = 512       # Window length
H_hop_length = 128   # Hop length (e.g., 25% of window length for 75% overlap)
n_fft = 512          # FFT length (can be longer than window for zero padding)

# 3. Define a window function (Hann window is common)
# Scipy provides convenient window functions
from scipy.signal import hann
window = hann(N_window)

# 4. Compute the STFT
spectrogram_data = stft_from_scratch(x, window, H_hop_length, nfft=n_fft)

# 5. Optional: Plot the spectrogram (visualization is key for STFT)
import matplotlib.pyplot as plt

# Get frequency and time axes for plotting
# Note: Frequencies returned by rfft go up to fs/2
frequencies = np.fft.rfftfreq(n_fft, 1/fs) 
times = np.arange(spectrogram_data.shape[1]) * H_hop_length / fs

plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies, np.abs(spectrogram_data), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [Sec]')
plt.title('Spectrogram (Magnitude of STFT)')
plt.colorbar(label='Magnitude')
plt.show()

    """