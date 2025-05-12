# This script explores and visualizes the LFP data from the NWB file

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/e392479c-8683-4424-a75b-34af512a17a2/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get LFP data
lfp = nwb.acquisition["LFP"]
print(f"LFP data shape: {lfp.data.shape}")
print(f"Sampling rate: {lfp.rate} Hz")

# Get a subset of the LFP data (first 5 seconds, all channels)
# 5 seconds at 1000 Hz = 5000 samples
time_slice = slice(0, 5000)
lfp_subset = lfp.data[time_slice, :]

# Create time vector for the subset (in seconds)
time_vector = np.arange(lfp_subset.shape[0]) / lfp.rate

# Plot LFP traces for the first 4 channels
plt.figure(figsize=(12, 8))
channels_to_plot = min(4, lfp_subset.shape[1])
for i in range(channels_to_plot):
    plt.subplot(channels_to_plot, 1, i+1)
    plt.plot(time_vector, lfp_subset[:, i])
    plt.title(f'LFP Channel {i}')
    plt.ylabel('Voltage (V)')
    if i == channels_to_plot-1:  # Only add xlabel to bottom plot
        plt.xlabel('Time (s)')
plt.tight_layout()
plt.savefig('explore/lfp_timeseries.png')
plt.close()

# Create a spectrogram for one of the LFP channels (e.g., channel 0)
channel = 0
plt.figure(figsize=(10, 6))

# Using scipy.signal.spectrogram instead of plt.mlab.specgram
# Compute spectrogram
f, t, Sxx = signal.spectrogram(lfp_subset[:, channel], fs=lfp.rate, 
                               nperseg=512, noverlap=256,
                               scaling='density')

# Plot spectrogram
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title(f'Spectrogram Channel {channel}')
plt.ylim(0, 100)  # Limit frequency display to 0-100 Hz
plt.tight_layout()
plt.savefig('explore/lfp_spectrogram.png')
plt.close()

# Calculate and plot power spectrum (averaged over time) for multiple channels
plt.figure(figsize=(10, 6))
for i in range(min(4, lfp_subset.shape[1])):
    # Calculate the power spectrum using scipy.signal.welch
    f, Pxx = signal.welch(lfp_subset[:, i], fs=lfp.rate, nperseg=1024)
    # Convert to dB scale
    Pxx_db = 10 * np.log10(Pxx)
    plt.plot(f, Pxx_db, label=f'Channel {i}')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.title('LFP Power Spectral Density')
plt.xlim(0, 100)  # Limit frequency display to 0-100 Hz
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/lfp_power_spectrum.png')
plt.close()

print("Visualizations completed and saved to explore directory.")