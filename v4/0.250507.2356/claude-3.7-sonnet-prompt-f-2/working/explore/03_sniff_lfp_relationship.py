# This script visualizes the sniffing data and explores its relationship with LFP activity

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/e392479c-8683-4424-a75b-34af512a17a2/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get data
lfp = nwb.acquisition["LFP"]
sniff = nwb.acquisition["SniffSignal"]
behavior = nwb.processing["behavior"]

# Print basic info
print("Basic Information:")
print(f"LFP data shape: {lfp.data.shape}")
print(f"Sniff signal shape: {sniff.data.shape}")
print(f"Sampling rate: {lfp.rate} Hz")

# Get a subset of the data (first 10 seconds)
# 10 seconds at 1000 Hz = 10000 samples
time_slice = slice(0, 10000)
lfp_subset = lfp.data[time_slice, 0]  # First channel only
sniff_subset = sniff.data[time_slice]
time_vector = np.arange(len(sniff_subset)) / sniff.rate

# Plot the sniff signal
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.plot(time_vector, sniff_subset)
plt.title('Raw Sniff Signal')
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
plt.grid(True)

# Plot the LFP signal (one channel for comparison)
plt.subplot(3, 1, 2)
plt.plot(time_vector, lfp_subset)
plt.title('LFP Signal (Channel 0)')
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
plt.grid(True)

# Plot the inhalation and exhalation timestamps
inhalation = behavior.data_interfaces["inhalation_time"]
exhalation = behavior.data_interfaces["exhalation_time"]

# Get inhalation and exhalation times within our time window
inh_mask = (inhalation.timestamps[:] >= 0) & (inhalation.timestamps[:] <= 10)
exh_mask = (exhalation.timestamps[:] >= 0) & (exhalation.timestamps[:] <= 10)

inh_times = inhalation.timestamps[inh_mask]
exh_times = exhalation.timestamps[exh_mask]
inh_data = inhalation.data[inh_mask]
exh_data = exhalation.data[exh_mask]

# Plot timing of inhalation and exhalation events
plt.subplot(3, 1, 3)
if len(inh_times) > 0:
    plt.stem(inh_times, np.ones_like(inh_times), 'b', markerfmt='bo', label='Inhalation')
if len(exh_times) > 0:
    plt.stem(exh_times, 0.8*np.ones_like(exh_times), 'r', markerfmt='ro', label='Exhalation')
plt.title('Sniffing Events (Inhalation and Exhalation)')
plt.ylabel('Event')
plt.xlabel('Time (s)')
plt.ylim(0, 1.2)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('explore/sniff_lfp_comparison.png')
plt.close()

# Calculate and plot respiratory rhythm (time between consecutive inhalations)
if len(inh_times) > 1:
    inh_intervals = np.diff(inh_times)
    plt.figure(figsize=(10, 5))
    plt.plot(inh_times[:-1], inh_intervals)
    plt.title('Respiratory Rhythm - Interval Between Inhalations')
    plt.ylabel('Interval (seconds)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('explore/respiratory_rhythm.png')
    plt.close()

# Analyze power in specific frequency bands of LFP around sniff events
# Get a longer segment of LFP data to analyze
lfp_data = lfp.data[0:50000, 0]  # 50 seconds of data, first channel

# Get timestamps within our 50 second window
inh_mask_long = (inhalation.timestamps[:] >= 0) & (inhalation.timestamps[:] <= 50)
inh_times_long = inhalation.timestamps[inh_mask_long]

if len(inh_times_long) > 0:
    # Define frequency bands
    bands = {
        'theta': (4, 12),
        'beta': (15, 30),
        'low_gamma': (30, 50),
        'high_gamma': (50, 100)
    }
    
    # Window around each inhalation (1 second before, 2 seconds after)
    pre_window = int(1 * lfp.rate)  # 1 second before
    post_window = int(2 * lfp.rate)  # 2 seconds after
    
    # Initialize arrays to store band power for each inhalation event
    band_powers = {band: [] for band in bands.keys()}
    
    for time in inh_times_long:
        # Convert time to sample index
        center_idx = int(time * lfp.rate)
        
        # Ensure the window is within data bounds
        if center_idx-pre_window >= 0 and center_idx+post_window < len(lfp_data):
            # Extract window of data
            window = lfp_data[center_idx-pre_window:center_idx+post_window]
            
            # Calculate power in each frequency band
            f, Pxx = signal.welch(window, fs=lfp.rate, nperseg=1024)
            
            for band, (low, high) in bands.items():
                # Find indices of frequencies in band
                idx = np.logical_and(f >= low, f <= high)
                # Average power in band
                if np.any(idx):  # Ensure we have frequencies in this band
                    band_powers[band].append(np.mean(Pxx[idx]))
    
    # Create a bar plot of average power in each frequency band
    if all(len(powers) > 0 for powers in band_powers.values()):
        plt.figure(figsize=(10, 6))
        # Calculate means
        means = [np.mean(band_powers[band]) for band in bands.keys()]
        # Calculate standard errors
        sems = [np.std(band_powers[band]) / np.sqrt(len(band_powers[band])) for band in bands.keys()]
        
        # Create bar plot
        plt.bar(bands.keys(), means, yerr=sems, capsize=10)
        plt.title('Average LFP Power in Frequency Bands Around Inhalation Events')
        plt.ylabel('Power (V²/Hz)')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig('explore/lfp_band_power_inhalation.png')
        plt.close()

print("Analysis completed and visualizations saved to explore directory.")