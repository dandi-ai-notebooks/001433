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

# Get a subset of the data (first 20 seconds)
# 20 seconds at 1000 Hz = 20000 samples
time_slice = slice(0, 20000)
lfp_subset = lfp.data[time_slice, 0]  # First channel only
sniff_subset = sniff.data[time_slice]
time_vector = np.arange(len(sniff_subset)) / sniff.rate

# Get inhalation and exhalation data
inhalation = behavior.data_interfaces["inhalation_time"]
exhalation = behavior.data_interfaces["exhalation_time"]

# Get inhalation and exhalation times within our time window
inh_mask = (inhalation.timestamps[:] >= 0) & (inhalation.timestamps[:] <= 20)
exh_mask = (exhalation.timestamps[:] >= 0) & (exhalation.timestamps[:] <= 20)

print(f"Inhalation events in time window: {np.sum(inh_mask)}")
print(f"Exhalation events in time window: {np.sum(exh_mask)}")

inh_times = inhalation.timestamps[inh_mask]
exh_times = exhalation.timestamps[exh_mask]

# Plot the data
plt.figure(figsize=(15, 12))

# Plot the sniff signal
plt.subplot(3, 1, 1)
plt.plot(time_vector, sniff_subset)
plt.title('Raw Sniff Signal')
plt.ylabel('Voltage (V)')
plt.grid(True)

# Plot the LFP signal
plt.subplot(3, 1, 2)
plt.plot(time_vector, lfp_subset)
plt.title('LFP Signal (Channel 0)')
plt.ylabel('Voltage (V)')
plt.grid(True)

# Plot the inhalation and exhalation events
plt.subplot(3, 1, 3)
# Create an event plot
ymin, ymax = -1, 1
for t in inh_times:
    plt.axvline(x=t, ymin=0.45, ymax=0.55, color='b', linewidth=1)
    
for t in exh_times:
    plt.axvline(x=t, ymin=0.55, ymax=0.65, color='r', linewidth=1)

# Add a clearer marker for inhalation and exhalation events
if len(inh_times) > 0:
    plt.scatter(inh_times, np.ones_like(inh_times) * 0.5, color='blue', marker='o', s=50, label='Inhalation')
if len(exh_times) > 0:
    plt.scatter(exh_times, np.ones_like(exh_times) * 0.6, color='red', marker='o', s=50, label='Exhalation')

plt.title('Sniffing Events')
plt.ylim(0, 1)
plt.ylabel('Event Type')
plt.xlabel('Time (s)')
plt.yticks([])  # Hide y-axis ticks as they're not meaningful
plt.legend()
plt.grid(True, axis='x')

plt.tight_layout()
plt.savefig('explore/sniff_lfp_comparison_fixed.png')
plt.close()

# Calculate the relationship between inhalation and exhalation
if len(inh_times) > 0 and len(exh_times) > 0:
    # Find inhalation-to-exhalation intervals
    inh_exh_pairs = []
    for inh_time in inh_times:
        # Find the next exhalation after this inhalation
        next_exh = exh_times[exh_times > inh_time]
        if len(next_exh) > 0:
            inh_exh_pairs.append((inh_time, next_exh[0]))
    
    if len(inh_exh_pairs) > 0:
        inh_exh_pairs = np.array(inh_exh_pairs)
        inh_exh_intervals = inh_exh_pairs[:, 1] - inh_exh_pairs[:, 0]
        
        plt.figure(figsize=(10, 5))
        plt.hist(inh_exh_intervals, bins=20, alpha=0.7)
        plt.title('Histogram of Inhalation to Exhalation Intervals')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Count')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('explore/inh_exh_intervals.png')
        plt.close()

# Calculate and plot the breathing cycle (inhalation-to-inhalation interval)
if len(inh_times) > 1:
    inh_intervals = np.diff(inh_times)
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Breathing cycle over time
    plt.subplot(2, 1, 1)
    plt.plot(inh_times[:-1], inh_intervals, marker='o', linestyle='-', markersize=4)
    plt.title('Breathing Cycle Duration Over Time')
    plt.ylabel('Inhalation-to-Inhalation Interval (s)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    
    # Plot 2: Histogram of breathing cycle durations
    plt.subplot(2, 1, 2)
    plt.hist(inh_intervals, bins=20, alpha=0.7)
    plt.title('Histogram of Breathing Cycle Durations')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('explore/breathing_cycle.png')
    plt.close()

# Analyze the relationship between LFP power in different frequency bands and the breathing cycle
if len(inh_times) > 5:  # Ensure we have enough data points
    # Define frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 80)
    }
    
    # Get a longer segment of LFP data for analysis
    time_window = 60  # seconds
    lfp_data = lfp.data[0:int(time_window*lfp.rate), 0]
    
    # Get inhalations within this window
    inh_mask_long = (inhalation.timestamps[:] >= 0) & (inhalation.timestamps[:] <= time_window)
    inh_times_long = inhalation.timestamps[inh_mask_long]
    
    # Define window around each inhalation
    pre_window = 0.5  # seconds before inhalation
    post_window = 1.0  # seconds after inhalation
    
    # Convert to samples
    pre_samples = int(pre_window * lfp.rate)
    post_samples = int(post_window * lfp.rate)
    
    # Initialize arrays for band power
    band_powers = {band: [] for band in bands}
    
    # For each inhalation, extract window and calculate band powers
    for inh_time in inh_times_long:
        idx = int(inh_time * lfp.rate)
        
        # Check if we have enough data before and after
        if idx >= pre_samples and idx + post_samples < len(lfp_data):
            # Extract window
            window = lfp_data[idx - pre_samples:idx + post_samples]
            
            # Calculate power spectrum
            f, Pxx = signal.welch(window, fs=lfp.rate, nperseg=min(512, len(window)))
            
            # Calculate power in each band
            for band, (low, high) in bands.items():
                band_mask = (f >= low) & (f <= high)
                if np.any(band_mask):
                    band_powers[band].append(np.mean(Pxx[band_mask]))
    
    # Plot average band powers
    plt.figure(figsize=(10, 6))
    band_names = list(bands.keys())
    mean_powers = [np.mean(band_powers[band]) for band in band_names]
    std_powers = [np.std(band_powers[band]) / np.sqrt(len(band_powers[band])) for band in band_names]
    
    plt.bar(band_names, mean_powers, yerr=std_powers, capsize=10, alpha=0.7)
    plt.title('LFP Power in Different Frequency Bands Around Inhalation')
    plt.ylabel('Power Spectral Density')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('explore/lfp_power_by_band.png')
    plt.close()

print("Analysis completed and visualizations saved to explore directory.")