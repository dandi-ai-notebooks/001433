# This script provides a revised exploration of the sniffing data
# We'll examine the raw breathing signal in more detail and make sure we correctly
# identify breathing events

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

# Create directory for plots if it doesn't exist
os.makedirs("explore", exist_ok=True)

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/63d19f03-2a35-48bd-a54f-9ab98ceb7be2/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get sniff signal
sniff_signal = nwb.acquisition["SniffSignal"]
print("SniffSignal Information:")
print(f"Description: {sniff_signal.description}")
print(f"Unit: {sniff_signal.unit}")
print(f"Sampling Rate: {sniff_signal.rate} Hz")
print(f"Data Shape: {sniff_signal.data.shape}")

# Get inhalation and exhalation times from the file
inhalation_time = nwb.processing["behavior"]["inhalation_time"]
exhalation_time = nwb.processing["behavior"]["exhalation_time"]

print("\nInhalation Time Information:")
print(f"Description: {inhalation_time.description}")
print(f"Number of events: {len(inhalation_time.timestamps)}")
print(f"First 10 timestamps (s):", inhalation_time.timestamps[:10])

print("\nExhalation Time Information:")
print(f"Description: {exhalation_time.description}")
print(f"Number of events: {len(exhalation_time.timestamps)}")
print(f"First 10 timestamps (s):", exhalation_time.timestamps[:10])

# Since the provided inhalation/exhalation events seem to have issues, let's detect them directly
# Using 30 seconds of data for a detailed view
segment_start = 0  # start at the beginning
segment_length = 30000  # 30 seconds at 1000 Hz
segment_end = segment_start + segment_length

# Extract data segment
time = np.arange(segment_start, segment_end) / sniff_signal.rate
sniff_data = sniff_signal.data[segment_start:segment_end]

# Detect peaks (inhalation) and troughs (exhalation)
# Using the negative of the signal since inhalations appear to be troughs in this recording
peaks, _ = find_peaks(-sniff_data, distance=50, prominence=1000)  # Adjust parameters as needed
troughs, _ = find_peaks(sniff_data, distance=50, prominence=1000)  # Adjust parameters as needed

# Convert indices to times
peak_times = peaks / sniff_signal.rate
trough_times = troughs / sniff_signal.rate

# Plot the raw signal with detected peaks and troughs
plt.figure(figsize=(15, 6))
plt.plot(time, sniff_data)
plt.plot(peak_times, sniff_data[peaks], "rv", label="Detected Inhalation")
plt.plot(trough_times, sniff_data[troughs], "g^", label="Detected Exhalation")

# Mark the original inhalation and exhalation events if they fall within this time window
original_inh_mask = (inhalation_time.timestamps[:] >= time[0]) & (inhalation_time.timestamps[:] <= time[-1])
original_exh_mask = (exhalation_time.timestamps[:] >= time[0]) & (exhalation_time.timestamps[:] <= time[-1])

if np.any(original_inh_mask):
    plt.plot(inhalation_time.timestamps[original_inh_mask], 
             np.zeros_like(inhalation_time.timestamps[original_inh_mask]) - 1000, 
             'bD', markersize=8, label="Original Inhalation")

if np.any(original_exh_mask):
    plt.plot(exhalation_time.timestamps[original_exh_mask], 
             np.zeros_like(exhalation_time.timestamps[original_exh_mask]) + 1000, 
             'mD', markersize=8, label="Original Exhalation")

plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.title('Raw Sniff Signal with Detected Breathing Events')
plt.legend()
plt.grid(True)
plt.savefig('explore/detected_breathing_events.png')
plt.close()

# Calculate sniff periods and frequencies from detected peaks
if len(peak_times) > 1:
    sniff_periods = np.diff(peak_times)
    sniff_frequencies = 1 / sniff_periods
    
    # Plot sniffing metrics over time
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(peak_times[:-1], sniff_periods, 'o-')
    plt.xlabel('Time (s)')
    plt.ylabel('Period (s)')
    plt.title('Sniff Period Over Time')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(peak_times[:-1], sniff_frequencies, 'o-')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Sniffing Frequency Over Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('explore/detected_sniff_metrics.png')
    plt.close()
    
    # Print statistics
    print("\nDetected Sniffing Statistics:")
    print(f"Number of detected inhalations: {len(peak_times)}")
    print(f"Number of detected exhalations: {len(trough_times)}")
    print(f"Average sniff period: {np.mean(sniff_periods):.4f} seconds")
    print(f"Average sniff frequency: {np.mean(sniff_frequencies):.4f} Hz")
    print(f"Standard deviation of sniff period: {np.std(sniff_periods):.4f} seconds")
    print(f"Min/Max sniff period: {np.min(sniff_periods):.4f}/{np.max(sniff_periods):.4f} seconds")

# Now with better detection, let's explore the relationship with LFP
# Create sniff-triggered LFP average using our detected inhalations
if len(peak_times) > 0:
    # Parameters for sniff-triggered average
    pre_event = 0.5   # seconds before inhalation
    post_event = 1.0  # seconds after inhalation
    pre_samples = int(pre_event * sniff_signal.rate)
    post_samples = int(post_event * sniff_signal.rate)
    window_size = pre_samples + post_samples
    time_axis = np.linspace(-pre_event, post_event, window_size)
    
    # Convert peak times to indices
    peak_indices = peaks
    
    # Filter out peaks that are too close to the beginning or end of the data
    valid_peaks = [idx for idx in peak_indices 
                  if idx >= pre_samples and idx < len(sniff_data) - post_samples]
    
    # Get LFP data
    lfp = nwb.acquisition["LFP"]
    
    # Initialize arrays for LFP segments
    lfp_segments_ch0 = np.zeros((len(valid_peaks), window_size))
    lfp_segments_ch1 = np.zeros((len(valid_peaks), window_size))
    
    # Extract LFP around each inhalation
    for i, peak_idx in enumerate(valid_peaks):
        start_idx = peak_idx - pre_samples
        end_idx = peak_idx + post_samples
        
        # Make sure we're within the LFP data range
        if start_idx >= 0 and end_idx < lfp.data.shape[0]:
            lfp_segments_ch0[i, :] = lfp.data[start_idx:end_idx, 0]
            lfp_segments_ch1[i, :] = lfp.data[start_idx:end_idx, 1]
    
    # Calculate the average
    avg_lfp_ch0 = np.mean(lfp_segments_ch0, axis=0)
    avg_lfp_ch1 = np.mean(lfp_segments_ch1, axis=0)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, avg_lfp_ch0)
    plt.axvline(0, color='green', linestyle='--', label='Inhalation')
    plt.xlabel('Time relative to inhalation (s)')
    plt.ylabel('Avg Voltage (V)')
    plt.title(f'Improved Sniff-Triggered Average LFP - Channel 0 (n={len(valid_peaks)})')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, avg_lfp_ch1)
    plt.axvline(0, color='green', linestyle='--', label='Inhalation')
    plt.xlabel('Time relative to inhalation (s)')
    plt.ylabel('Avg Voltage (V)')
    plt.title(f'Improved Sniff-Triggered Average LFP - Channel 1 (n={len(valid_peaks)})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('explore/improved_sniff_triggered_lfp.png')
    plt.close()