# This script explores the sniffing data from the NWB file
# We'll load the data, examine inhalation and exhalation events, and visualize relationships with LFP

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory for plots if it doesn't exist
os.makedirs("explore", exist_ok=True)

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/63d19f03-2a35-48bd-a54f-9ab98ceb7be2/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get sniff signal and related data
sniff_signal = nwb.acquisition["SniffSignal"]
print("SniffSignal Information:")
print(f"Description: {sniff_signal.description}")
print(f"Unit: {sniff_signal.unit}")
print(f"Sampling Rate: {sniff_signal.rate} Hz")
print(f"Data Shape: {sniff_signal.data.shape}")

# Get inhalation and exhalation times
inhalation_time = nwb.processing["behavior"]["inhalation_time"]
exhalation_time = nwb.processing["behavior"]["exhalation_time"]

print("\nInhalation Time Information:")
print(f"Description: {inhalation_time.description}")
print(f"Number of events: {len(inhalation_time.timestamps)}")
print(f"First 5 timestamps (s):", inhalation_time.timestamps[:5])
print(f"First 5 values:", inhalation_time.data[:5])

print("\nExhalation Time Information:")
print(f"Description: {exhalation_time.description}")
print(f"Number of events: {len(exhalation_time.timestamps)}")
print(f"First 5 timestamps (s):", exhalation_time.timestamps[:5])
print(f"First 5 values:", exhalation_time.data[:5])

# Plot a 20-second segment of the raw sniff signal
plt.figure(figsize=(15, 5))
time_slice = slice(0, 20000)  # 20 seconds at 1000 Hz
time = np.arange(time_slice.start, time_slice.stop) / sniff_signal.rate
sniff_data = sniff_signal.data[time_slice]

plt.plot(time, sniff_data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.title('Raw Sniff Signal (First 20 seconds)')
plt.grid(True)
plt.savefig('explore/raw_sniff_signal.png')
plt.close()

# Plot sniff events (inhalations and exhalations) for first 60 seconds
plt.figure(figsize=(15, 6))

# Find inhalation and exhalation events within the first 60 seconds
inh_mask = inhalation_time.timestamps[:] < 60
exh_mask = exhalation_time.timestamps[:] < 60

inh_times = inhalation_time.timestamps[inh_mask]
exh_times = exhalation_time.timestamps[exh_mask]

# Calculate sniff durations and periods
sniff_durations = []
sniff_periods = []
sniff_times = []

# Ensure we have pairs of inhalation and exhalation
min_events = min(len(inh_times), len(exh_times))
for i in range(min_events-1):  # -1 to ensure we have a next inhalation for period calculation
    if exh_times[i] > inh_times[i]:  # Ensure proper sequence: inhalation->exhalation
        duration = exh_times[i] - inh_times[i]
        sniff_durations.append(duration)
        
        # Calculate period (from current inhalation to next inhalation)
        period = inh_times[i+1] - inh_times[i]
        sniff_periods.append(period)
        sniff_times.append(inh_times[i])

# Plot sniff durations
plt.subplot(3, 1, 1)
plt.plot(sniff_times, sniff_durations, 'o-')
plt.xlabel('Time (s)')
plt.ylabel('Duration (s)')
plt.title('Inhalation Duration Over Time')
plt.grid(True)

# Plot sniff periods
plt.subplot(3, 1, 2)
plt.plot(sniff_times, sniff_periods, 'o-')
plt.xlabel('Time (s)')
plt.ylabel('Period (s)')
plt.title('Sniff Period Over Time')
plt.grid(True)

# Plot sniff frequency (1/period)
plt.subplot(3, 1, 3)
sniff_freq = [1/period for period in sniff_periods]
plt.plot(sniff_times, sniff_freq, 'o-')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Sniffing Frequency Over Time')
plt.grid(True)

plt.tight_layout()
plt.savefig('explore/sniff_metrics.png')
plt.close()

# Plot simultaneous LFP and sniffing for a short segment
plt.figure(figsize=(15, 10))

# Time window (10 seconds)
start_time = 10  # seconds
end_time = 20    # seconds
time_slice = slice(int(start_time * sniff_signal.rate), int(end_time * sniff_signal.rate))
time = np.arange(time_slice.start, time_slice.stop) / sniff_signal.rate

# Get sniff data for this window
sniff_data = sniff_signal.data[time_slice]

# Get LFP data (from first 2 channels) for this window
lfp = nwb.acquisition["LFP"]
lfp_data_1 = lfp.data[time_slice, 0]
lfp_data_2 = lfp.data[time_slice, 1]

# Find inhalation and exhalation events within this window
inh_mask = (inhalation_time.timestamps[:] >= start_time) & (inhalation_time.timestamps[:] <= end_time)
exh_mask = (exhalation_time.timestamps[:] >= start_time) & (exhalation_time.timestamps[:] <= end_time)

inh_times = inhalation_time.timestamps[inh_mask]
exh_times = exhalation_time.timestamps[exh_mask]

# Plot Sniff Signal
plt.subplot(3, 1, 1)
plt.plot(time, sniff_data)
# Add vertical lines for inhalation and exhalation events
for t in inh_times:
    plt.axvline(t, color='green', linestyle='--', alpha=0.7)
for t in exh_times:
    plt.axvline(t, color='red', linestyle=':', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.title('Raw Sniff Signal')
# Add a legend
plt.axvline(-1, color='green', linestyle='--', alpha=0.7, label='Inhalation')
plt.axvline(-1, color='red', linestyle=':', alpha=0.7, label='Exhalation')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.grid(True)

# Plot LFP from first channel
plt.subplot(3, 1, 2)
plt.plot(time, lfp_data_1)
# Add vertical lines for inhalation and exhalation events
for t in inh_times:
    plt.axvline(t, color='green', linestyle='--', alpha=0.7)
for t in exh_times:
    plt.axvline(t, color='red', linestyle=':', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('LFP Channel 0')
plt.grid(True)

# Plot LFP from second channel
plt.subplot(3, 1, 3)
plt.plot(time, lfp_data_2)
# Add vertical lines for inhalation and exhalation events
for t in inh_times:
    plt.axvline(t, color='green', linestyle='--', alpha=0.7)
for t in exh_times:
    plt.axvline(t, color='red', linestyle=':', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('LFP Channel 1')
plt.grid(True)

plt.tight_layout()
plt.savefig('explore/sniff_lfp_alignment.png')
plt.close()

# Calculate and plot average LFP around sniff events
# We'll create a sniff-triggered average of LFP
plt.figure(figsize=(12, 8))

# Parameters for sniff-triggered average
pre_event = 0.5   # seconds before inhalation
post_event = 1.0  # seconds after inhalation
pre_samples = int(pre_event * lfp.rate)
post_samples = int(post_event * lfp.rate)
window_size = pre_samples + post_samples
time_axis = np.linspace(-pre_event, post_event, window_size)

# Collect LFP segments around each inhalation
# Use only inhalations that are sufficiently far from the start and end of recording
valid_inh_mask = (inhalation_time.timestamps[:] > pre_event) & \
                (inhalation_time.timestamps[:] < (lfp.data.shape[0] / lfp.rate - post_event))
valid_inh_times = inhalation_time.timestamps[valid_inh_mask]

# Limit to first 500 inhalations to keep computation manageable
max_events = 500
if len(valid_inh_times) > max_events:
    valid_inh_times = valid_inh_times[:max_events]

# Initialize array to hold LFP segments
lfp_segments = np.zeros((len(valid_inh_times), window_size, 2))  # For 2 channels

# Extract LFP around each inhalation
for i, inh_time in enumerate(valid_inh_times):
    inh_idx = int(inh_time * lfp.rate)  # Convert time to index
    start_idx = inh_idx - pre_samples
    end_idx = inh_idx + post_samples
    
    # Extract data for 2 channels
    lfp_segments[i, :, 0] = lfp.data[start_idx:end_idx, 0]
    lfp_segments[i, :, 1] = lfp.data[start_idx:end_idx, 1]

# Calculate the average
avg_lfp = np.mean(lfp_segments, axis=0)

# Plot the results
plt.subplot(2, 1, 1)
plt.plot(time_axis, avg_lfp[:, 0])
plt.axvline(0, color='green', linestyle='--', label='Inhalation')
plt.xlabel('Time relative to inhalation (s)')
plt.ylabel('Avg Voltage (V)')
plt.title(f'Sniff-Triggered Average LFP - Channel 0 (n={len(valid_inh_times)})')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_axis, avg_lfp[:, 1])
plt.axvline(0, color='green', linestyle='--', label='Inhalation')
plt.xlabel('Time relative to inhalation (s)')
plt.ylabel('Avg Voltage (V)')
plt.title(f'Sniff-Triggered Average LFP - Channel 1 (n={len(valid_inh_times)})')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('explore/sniff_triggered_lfp.png')
plt.close()

# Calculate sniffing statistics
sniff_periods_all = np.diff(inhalation_time.timestamps[:])
avg_sniff_period = np.mean(sniff_periods_all)
avg_sniff_freq = 1/avg_sniff_period
std_sniff_period = np.std(sniff_periods_all)
min_sniff_period = np.min(sniff_periods_all)
max_sniff_period = np.max(sniff_periods_all)

print("\nSniffing Statistics:")
print(f"Average sniff period: {avg_sniff_period:.4f} seconds")
print(f"Average sniff frequency: {avg_sniff_freq:.4f} Hz")
print(f"Standard deviation of sniff period: {std_sniff_period:.4f} seconds")
print(f"Minimum sniff period: {min_sniff_period:.4f} seconds")
print(f"Maximum sniff period: {max_sniff_period:.4f} seconds")
print(f"Total number of sniffs: {len(inhalation_time.timestamps[:])}")
print(f"Recording duration: {sniff_signal.data.shape[0]/sniff_signal.rate:.2f} seconds")