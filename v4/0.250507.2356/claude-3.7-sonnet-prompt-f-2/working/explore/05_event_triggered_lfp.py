# This script analyzes LFP activity around exhalation events (event-triggered analysis)

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/e392479c-8683-4424-a75b-34af512a17a2/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get data
lfp = nwb.acquisition["LFP"]
behavior = nwb.processing["behavior"]
exhalation = behavior.data_interfaces["exhalation_time"]

# Parameters for event-triggered analysis
fs = lfp.rate  # Sampling rate
pre_window = 2.0  # seconds before event
post_window = 2.0  # seconds after event
pre_samples = int(pre_window * fs)
post_samples = int(post_window * fs)

# Get exhalation timestamps
exh_times = exhalation.timestamps[:]

# Select events to analyze (first 100 events for efficiency)
n_events_max = 100
n_events = min(n_events_max, len(exh_times))
selected_events = exh_times[:n_events]
print(f"Analyzing {n_events} exhalation events.")

# Choose a few electrode channels to analyze
channels_to_analyze = [0, 4, 8, 12]  # Sample channels across the 16 channels
num_channels = len(channels_to_analyze)

# Initialize array to hold event-triggered data
# Shape: [n_events, n_channels, n_timepoints]
event_triggered_data = np.zeros((n_events, num_channels, pre_samples + post_samples))

# Extract LFP segments around each event
for i, event_time in enumerate(selected_events):
    # Convert event time to sample index
    event_sample = int(event_time * fs)
    
    # Define extraction range
    start_idx = max(0, event_sample - pre_samples)
    end_idx = min(lfp.data.shape[0], event_sample + post_samples)
    
    # Check if we have enough data before and after
    if start_idx >= 0 and end_idx <= lfp.data.shape[0] and (end_idx - start_idx) == (pre_samples + post_samples):
        for j, chan in enumerate(channels_to_analyze):
            event_triggered_data[i, j, :] = lfp.data[start_idx:end_idx, chan]

# Create time vector for plotting
time_vector = np.linspace(-pre_window, post_window, pre_samples + post_samples)

# Calculate event-triggered average for each channel
event_triggered_avg = np.mean(event_triggered_data, axis=0)

# Plot event-triggered averages
plt.figure(figsize=(12, 10))
for i, chan in enumerate(channels_to_analyze):
    plt.subplot(num_channels, 1, i+1)
    plt.plot(time_vector, event_triggered_avg[i, :])
    plt.axvline(x=0, color='r', linestyle='--')  # Mark event time
    plt.title(f'Event-Triggered Average for Channel {chan}')
    plt.ylabel('Voltage (V)')
    if i == num_channels - 1:  # Only add xlabel to bottom subplot
        plt.xlabel('Time (s)')
    plt.grid(True)
plt.tight_layout()
plt.savefig('explore/event_triggered_avg_lfp.png')
plt.close()

# Calculate and plot time-frequency analysis for channel 0
channel = 0
chan_idx = channels_to_analyze.index(channel) if channel in channels_to_analyze else 0

# Get the event-triggered average for this channel
avg_lfp = event_triggered_avg[chan_idx, :]

# Calculate time-frequency representation using spectrogram
f, t, Sxx = signal.spectrogram(avg_lfp, fs=fs, nperseg=256, noverlap=128, scaling='density')

# Plot time-frequency representation
plt.figure(figsize=(10, 6))
plt.pcolormesh(t - pre_window, f, 10 * np.log10(Sxx), shading='auto')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.axvline(x=0, color='r', linestyle='--')  # Mark event time
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title(f'Time-Frequency Analysis around Exhalation Events (Channel {channel})')
plt.ylim(0, 100)  # Limit frequency display to 0-100 Hz
plt.tight_layout()
plt.savefig('explore/event_triggered_freq.png')
plt.close()

print("Analysis completed and visualizations saved to explore directory.")