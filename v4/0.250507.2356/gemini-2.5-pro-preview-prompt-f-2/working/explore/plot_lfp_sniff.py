# This script loads a segment of LFP and SniffSignal data and plots them.
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Apply seaborn styling
sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/63d19f03-2a35-48bd-a54f-9ab98ceb7be2/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Get LFP and SniffSignal data
LFP = nwb.acquisition["LFP"]
SniffSignal = nwb.acquisition["SniffSignal"]

# Define segment to plot (e.g., first 2 seconds)
sampling_rate_lfp = LFP.rate
duration_seconds = 2
num_samples_lfp = int(sampling_rate_lfp * duration_seconds)

sampling_rate_sniff = SniffSignal.rate
num_samples_sniff = int(sampling_rate_sniff * duration_seconds)

# Load data for the first channel of LFP and SniffSignal
lfp_data_segment = LFP.data[:num_samples_lfp, 0] # First channel
sniff_data_segment = SniffSignal.data[:num_samples_sniff]
time_lfp = np.arange(num_samples_lfp) / sampling_rate_lfp
time_sniff = np.arange(num_samples_sniff) / sampling_rate_sniff

# Create plot
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot LFP data
axs[0].plot(time_lfp, lfp_data_segment)
axs[0].set_title('LFP Data (Channel 0)')
axs[0].set_ylabel('Voltage (Volts)')

# Plot SniffSignal data
axs[1].plot(time_sniff, sniff_data_segment)
axs[1].set_title('Sniff Signal')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Voltage (Volts)')

plt.tight_layout()
plt.savefig('explore/lfp_sniff_plot.png')
print("Plot saved to explore/lfp_sniff_plot.png")

io.close()