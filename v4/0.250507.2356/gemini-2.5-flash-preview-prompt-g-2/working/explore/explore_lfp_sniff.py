# This script loads and visualizes a subset of LFP and SniffSignal data.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load
url = "https://api.dandiarchive.org/api/assets/63d19f03-2a35-48bd-a54f-9ab98ceb7be2/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get LFP and SniffSignal data
lfp_data = nwb.acquisition["LFP"].data
sniff_data = nwb.acquisition["SniffSignal"].data

# Define the time window to load (e.g., first 10 seconds)
start_time = 0
end_time = 10
sampling_rate = nwb.acquisition["LFP"].rate # Assuming both have the same rate
start_index = int(start_time * sampling_rate)
end_index = int(end_time * sampling_rate)

# Load a subset of the data
lfp_subset = lfp_data[start_index:end_index, 0] # Load data from the first LFP channel
sniff_subset = sniff_data[start_index:end_index]

# Create a time vector for the subset
time_vector = np.arange(start_index, end_index) / sampling_rate

# Create plots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(time_vector, lfp_subset)
ax1.set_ylabel(f'LFP ({nwb.acquisition["LFP"].unit})')
ax1.set_title('LFP and Sniff Signal (subset)')

ax2.plot(time_vector, sniff_subset)
ax2.set_ylabel(f'Sniff Signal ({nwb.acquisition["SniffSignal"].unit})')
ax2.set_xlabel('Time (s)')

plt.tight_layout()

# Save the plot to a file
plt.savefig('explore/lfp_sniff_subset.png')

# Do not use plt.show() as it will hang the script.

io.close()