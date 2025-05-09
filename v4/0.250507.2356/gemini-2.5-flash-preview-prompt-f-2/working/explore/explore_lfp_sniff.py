# Explore LFP and SniffSignal data

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# This script shows how to load the NWB file at https://api.dandiarchive.org/api/assets/63d19f03-2a35-48bd-a54f-9ab98ceb7be2/download/ in Python using PyNWB

# Load
url = "https://api.dandiarchive.org/api/assets/63d19f03-2a35-48bd-a54f-9ab98ceb7be2/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access LFP and SniffSignal data
lfp_data = nwb.acquisition["LFP"].data
sniff_data = nwb.acquisition["SniffSignal"].data
lfp_starting_time = nwb.acquisition["LFP"].starting_time
lfp_rate = nwb.acquisition["LFP"].rate

# Load a small portion of data for plotting
num_points = 10000
lfp_subset = lfp_data[0:num_points, 0] # First channel of LFP
sniff_subset = sniff_data[0:num_points]
timestamps_subset = lfp_starting_time + (1/lfp_rate) * np.arange(num_points)

# Create a plot
plt.figure(figsize=(12, 6))
plt.plot(timestamps_subset, lfp_subset, label="LFP (Channel 0)")
plt.plot(timestamps_subset, sniff_subset, label="Sniff Signal")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("LFP and Sniff Signal Subset")
plt.legend()
plt.savefig("explore/lfp_sniff_subset.png")
plt.close()

io.close() # Close the NWB file