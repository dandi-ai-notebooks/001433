# This script loads the first five seconds of LFP data from the NWB file
# https://api.dandiarchive.org/api/assets/63d19f03-2a35-48bd-a54f-9ab98ceb7be2/download/
# and creates an overview plot of all 16 channels. It also prints stats about the segment.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/63d19f03-2a35-48bd-a54f-9ab98ceb7be2/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()
LFP = nwb.acquisition["LFP"]
rate = LFP.rate  # 1000.0 Hz
n_channels = LFP.data.shape[1]
n_samples_5sec = int(5 * rate)

lfp_snippet = LFP.data[:n_samples_5sec, :]

# Get the time vector (in seconds)
time = np.arange(n_samples_5sec) / rate

plt.figure(figsize=(12, 10))
for i in range(n_channels):
    plt.plot(time, lfp_snippet[:, i] + i * 0.2, label=f'Ch {i}')  # offset for clarity

plt.xlabel('Time (s)')
plt.ylabel('Voltage + offset (V)')
plt.title('First 5 seconds of LFP (all 16 channels, offset for clarity)')
plt.tight_layout()
plt.savefig('explore/lfp_overview.png')

print("LFP segment shape:", lfp_snippet.shape)
print("LFP segment mean (all channels):", np.mean(lfp_snippet))
print("LFP segment std (all channels):", np.std(lfp_snippet))
print("Channels plotted:", n_channels)
print("Samples per channel (5 seconds):", n_samples_5sec)