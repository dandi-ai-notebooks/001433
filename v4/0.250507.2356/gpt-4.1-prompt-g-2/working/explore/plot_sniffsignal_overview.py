# This script loads the first five seconds of SniffSignal data from the NWB file
# https://api.dandiarchive.org/api/assets/63d19f03-2a35-48bd-a54f-9ab98ceb7be2/download/
# and creates an overview plot. It also prints stats about the segment.

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
SniffSignal = nwb.acquisition["SniffSignal"]
rate = SniffSignal.rate  # 1000.0 Hz
n_samples_5sec = int(5 * rate)

sniff_snippet = SniffSignal.data[:n_samples_5sec]

time = np.arange(n_samples_5sec) / rate

plt.figure(figsize=(12, 4))
plt.plot(time, sniff_snippet)
plt.xlabel('Time (s)')
plt.ylabel('Sniff signal (V)')
plt.title('First 5 seconds of SniffSignal')
plt.tight_layout()
plt.savefig('explore/sniffsignal_overview.png')

print("SniffSignal segment shape:", sniff_snippet.shape)
print("SniffSignal segment mean:", np.mean(sniff_snippet))
print("SniffSignal segment std:", np.std(sniff_snippet))
print("Samples plotted (5 seconds):", n_samples_5sec)