# This script loads the inhalation and exhalation event times from the NWB file
# https://api.dandiarchive.org/api/assets/63d19f03-2a35-48bd-a54f-9ab98ceb7be2/download/
# and creates a raster/event plot for the first 5 seconds. It also prints event counts and timing range.

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
behavior = nwb.processing["behavior"]
inh = behavior.data_interfaces["inhalation_time"]
exh = behavior.data_interfaces["exhalation_time"]

inh_times = inh.timestamps[:]
exh_times = exh.timestamps[:]

# For plotting, select events in the first 5 seconds
window_end = 5.0
inh_times_win = inh_times[inh_times <= window_end]
exh_times_win = exh_times[exh_times <= window_end]

plt.figure(figsize=(12, 2))
plt.eventplot([inh_times_win, exh_times_win], colors=['tab:blue', 'tab:orange'], lineoffsets=[1, 0], linelengths=0.8)
plt.yticks([1, 0], ['Inhalation', 'Exhalation'])
plt.xlabel('Time (s)')
plt.title('Inhalation and Exhalation Events (First 5 seconds)')
plt.xlim(0, window_end)
plt.tight_layout()
plt.savefig('explore/inhal_exhal_events.png')

print('Total inhalation events in first 5 seconds:', len(inh_times_win))
print('Total exhalation events in first 5 seconds:', len(exh_times_win))
print('First 3 inhalation times:', inh_times_win[:3])
print('First 3 exhalation times:', exh_times_win[:3])