# This script specifically explores the exhalation events in the dataset

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/e392479c-8683-4424-a75b-34af512a17a2/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get behavior data with inhalation and exhalation times
behavior = nwb.processing["behavior"]
exhalation = behavior.data_interfaces["exhalation_time"]
inhalation = behavior.data_interfaces["inhalation_time"]

# Get all exhalation and inhalation timestamps
exh_times = exhalation.timestamps[:]
inh_times = inhalation.timestamps[:]

# Print summary statistics
print(f"Total number of exhalation events: {len(exh_times)}")
print(f"Total number of inhalation events: {len(inh_times)}")
print(f"Time range of exhalation events: {exh_times.min()} to {exh_times.max()} seconds")
print(f"Time range of inhalation events: {inh_times.min()} to {inh_times.max()} seconds")

# Create histogram of exhalation events
plt.figure(figsize=(12, 6))
plt.hist(exh_times, bins=50, alpha=0.7)
plt.title('Histogram of Exhalation Event Times')
plt.xlabel('Time (seconds)')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/exhalation_hist.png')
plt.close()

# Create a plot showing exhalation times
plt.figure(figsize=(15, 5))
plt.plot(np.arange(len(exh_times)), exh_times, 'r.', markersize=3)
plt.title('Exhalation Event Times')
plt.xlabel('Event Number')
plt.ylabel('Time (seconds)')
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/exhalation_times.png')
plt.close()

# Now let's take a 30-second window of the sniff signal and mark the exhalation times
# Get sniff signal
sniff = nwb.acquisition["SniffSignal"]
start_time = 200  # Start at 200 seconds
window_size = 30  # 30 seconds
end_time = start_time + window_size

# Get the sniff signal data for this window
start_idx = int(start_time * sniff.rate)
end_idx = int(end_time * sniff.rate)
sniff_data = sniff.data[start_idx:end_idx]
time_vector = np.arange(len(sniff_data)) / sniff.rate + start_time

# Find exhalation events in this window
exh_in_window = exh_times[(exh_times >= start_time) & (exh_times < end_time)]

# Plot the sniff signal with exhalation marks
plt.figure(figsize=(15, 7))
plt.plot(time_vector, sniff_data, 'b-')

# Add markers for exhalation events
for t in exh_in_window:
    plt.axvline(x=t, color='r', linestyle='-', alpha=0.5)

plt.title(f'Sniff Signal with Exhalation Events (Time Window: {start_time}-{end_time}s)')
plt.xlabel('Time (seconds)')
plt.ylabel('Sniff Signal (Voltage)')
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/sniff_with_exhalations.png')
plt.close()

print("Analysis completed and visualizations saved to explore directory.")