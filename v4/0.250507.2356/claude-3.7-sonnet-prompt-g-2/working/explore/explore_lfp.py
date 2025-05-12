# This script explores the LFP data from the NWB file
# We'll load the NWB file, examine its structure, and visualize some LFP data

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

# Print basic information about the file
print(f"NWB File ID: {nwb.identifier}")
print(f"Session Description: {nwb.session_description}")
print(f"Experiment Description: {nwb.experiment_description}")
print(f"Lab: {nwb.lab}")
print(f"Institution: {nwb.institution}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject Sex: {nwb.subject.sex}")
print(f"Subject Age: {nwb.subject.age}")
print(f"Subject Species: {nwb.subject.species}")
print(f"Keywords: {nwb.keywords[:]}")

# Access the LFP data
lfp = nwb.acquisition["LFP"]
print("\nLFP Data Information:")
print(f"Description: {lfp.description}")
print(f"Unit: {lfp.unit}")
print(f"Sampling Rate: {lfp.rate} Hz")
print(f"Data Shape: {lfp.data.shape}")

# Get information about electrodes
electrodes_df = nwb.electrodes.to_dataframe()
print("\nElectrodes Information:")
print(electrodes_df)

# Plot a short segment of LFP data (first 10 seconds, first 5 channels)
# 10 seconds at 1000Hz = 10,000 samples
time_slice = slice(0, 10000)
channel_slice = slice(0, 5)

# Create time array (in seconds)
time = np.arange(time_slice.start, time_slice.stop) / lfp.rate

# Get data subset
lfp_data_subset = lfp.data[time_slice, channel_slice]

# Plot
plt.figure(figsize=(12, 8))
for i in range(lfp_data_subset.shape[1]):
    # Offset each channel for better visualization
    plt.plot(time, lfp_data_subset[:, i] + i*0.0005, label=f"Channel {i}")

plt.xlabel('Time (s)')
plt.ylabel('Voltage (V) + offset')
plt.title('LFP Signals from First 5 Channels (First 10 seconds)')
plt.legend()
plt.grid(True)
plt.savefig('explore/lfp_first_10s.png')
plt.close()

# Plot a spectrogram of the first channel
plt.figure(figsize=(10, 6))
# Take first 60 seconds of data for channel 0 to see frequency components
# 60 seconds at 1000Hz = 60,000 samples
data_for_spectrogram = lfp.data[0:60000, 0]
plt.specgram(data_for_spectrogram, NFFT=1024, Fs=lfp.rate, 
            noverlap=512, cmap='viridis')
plt.colorbar(label='Power Spectral Density (dB)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram of First LFP Channel (First 60 seconds)')
plt.savefig('explore/lfp_spectrogram.png')
plt.close()

# Display electrode group information
print("\nElectrode Group Information:")
for group_name, group in nwb.electrode_groups.items():
    print(f"Group Name: {group_name}")
    print(f"Description: {group.description}")
    print(f"Location: {group.location}")
    print(f"Device: {group.device.description}")