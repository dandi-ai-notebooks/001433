# %% [markdown]
# # Exploring Dandiset 001433: Breathing rhythm and place dataset
# 
# **Note**: This notebook was AI-generated and has not been fully verified. Please interpret the code and results cautiously.

# %% [markdown]
# ## Overview of Dandiset
# The Dandiset **Breathing rhythm and place dataset** (ID: 001433, Version: 0.250507.2356) contains behavioral and electrophysiological data from recordings of sniffing, video, and olfactory bulb electrophysiology in freely behaving mice.
# Link: https://dandiarchive.org/dandiset/001433/0.250507.2356

# %% [markdown]
# **Notebook Contents**
# 1. Load Dandiset metadata and assets using the DANDI API  
# 2. Select and load an NWB file from the Dandiset  
# 3. Inspect NWB file structure and metadata  
# 4. Visualize LFP and sniff signals  
# 5. Advanced visualization  
# 6. Summary and future directions

# %% [markdown]
# ## Required Packages
# The following packages are required and assumed to be installed:
# - itertools  
# - dandi.dandiapi  
# - pynwb  
# - h5py  
# - remfile  
# - numpy  
# - pandas  
# - matplotlib

# %%
from itertools import islice
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001433", "0.250507.2356")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# List some assets in the Dandiset
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier})")

# %% [markdown]
# ## Select and Load NWB File
# We select the NWB file from subject 4127 (session at 2025-05-07T15:30:20).
# - File path: `sub-4127/sub-4127_ses-20250507T153020_ecephys.nwb`  
# - Asset URL: https://api.dandiarchive.org/api/assets/e392479c-8683-4424-a75b-34af512a17a2/download/

# %%
import pynwb
import h5py
import remfile

url = "https://api.dandiarchive.org/api/assets/e392479c-8683-4424-a75b-34af512a17a2/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, mode='r')
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# %% [markdown]
# ### NWB File Metadata
# - Session description: `nwb.session_description`  
# - Identifier: `nwb.identifier`  
# - Session start time: `nwb.session_start_time`  
# - Institution: `nwb.institution`  
# - Laboratory: `nwb.lab`  
# - Experimenter: `nwb.experimenter`

# %%
print("Session description:", nwb.session_description)
print("Identifier:", nwb.identifier)
print("Session start time:", nwb.session_start_time)
print("Institution:", nwb.institution)
print("Laboratory:", nwb.lab)
print("Experimenter:", nwb.experimenter)

# %% [markdown]
# ## NWB File Structure Summary
# ```
# - acquisition
#   - LFP (ElectricalSeries)
#   - SniffSignal (TimeSeries)
# - processing
#   - behavior (ProcessingModule)
#     - exhalation_time (TimeSeries)
#     - inhalation_time (TimeSeries)
# - electrodes (DynamicTable)
# - electrode_groups (ElectrodeGroup)
# - devices (Device)
# - subject (Subject)
# ```
# 
# **Explore in NeuroSift:**  
# https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/e392479c-8683-4424-a75b-34af512a17a2/download/&dandisetId=001433&dandisetVersion=draft

# %% [markdown]
# ## Visualize LFP and Sniff Signals
# We load a subset of data for efficiency (first 10,000 samples).

# %%
import numpy as np
import matplotlib.pyplot as plt

# Access time series
LFP = nwb.acquisition["LFP"]
Sniff = nwb.acquisition["SniffSignal"]

# Subset data
n_samples = 10000
lfp_data = LFP.data[:n_samples, 0]
lfp_times = np.arange(n_samples) / LFP.rate

sniff_data = Sniff.data[:n_samples]
sniff_times = np.arange(n_samples) / Sniff.rate

# Plot
plt.figure(figsize=(12, 4))
plt.plot(lfp_times, lfp_data, label='LFP (ch0)', alpha=0.7)
plt.plot(sniff_times, sniff_data, label='Sniff', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Volts')
plt.title('LFP Channel 0 and Sniff Signal (first 10k samples)')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Advanced Visualization
# Example: Plot the first 4 LFP channels in stacked traces.

# %%
n_ch = 4
times = lfp_times

fig, axs = plt.subplots(n_ch, 1, figsize=(10, 6), sharex=True)
for i, ax in enumerate(axs):
    ax.plot(times, LFP.data[:n_samples, i], color='C{}'.format(i))
    ax.set_ylabel(f'ch{i}')
axs[-1].set_xlabel('Time (s)')
fig.suptitle('First 4 LFP Channels (first 10k samples)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary and Future Directions
# This notebook demonstrated how to:
# - Load Dandiset metadata and assets  
# - Select and stream NWB data remotely using PyNWB  
# - Inspect NWB file metadata and structure  
# - Visualize time series data (LFP and sniff signals)  
# - Create multi-channel and combined plots  
# 
# **Future analyses** could include:
# - Automated sniff event detection and alignment  
# - Frequency-domain analysis (e.g., power spectral density of LFP)  
# - Correlation analysis between sniffing and LFP rhythms  
# - Exploration of processed behavioral features (e.g., exhalation/inhalation timestamps)