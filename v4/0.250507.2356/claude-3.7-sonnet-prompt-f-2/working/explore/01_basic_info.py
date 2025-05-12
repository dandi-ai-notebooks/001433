# This script explores the basic information about the NWB file, 
# including metadata and data shapes

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/e392479c-8683-4424-a75b-34af512a17a2/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information
print("NWB File Information:")
print(f"Session Description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"Experiment Description: {nwb.experiment_description}")
print(f"Lab: {nwb.lab}")
print(f"Institution: {nwb.institution}")
print(f"Keywords: {list(nwb.keywords[:])}")

# Print subject information
print("\nSubject Information:")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Age: {nwb.subject.age}")
print(f"Description: {nwb.subject.description}")

# Print acquisition data
print("\nAcquisition Data:")
for key in nwb.acquisition.keys():
    data = nwb.acquisition[key]
    if hasattr(data, 'data'):
        data_shape = data.data.shape
        data_type = data.data.dtype
    else:
        data_shape = "N/A"
        data_type = "N/A"
    print(f"{key}: Shape {data_shape}, Type {data_type}")
    print(f"  Description: {data.description}")
    print(f"  Unit: {data.unit}")
    print(f"  Rate: {data.rate} Hz")
    
# Print processing data
print("\nProcessing Data:")
for module_name in nwb.processing.keys():
    module = nwb.processing[module_name]
    print(f"Module: {module_name}")
    print(f"  Description: {module.description}")
    for interface_name in module.data_interfaces.keys():
        interface = module.data_interfaces[interface_name]
        if hasattr(interface, 'data'):
            data_shape = interface.data.shape
            data_type = interface.data.dtype
        else:
            data_shape = "N/A"
            data_type = "N/A"
        print(f"  {interface_name}: Shape {data_shape}, Type {data_type}")
        print(f"    Description: {interface.description}")
        if hasattr(interface, 'timestamps'):
            print(f"    Timestamps Shape: {interface.timestamps.shape}")
            
# Print electrode information
print("\nElectrode Information:")
if hasattr(nwb, 'electrodes'):
    electrodes_df = nwb.electrodes.to_dataframe()
    print(f"Total Electrodes: {len(electrodes_df)}")
    print(electrodes_df.head())

# Access a small sample of LFP data to verify it can be read
print("\nSample of LFP Data:")
lfp = nwb.acquisition["LFP"]
sample_size = 5  # Just get a few data points
sample_data = lfp.data[0:sample_size, 0:3]  # First 5 timepoints, first 3 channels
print(sample_data)

# Access a small sample of sniff signal data
print("\nSample of Sniff Signal Data:")
sniff = nwb.acquisition["SniffSignal"]
sniff_sample = sniff.data[0:sample_size]
print(sniff_sample)

# Access a small sample of behavior data
print("\nSample of Behavior Data (Exhalation Times):")
if 'behavior' in nwb.processing:
    behavior = nwb.processing['behavior']
    if 'exhalation_time' in behavior.data_interfaces:
        exhalation = behavior.data_interfaces['exhalation_time']
        sample = min(5, len(exhalation.data))
        exh_data = exhalation.data[0:sample]
        exh_timestamps = exhalation.timestamps[0:sample]
        print(f"Data: {exh_data}")
        print(f"Timestamps: {exh_timestamps}")