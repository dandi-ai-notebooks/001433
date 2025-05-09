# This script loads the NWB file and prints basic information.
import pynwb
import h5py
import remfile

# Load
url = "https://api.dandiarchive.org/api/assets/63d19f03-2a35-48bd-a54f-9ab98ceb7be2/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Added mode='r' for read-only
nwb = io.read()

print(f"Session Description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"Experimenter: {nwb.experimenter}")
print(f"Keywords: {nwb.keywords[:]}")

# Print LFP info
LFP = nwb.acquisition["LFP"]
print(f"LFP data shape: {LFP.data.shape}")
print(f"LFP rate: {LFP.rate}")
print(f"LFP electrodes columns: {LFP.electrodes.table.colnames}")
# print LFP electrodes table first 5 rows
print("LFP electrodes table (first 5 rows):")
print(LFP.electrodes.table.to_dataframe().head())


# Print SniffSignal info
SniffSignal = nwb.acquisition["SniffSignal"]
print(f"SniffSignal data shape: {SniffSignal.data.shape}")
print(f"SniffSignal rate: {SniffSignal.rate}")

# Print processing module info
if "behavior" in nwb.processing:
    behavior = nwb.processing["behavior"]
    print("Behavior processing module found.")
    if "exhalation_time" in behavior.data_interfaces:
        exhalation_time = behavior.data_interfaces["exhalation_time"]
        print(f"Exhalation time data shape: {exhalation_time.data.shape}")
        print(f"Exhalation time timestamps shape: {exhalation_time.timestamps.shape}")
    if "inhalation_time" in behavior.data_interfaces:
        inhalation_time = behavior.data_interfaces["inhalation_time"]
        print(f"Inhalation time data shape: {inhalation_time.data.shape}")
        print(f"Inhalation time timestamps shape: {inhalation_time.timestamps.shape}")

io.close() # It's good practice to close the file