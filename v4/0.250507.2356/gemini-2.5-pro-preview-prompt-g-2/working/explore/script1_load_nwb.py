# script1_load_nwb.py
# Objective: Load the NWB file and print basic information about its structure and metadata.

import pynwb
import h5py
import remfile

# URL for the NWB file (hard-coded as per instructions)
url = "https://api.dandiarchive.org/api/assets/63d19f03-2a35-48bd-a54f-9ab98ceb7be2/download/"

print(f"Attempting to load NWB file from: {url}")

try:
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file, 'r') # Ensure read-only mode
    io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Ensure read-only mode for NWBHDF5IO
    nwb = io.read()

    print("\nNWB File Loaded Successfully.")
    print("==============================")
    print(f"Session Description: {nwb.session_description}")
    print(f"Identifier: {nwb.identifier}")
    print(f"Session Start Time: {nwb.session_start_time}")
    print(f"Experimenter: {nwb.experimenter}")
    print(f"Lab: {nwb.lab}")
    print(f"Institution: {nwb.institution}")
    print(f"Keywords: {list(nwb.keywords[:])}")

    print("\nAcquisition Data:")
    print("-----------------")
    if "LFP" in nwb.acquisition:
        LFP = nwb.acquisition["LFP"]
        print(f"  LFP ElectricalSeries:")
        print(f"    Description: {LFP.description}")
        print(f"    Data shape: {LFP.data.shape}")
        print(f"    Rate: {LFP.rate} Hz")
        print(f"    Unit: {LFP.unit}")
        print(f"    Electrodes table shape: {LFP.electrodes.table.to_dataframe().shape}")
        print(f"    Electrodes table columns: {list(LFP.electrodes.table.colnames)}")
    else:
        print("  LFP data not found in acquisition.")

    if "SniffSignal" in nwb.acquisition:
        SniffSignal = nwb.acquisition["SniffSignal"]
        print(f"  SniffSignal TimeSeries:")
        print(f"    Description: {SniffSignal.description}")
        print(f"    Data shape: {SniffSignal.data.shape}")
        print(f"    Rate: {SniffSignal.rate} Hz")
        print(f"    Unit: {SniffSignal.unit}")
    else:
        print("  SniffSignal data not found in acquisition.")

    print("\nProcessing Data (behavior):")
    print("---------------------------")
    if "behavior" in nwb.processing:
        behavior_module = nwb.processing["behavior"]
        print(f"  Behavior ProcessingModule description: {behavior_module.description}")
        if "exhalation_time" in behavior_module.data_interfaces:
            exhalation_time = behavior_module.data_interfaces["exhalation_time"]
            print(f"    Exhalation TimeSeries:")
            print(f"      Description: {exhalation_time.description}")
            print(f"      Data shape: {exhalation_time.data.shape}")
            print(f"      Timestamps shape: {exhalation_time.timestamps.shape}")
            print(f"      Unit: {exhalation_time.unit}")
        else:
            print("    Exhalation time data not found.")

        if "inhalation_time" in behavior_module.data_interfaces:
            inhalation_time = behavior_module.data_interfaces["inhalation_time"]
            print(f"    Inhalation TimeSeries:")
            print(f"      Description: {inhalation_time.description}")
            print(f"      Data shape: {inhalation_time.data.shape}")
            print(f"      Timestamps shape: {inhalation_time.timestamps.shape}")
            print(f"      Unit: {inhalation_time.unit}")
        else:
            print("    Inhalation time data not found.")
    else:
        print("  Behavior processing module not found.")
    
    print("\nElectrodes Table:")
    print("-----------------")
    if nwb.electrodes is not None:
        electrodes_df = nwb.electrodes.to_dataframe()
        print(f"  Electrodes table shape: {electrodes_df.shape}")
        print(f"  Electrodes table columns: {list(electrodes_df.columns)}")
        print("  First 5 rows of electrodes table:")
        print(electrodes_df.head().to_string())
    else:
        print("  No electrodes table found at the root of the NWB file.")


except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    if 'io' in locals() and io is not None:
        io.close()
    print("\nScript finished.")