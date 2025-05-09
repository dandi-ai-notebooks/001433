# script2_plot_signals.py
# Objective: Load a segment of LFP data and SniffSignal data and plot them.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Use seaborn styling
sns.set_theme()

# URL for the NWB file
url = "https://api.dandiarchive.org/api/assets/63d19f03-2a35-48bd-a54f-9ab98ceb7be2/download/"

# Output file paths for plots
lfp_plot_path = "explore/lfp_segment.png"
sniff_plot_path = "explore/sniff_signal_segment.png"

print(f"Attempting to load NWB file from: {url}")
io = None  # Initialize io to None
try:
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file, 'r')
    io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
    nwb = io.read()
    print("NWB File Loaded Successfully.")

    # --- LFP Data ---
    if "LFP" in nwb.acquisition:
        LFP = nwb.acquisition["LFP"]
        print(f"LFP data shape: {LFP.data.shape}, Rate: {LFP.rate} Hz")

        # Select a 2-second segment
        time_duration_seconds = 2
        num_samples_to_plot = int(time_duration_seconds * LFP.rate)
        
        # Ensure we don't exceed available data
        num_samples_to_plot = min(num_samples_to_plot, LFP.data.shape[0])
        
        # Select first 3 channels, or fewer if not available
        num_channels_to_plot = min(3, LFP.data.shape[1])

        if num_samples_to_plot > 0 and num_channels_to_plot > 0:
            lfp_segment = LFP.data[:num_samples_to_plot, :num_channels_to_plot]
            time_vector_lfp = np.arange(num_samples_to_plot) / LFP.rate

            plt.figure(figsize=(12, 6))
            for i in range(num_channels_to_plot):
                plt.plot(time_vector_lfp, lfp_segment[:, i], label=f'Channel {LFP.electrodes.table["id"][i]}')
            plt.title(f'LFP Data (First {time_duration_seconds}s, {num_channels_to_plot} Channels)')
            plt.xlabel('Time (s)')
            plt.ylabel(f'Amplitude ({LFP.unit})')
            plt.legend()
            plt.grid(True)
            plt.savefig(lfp_plot_path)
            plt.close() # Close the plot to free memory
            print(f"LFP segment plot saved to {lfp_plot_path}")
        else:
            print("Not enough LFP data or channels to plot.")
    else:
        print("LFP data not found.")

    # --- Sniff Signal Data ---
    if "SniffSignal" in nwb.acquisition:
        SniffSignal = nwb.acquisition["SniffSignal"]
        print(f"SniffSignal data shape: {SniffSignal.data.shape}, Rate: {SniffSignal.rate} Hz")

        # Select a 2-second segment (same duration as LFP for comparison)
        time_duration_seconds = 2
        num_samples_to_plot_sniff = int(time_duration_seconds * SniffSignal.rate)
        
        # Ensure we don't exceed available data
        num_samples_to_plot_sniff = min(num_samples_to_plot_sniff, SniffSignal.data.shape[0])

        if num_samples_to_plot_sniff > 0:
            sniff_segment = SniffSignal.data[:num_samples_to_plot_sniff]
            time_vector_sniff = np.arange(num_samples_to_plot_sniff) / SniffSignal.rate

            plt.figure(figsize=(12, 4))
            plt.plot(time_vector_sniff, sniff_segment)
            plt.title(f'Sniff Signal (First {time_duration_seconds}s)')
            plt.xlabel('Time (s)')
            plt.ylabel(f'Amplitude ({SniffSignal.unit})')
            plt.grid(True)
            plt.savefig(sniff_plot_path)
            plt.close() # Close the plot to free memory
            print(f"Sniff signal segment plot saved to {sniff_plot_path}")
        else:
            print("Not enough SniffSignal data to plot.")
    else:
        print("SniffSignal data not found.")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    if io is not None:
        io.close()
    print("\nScript finished.")