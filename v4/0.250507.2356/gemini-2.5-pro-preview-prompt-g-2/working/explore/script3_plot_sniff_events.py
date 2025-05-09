# script3_plot_sniff_events.py
# Objective: Load inhalation and exhalation event times and plot them on a timeline.

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

# Output file path for plot
event_plot_path = "explore/sniff_events.png"

print(f"Attempting to load NWB file from: {url}")
io = None  # Initialize io to None
try:
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file, 'r')
    io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
    nwb = io.read()
    print("NWB File Loaded Successfully.")

    inhalation_times = None
    exhalation_times = None

    if "behavior" in nwb.processing:
        behavior_module = nwb.processing["behavior"]
        if "inhalation_time" in behavior_module.data_interfaces:
            inhalation_time_series = behavior_module.data_interfaces["inhalation_time"]
            # Assuming timestamps are sample indices or ms, convert to seconds by dividing by 1000.0 (based on LFP/SniffSignal rate)
            inhalation_times = inhalation_time_series.timestamps[:] / 1000.0
            print(f"Loaded and converted {len(inhalation_times)} inhalation events.")
        else:
            print("Inhalation time data not found.")
            inhalation_times = np.array([]) # Ensure it's an empty array if not found

        if "exhalation_time" in behavior_module.data_interfaces:
            exhalation_time_series = behavior_module.data_interfaces["exhalation_time"]
            # Assuming timestamps are sample indices or ms, convert to seconds by dividing by 1000.0
            exhalation_times = exhalation_time_series.timestamps[:] / 1000.0
            print(f"Loaded and converted {len(exhalation_times)} exhalation events.")
        else:
            print("Exhalation time data not found.")
            exhalation_times = np.array([]) # Ensure it's an empty array if not found
    else:
        print("Behavior processing module not found.")
        inhalation_times = np.array([])
        exhalation_times = np.array([])

    if len(inhalation_times) > 0 or len(exhalation_times) > 0: # Check if any data was loaded
        # Select a segment of time to plot, e.g., up to 5 seconds
        max_time_to_plot = 5  # seconds
        
        # Print first few timestamps for debugging (after conversion)
        print(f"First 5 inhalation_times (s): {inhalation_times[:5] if len(inhalation_times) > 0 else 'No inhalation events'}")
        print(f"First 5 exhalation_times (s): {exhalation_times[:5] if len(exhalation_times) > 0 else 'No exhalation events'}")

        inhalation_times_segment = inhalation_times[inhalation_times <= max_time_to_plot]
        exhalation_times_segment = exhalation_times[exhalation_times <= max_time_to_plot]

        if len(inhalation_times_segment) > 0 or len(exhalation_times_segment) > 0:
            plt.figure(figsize=(15, 4))
            
            plot_event_data = []
            plot_colors = []
            plot_lineoffsets = []
            plot_ylabels = []
            plot_yticks = []

            if len(inhalation_times_segment) > 0:
                plot_event_data.append(inhalation_times_segment)
                plot_colors.append('blue')
                plot_lineoffsets.append(1)
                plot_ylabels.append('Inhalation')
                plot_yticks.append(1)
            
            if len(exhalation_times_segment) > 0:
                plot_event_data.append(exhalation_times_segment)
                plot_colors.append('red')
                plot_lineoffsets.append(-1)
                plot_ylabels.append('Exhalation')
                plot_yticks.append(-1)

            if plot_event_data: # Check if there's anything to plot
                plt.eventplot(plot_event_data, colors=plot_colors, lineoffsets=plot_lineoffsets, linelengths=0.8)
                plt.yticks(plot_yticks, plot_ylabels)
                plt.title(f'Sniff Events (First {max_time_to_plot}s)')
                plt.xlabel('Time (s)')
                # Determine x-axis limits dynamically or stick to max_time_to_plot
                # x_max = max_time_to_plot
                # if len(inhalation_times_segment) > 0: x_max = max(x_max, np.max(inhalation_times_segment))
                # if len(exhalation_times_segment) > 0: x_max = max(x_max, np.max(exhalation_times_segment))
                plt.xlim(0, max_time_to_plot + 1) # Add a bit of padding
                plt.grid(True, axis='x') # Grid only on x-axis for clarity
                plt.savefig(event_plot_path)
                plt.close()
                print(f"Sniff events plot saved to {event_plot_path}")
            else:
                print("No sniff events to plot in the selected time segment.")
        else:
            print("No sniff events found within the first {max_time_to_plot} seconds.")
    else:
        print("Could not plot sniff events due to missing data.")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    if io is not None:
        io.close()
    print("\nScript finished.")