# This script loads inhalation and exhalation event times and plots the distribution of cycle durations.
# Durations are calculated as np.diff(timestamps).
# Non-positive durations are filtered out, and x-axis range is fixed for clarity.
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Apply seaborn styling
sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/63d19f03-2a35-48bd-a54f-9ab98ceb7be2/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
plot_created = False
# Define a practical upper limit for typical sniff cycle durations (e.g., 1 second)
# This helps in creating a reasonable histogram range.
x_axis_hist_range = (0, 1.0) # Plot durations from 0 to 1.0 seconds

if "behavior" in nwb.processing:
    behavior = nwb.processing["behavior"]
    
    # Plot Inhalation Cycle Durations
    if "inhalation_time" in behavior.data_interfaces:
        inhalation_events = behavior.data_interfaces["inhalation_time"]
        inhalation_timestamps = inhalation_events.timestamps[:]
        if len(inhalation_timestamps) > 1:
            inhalation_durations = np.diff(inhalation_timestamps)
            positive_inhalation_durations = inhalation_durations[inhalation_durations > 0]
            if len(positive_inhalation_durations) > 0:
                axs[0].hist(positive_inhalation_durations, bins=50, color='skyblue', edgecolor='black', range=x_axis_hist_range)
                axs[0].set_title('Distribution of Inhalation Cycle Durations (Calculated from Timestamps)')
                axs[0].set_xlabel('Duration (s)')
                axs[0].set_ylabel('Count')
                # axs[0].set_xlim(x_axis_hist_range) # xlim applied by hist range
                plot_created = True
                print(f"Positive inhalation durations (from timestamps): min={np.min(positive_inhalation_durations):.4f}, max={np.max(positive_inhalation_durations):.4f}, mean={np.mean(positive_inhalation_durations):.4f}")
            else:
                axs[0].text(0.5, 0.5, 'No positive inhalation durations found after diff(timestamps)', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
        else:
            axs[0].text(0.5, 0.5, 'Not enough inhalation timestamps for durations', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
        axs[0].set_title('Inhalation Cycle Durations (Calculated from Timestamps)')
    else:
        axs[0].text(0.5, 0.5, 'Inhalation time data not found', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
        axs[0].set_title('Inhalation Cycle Durations (Calculated from Timestamps)')

    # Plot Exhalation Cycle Durations
    if "exhalation_time" in behavior.data_interfaces:
        exhalation_events = behavior.data_interfaces["exhalation_time"]
        exhalation_timestamps = exhalation_events.timestamps[:]
        if len(exhalation_timestamps) > 1:
            exhalation_durations = np.diff(exhalation_timestamps)
            positive_exhalation_durations = exhalation_durations[exhalation_durations > 0]
            if len(positive_exhalation_durations) > 0:
                axs[1].hist(positive_exhalation_durations, bins=50, color='salmon', edgecolor='black', range=x_axis_hist_range)
                axs[1].set_title('Distribution of Exhalation Cycle Durations (Calculated from Timestamps)')
                axs[1].set_xlabel('Duration (s)')
                axs[1].set_ylabel('Count')
                # axs[1].set_xlim(x_axis_hist_range) # xlim applied by hist range
                plot_created = True
                print(f"Positive exhalation durations (from timestamps): min={np.min(positive_exhalation_durations):.4f}, max={np.max(positive_exhalation_durations):.4f}, mean={np.mean(positive_exhalation_durations):.4f}")
            else:
                axs[1].text(0.5, 0.5, 'No positive exhalation durations found after diff(timestamps)', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        else:
            axs[1].text(0.5, 0.5, 'Not enough exhalation timestamps for durations', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        axs[1].set_title('Exhalation Cycle Durations (Calculated from Timestamps)')
    else:
        axs[1].text(0.5, 0.5, 'Exhalation time data not found', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        axs[1].set_title('Exhalation Cycle Durations (Calculated from Timestamps)')
else:
    axs[0].text(0.5, 0.5, 'Behavior processing module not found', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
    axs[0].set_title('Inhalation Cycle Durations')
    axs[1].text(0.5, 0.5, 'Behavior processing module not found', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
    axs[1].set_title('Exhalation Cycle Durations')

if plot_created:
    plt.tight_layout()
    plt.savefig('explore/sniff_cycle_durations_v3.png')
    print("Plot saved to explore/sniff_cycle_durations_v3.png")
else:
    print("No plot generated due to missing data or insufficient data points for v3.")
    fig_blank, ax_blank = plt.subplots()
    ax_blank.text(0.5, 0.5, "Data for sniff cycle durations (v3) not available or insufficient.", horizontalalignment='center', verticalalignment='center')
    ax_blank.set_xticks([])
    ax_blank.set_yticks([])
    plt.savefig('explore/sniff_cycle_durations_v3.png')
    print("Blank plot saved to explore/sniff_cycle_durations_v3.png")

io.close()