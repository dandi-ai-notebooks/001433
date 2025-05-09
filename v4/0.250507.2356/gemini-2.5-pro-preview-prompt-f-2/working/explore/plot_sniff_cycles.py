# This script loads inhalation and exhalation event times and plots the distribution of cycle durations.
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

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=False) # sharex might not be appropriate for different distributions

plot_created = False
if "behavior" in nwb.processing:
    behavior = nwb.processing["behavior"]
    if "inhalation_time" in behavior.data_interfaces:
        inhalation_events = behavior.data_interfaces["inhalation_time"]
        inhalation_timestamps = inhalation_events.timestamps[:]
        if len(inhalation_timestamps) > 1:
            inhalation_durations = np.diff(inhalation_timestamps)
            axs[0].hist(inhalation_durations, bins=50, color='skyblue', edgecolor='black')
            axs[0].set_title('Distribution of Inhalation Cycle Durations')
            axs[0].set_xlabel('Duration (s)')
            axs[0].set_ylabel('Count')
            plot_created = True
        else:
            axs[0].text(0.5, 0.5, 'Not enough inhalation data to plot durations', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
            axs[0].set_title('Inhalation Cycle Durations')
    else:
        axs[0].text(0.5, 0.5, 'Inhalation time data not found', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
        axs[0].set_title('Inhalation Cycle Durations')

    if "exhalation_time" in behavior.data_interfaces:
        exhalation_events = behavior.data_interfaces["exhalation_time"]
        exhalation_timestamps = exhalation_events.timestamps[:]
        if len(exhalation_timestamps) > 1:
            exhalation_durations = np.diff(exhalation_timestamps)
            axs[1].hist(exhalation_durations, bins=50, color='salmon', edgecolor='black')
            axs[1].set_title('Distribution of Exhalation Cycle Durations')
            axs[1].set_xlabel('Duration (s)')
            axs[1].set_ylabel('Count')
            plot_created = True
        else:
            axs[1].text(0.5, 0.5, 'Not enough exhalation data to plot durations', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
            axs[1].set_title('Exhalation Cycle Durations')

    else:
        axs[1].text(0.5, 0.5, 'Exhalation time data not found', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        axs[1].set_title('Exhalation Cycle Durations')
else:
    axs[0].text(0.5, 0.5, 'Behavior processing module not found', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
    axs[0].set_title('Inhalation Cycle Durations')
    axs[1].text(0.5, 0.5, 'Behavior processing module not found', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
    axs[1].set_title('Exhalation Cycle Durations')


if plot_created:
    plt.tight_layout()
    plt.savefig('explore/sniff_cycle_durations.png')
    print("Plot saved to explore/sniff_cycle_durations.png")
else:
    # If no actual plot was made (e.g. data missing), save a placeholder or skip saving
    print("No plot generated due to missing data or insufficient data points.")
    # Optionally, create a blank figure with text indicating no data
    fig_blank, ax_blank = plt.subplots()
    ax_blank.text(0.5, 0.5, "Data for sniff cycle durations not available or insufficient.", horizontalalignment='center', verticalalignment='center')
    ax_blank.set_xticks([])
    ax_blank.set_yticks([])
    plt.savefig('explore/sniff_cycle_durations.png') # save a blank plot to avoid read_image error
    print("Blank plot saved to explore/sniff_cycle_durations.png")


io.close()