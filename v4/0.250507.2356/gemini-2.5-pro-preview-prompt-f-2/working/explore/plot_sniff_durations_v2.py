# This script loads inhalation and exhalation data and plots their distributions.
# It assumes the .data attribute of the TimeSeries contains the durations.
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
x_axis_limit_seconds = 2.0 # Max duration to display on x-axis for clarity

if "behavior" in nwb.processing:
    behavior = nwb.processing["behavior"]
    # Plot Inhalation Durations
    if "inhalation_time" in behavior.data_interfaces:
        inhalation_series = behavior.data_interfaces["inhalation_time"]
        inhalation_durations_data = inhalation_series.data[:]
        # Assuming data are durations in seconds as per description "inhalation_time (s)"
        # Filter out non-positive durations
        inhalation_durations_positive = inhalation_durations_data[inhalation_durations_data > 0]
        
        if len(inhalation_durations_positive) > 0:
            axs[0].hist(inhalation_durations_positive, bins=50, color='skyblue', edgecolor='black', range=(0, x_axis_limit_seconds))
            axs[0].set_title('Distribution of Inhalation Durations (from .data attribute)')
            axs[0].set_xlabel('Duration (s)')
            axs[0].set_ylabel('Count')
            axs[0].set_xlim(0, x_axis_limit_seconds)
            plot_created = True
            print(f"Inhalation durations: min={np.min(inhalation_durations_positive)}, max={np.max(inhalation_durations_positive)}, mean={np.mean(inhalation_durations_positive)}")
        else:
            axs[0].text(0.5, 0.5, 'No positive inhalation duration data found in .data', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
            axs[0].set_title('Inhalation Durations')
    else:
        axs[0].text(0.5, 0.5, 'Inhalation time data not found', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
        axs[0].set_title('Inhalation Durations')

    # Plot Exhalation Durations
    if "exhalation_time" in behavior.data_interfaces:
        exhalation_series = behavior.data_interfaces["exhalation_time"]
        exhalation_durations_data = exhalation_series.data[:]
        # Assuming data are durations in seconds
        # Filter out non-positive durations
        exhalation_durations_positive = exhalation_durations_data[exhalation_durations_data > 0]

        if len(exhalation_durations_positive) > 0:
            axs[1].hist(exhalation_durations_positive, bins=50, color='salmon', edgecolor='black', range=(0, x_axis_limit_seconds))
            axs[1].set_title('Distribution of Exhalation Durations (from .data attribute)')
            axs[1].set_xlabel('Duration (s)')
            axs[1].set_ylabel('Count')
            axs[1].set_xlim(0, x_axis_limit_seconds)
            plot_created = True
            print(f"Exhalation durations: min={np.min(exhalation_durations_positive)}, max={np.max(exhalation_durations_positive)}, mean={np.mean(exhalation_durations_positive)}")
        else:
            axs[1].text(0.5, 0.5, 'No positive exhalation duration data found in .data', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
            axs[1].set_title('Exhalation Durations')
    else:
        axs[1].text(0.5, 0.5, 'Exhalation time data not found', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        axs[1].set_title('Exhalation Durations')
else:
    axs[0].text(0.5, 0.5, 'Behavior processing module not found', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
    axs[0].set_title('Inhalation Durations')
    axs[1].text(0.5, 0.5, 'Behavior processing module not found', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
    axs[1].set_title('Exhalation Durations')

if plot_created:
    plt.tight_layout()
    plt.savefig('explore/sniff_durations_v2.png')
    print("Plot saved to explore/sniff_durations_v2.png")
else:
    print("No plot generated due to missing data or insufficient data points.")
    fig_blank, ax_blank = plt.subplots()
    ax_blank.text(0.5, 0.5, "Data for sniff durations (v2) not available or insufficient.", horizontalalignment='center', verticalalignment='center')
    ax_blank.set_xticks([])
    ax_blank.set_yticks([])
    plt.savefig('explore/sniff_durations_v2.png')
    print("Blank plot saved to explore/sniff_durations_v2.png")

io.close()