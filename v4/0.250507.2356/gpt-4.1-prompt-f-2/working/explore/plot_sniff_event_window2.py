# Plot a 5-second window of the sniff signal containing both inhalation and exhalation events, ideally at the start of the first inhalation event
# Output: explore/sniff_event_window2.png

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

def main():
    url = "https://api.dandiarchive.org/api/assets/63d19f03-2a35-48bd-a54f-9ab98ceb7be2/download/"
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()

    SniffSignal = nwb.acquisition['SniffSignal']
    sample_rate = int(SniffSignal.rate)
    n_samples = SniffSignal.data.shape[0]
    
    inh = nwb.processing['behavior'].data_interfaces['inhalation_time']
    ex = nwb.processing['behavior'].data_interfaces['exhalation_time']
    inh_t = inh.timestamps[:]
    ex_t = ex.timestamps[:]

    # Focus on window centered at first inhalation event (as from summary: ~58s)
    if len(inh_t) == 0:
        print("No inhalation events found.")
        return

    t0 = inh_t[0] - 2.5  # 2.5s before first inhale
    t0 = max(t0, 0)
    t1 = t0 + 5
    idx0 = int(t0 * sample_rate)
    idx1 = int(t1 * sample_rate)
    if idx1 > n_samples:
        idx1 = n_samples
        idx0 = max(0, idx1 - int(5 * sample_rate))

    times = np.arange(idx0, idx1) / sample_rate
    data = SniffSignal.data[idx0:idx1]

    # Select events in window
    inh_t_win = inh_t[(inh_t >= t0) & (inh_t < t1)]
    ex_t_win = ex_t[(ex_t >= t0) & (ex_t < t1)]

    plt.figure(figsize=(10, 3))
    plt.plot(times, data, label='Sniff signal')
    for t in inh_t_win:
        plt.axvline(t, color='b', alpha=0.7, label='Inhalation')
    for t in ex_t_win:
        plt.axvline(t, color='r', alpha=0.7, label='Exhalation')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.title("Sniff signal (5s window, first inhalation event)")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.tight_layout()
    plt.savefig('explore/sniff_event_window2.png')
    plt.close()
    io.close()
    h5_file.close()
    remote_file.close()

if __name__ == "__main__":
    main()