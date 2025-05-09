# Plot the first 5 seconds of the sniff signal with inhalation/exhalation event markers
# Output: explore/sniff_segment.png

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
    t_total = SniffSignal.data.shape[0] / sample_rate
    segment_samples = 5 * sample_rate
    data = SniffSignal.data[:segment_samples]
    times = np.arange(segment_samples) / sample_rate

    # Get inhalation and exhalation event times within the first 5 seconds
    inh = nwb.processing['behavior'].data_interfaces['inhalation_time']
    ex = nwb.processing['behavior'].data_interfaces['exhalation_time']
    inh_t = inh.timestamps[:]
    ex_t = ex.timestamps[:]
    # Select those within 0-5 sec
    inh_t = inh_t[(inh_t >= 0) & (inh_t < 5)]
    ex_t = ex_t[(ex_t >= 0) & (ex_t < 5)]

    plt.figure(figsize=(10, 3))
    plt.plot(times, data, label='Sniff signal')
    plt.vlines(inh_t, ymin=np.min(data), ymax=np.max(data), color='b', alpha=0.5, label='Inhalation')
    plt.vlines(ex_t, ymin=np.min(data), ymax=np.max(data), color='r', alpha=0.5, label='Exhalation')
    plt.title("Sniff signal (first 5 sec) with inhalation/exhalation events")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('explore/sniff_segment.png')
    plt.close()
    io.close()
    h5_file.close()
    remote_file.close()

if __name__ == "__main__":
    main()