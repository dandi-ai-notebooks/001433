# Plot the first 5 seconds of LFP (all channels) to visualize overall activity
# Output: explore/lfp_segment.png

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

    LFP = nwb.acquisition['LFP']
    sample_rate = int(LFP.rate)
    segment_samples = 5 * sample_rate  # 5 seconds
    data = LFP.data[:segment_samples, :]  # shape (5000, 16)

    plt.figure(figsize=(10, 8))
    for ch in range(data.shape[1]):
        plt.plot(np.arange(segment_samples) / sample_rate, data[:, ch] + ch * 200, label=f'Ch {ch}')
    plt.xlabel('Time (s)')
    plt.yticks([])
    plt.title('LFP: first 5 seconds, all 16 channels (vertically offset)')
    plt.tight_layout()
    plt.savefig('explore/lfp_segment.png')
    plt.close()
    io.close()
    h5_file.close()
    remote_file.close()

if __name__ == "__main__":
    main()