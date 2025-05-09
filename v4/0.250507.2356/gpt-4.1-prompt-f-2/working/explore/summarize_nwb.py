# Summarize the NWB file's main contents (for notebook planning)
# Outputs: text summary with dataset shapes, basic metadata fields, first few values for some fields
import pynwb
import h5py
import remfile

def main():
    url = "https://api.dandiarchive.org/api/assets/63d19f03-2a35-48bd-a54f-9ab98ceb7be2/download/"
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()

    print("Session:", nwb.session_description)
    print("Identifier:", nwb.identifier)
    print("Experimenter:", nwb.experimenter)
    print("Session start time:", nwb.session_start_time)
    print("File create date:", nwb.file_create_date)
    print("Experiment description:", getattr(nwb, 'experiment_description', ''))
    print("Lab:", getattr(nwb, 'lab', ''))
    print("Institution:", getattr(nwb, 'institution', ''))
    print("Subject info:")
    subject = nwb.subject
    for k in ['description', 'species', 'sex', 'age', 'subject_id']:
        print(f"  {k}: {getattr(subject, k, None)}")
    print("Keywords:", nwb.keywords[:])

    print("\nAcquisition keys:", list(nwb.acquisition.keys()))
    LFP = nwb.acquisition['LFP']
    print("LFP shape:", LFP.data.shape, "dtype:", LFP.data.dtype)
    print("LFP description:", LFP.description)
    print("LFP rate:", LFP.rate, "Hz")
    print("LFP start time:", LFP.starting_time)
    print("LFP data sample:", LFP.data[0:3, 0:3])

    SniffSignal = nwb.acquisition['SniffSignal']
    print("SniffSignal shape:", SniffSignal.data.shape, "dtype:", SniffSignal.data.dtype)
    print("SniffSignal description:", SniffSignal.description)
    print("SniffSignal rate:", SniffSignal.rate, "Hz")
    print("SniffSignal data sample:", SniffSignal.data[0:10])

    print("\nElectrodes DataFrame (first 5):")
    print(nwb.electrodes.to_dataframe().head())

    print("\nProcessing modules:", list(nwb.processing.keys()))
    behavior = nwb.processing['behavior']
    print("Behavior module description:", behavior.description)
    print("Behavior data_interfaces:", list(behavior.data_interfaces.keys()))

    inhalation_time = behavior.data_interfaces['inhalation_time']
    print("Inhalation_time.shape:", inhalation_time.data.shape)
    print("Inhalation_time first 5:", inhalation_time.data[:5], "timestamps:", inhalation_time.timestamps[:5])
    exhalation_time = behavior.data_interfaces['exhalation_time']
    print("Exhalation_time.shape:", exhalation_time.data.shape)
    print("Exhalation_time first 5:", exhalation_time.data[:5], "timestamps:", exhalation_time.timestamps[:5])

    io.close()
    h5_file.close()
    remote_file.close()

if __name__ == "__main__":
    main()