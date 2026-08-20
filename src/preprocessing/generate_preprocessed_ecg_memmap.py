import numpy as np
import pandas as pd
import wfdb
import os
from tqdm import tqdm
import warnings

import ecg_preprocess

# Data Root
ROOT_DIR = "./data/harvard-emory-dataset"
OUTPUT_DIR = "./data/harvard-emory-dataset"
CSV_PATH = "./data/harvard-emory-dataset/hf_risk_i0001_and_i0006_prevalent_mi_hyp_additional_factors.csv"
NUM_CHANNELS = 12
TARGET_LENGTH = 5000
TARGET_SAMPLE_RATE = 500





def gen_memmap():
    # This suppresses the noisy internal warnings from numpy and neurokit
    warnings.filterwarnings("ignore")
    df = pd.read_csv(CSV_PATH)

    memmap = np.memmap(
        os.path.join(OUTPUT_DIR, "heedb_full_waveform.memmap"),
        dtype=np.float32,
        mode='w+',
        shape=(len(df), NUM_CHANNELS, TARGET_LENGTH)
    )
    failures = 0
    failure_log = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        relative_path = row['FileName']

        relative_path = relative_path.lstrip("/").lstrip("\\")

        if relative_path.startswith("WFDB/"):
            full_path = os.path.join(ROOT_DIR,'I0006', relative_path)

        else:
            full_path = os.path.join(ROOT_DIR, 'I0001/WFDB', relative_path)


        try:
            record = wfdb.rdrecord(full_path)
            signal = record.p_signal # -> (samples, 12)
            fs = record.fs
            signal = signal.T # -> (12, samples)


            # Resample
            signal = ecg_preprocess.resample(signal, fs, TARGET_SAMPLE_RATE)

            # Normalise using zscore
            normalised_signal = ecg_preprocess.normalise_ecg(signal, TARGET_SAMPLE_RATE)

            # Pad/truncate signal to ensure same length
            processed_signal = ecg_preprocess.pad_truncate_ecg(normalised_signal, TARGET_LENGTH)


            if processed_signal.shape == (12, TARGET_LENGTH):
                memmap[index] = processed_signal


        except Exception as e:
            failures += 1
            failure_log.append({'index': index, 'file': relative_path, 'error': str(e)})
            print(f"ERROR [{index}] {relative_path}: {e}")


    del memmap
    print("Memmap generation complete.")
    print(f"Failed: {failures}/{len(df)}")

    if failure_log:
        pd.DataFrame(failure_log).to_csv(os.path.join(OUTPUT_DIR, 'fullwave_memmap_failures.csv'), index=False)
        print(f"Failure log saved to fullwave_memmap_failures.csv")





if __name__ == "__main__":
    gen_memmap()



