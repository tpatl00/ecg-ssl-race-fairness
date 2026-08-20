import os
import torch
from torch.utils.data import Dataset
import numpy as np



class HEEDBDataset(Dataset):
    def __init__(self, data_dir, mode, embedding_file=None, waveform_dtype="float16"):
        # mode: "embedding" | "waveform"
        # embedding_file: only used when mode == "embedding".
        #                 None -> default heedb_embeddings.npy
        print(f"Initialising dataset from {data_dir}...")
        self.data_dir = data_dir
        self.n_leads = 12
        self.ecg_output_length = 5000

        self.tabular_features = np.load(os.path.join(data_dir, 'heedb_tabular_features.npy'))
        self.labels = np.load(os.path.join(data_dir, 'heedb_labels.npy'))
        self.patient_ids = np.load(os.path.join(data_dir, 'heedb_patient_ids.npy'), allow_pickle=True)
        self.patient_ids = self.patient_ids.astype(np.int32)

        if mode == "embedding":
            if embedding_file is not None:
                print(f"Loading custom embedding file: {embedding_file}")
                self.ecg_data = np.load(embedding_file)
            else:
                self.ecg_data = np.load(os.path.join(data_dir, 'heedb_embeddings.npy'))

            print(f"ECG data shape: {self.ecg_data.shape}")
            print("Dataset initialisation complete.")


        if mode == "waveform":
            n_samples = len(self.labels)
            waveform_shape = (n_samples, self.n_leads, self.ecg_output_length)
            memmap_path = os.path.join(data_dir, 'heedb_full_waveform.memmap')

            print(f"Pre-loading {n_samples} waveforms as float16 (~{n_samples * 12 * 5000 * 2 / 1e9:.1f} GB)...")
            memmap_data = np.memmap(memmap_path, dtype=np.float32, mode='r', shape=waveform_shape)
            self.ecg_data = np.empty(waveform_shape, dtype=np.float16)
            chunk_size = 10000
            for i in range(0, n_samples, chunk_size):
                end = min(i + chunk_size, n_samples)
                self.ecg_data[i:end] = memmap_data[i:end]
                if (i // chunk_size) % 5 == 0:
                    print(f"Loaded {end}/{n_samples} samples...")
            del memmap_data  # release memmap
            print(
                f"Dataset initialization complete. "
                f"{len(self)} samples, waveform shape per sample: "
                f"({self.n_leads}, {self.ecg_output_length}), dtype=float16"
            )


    def __len__(self):
        return len(self.labels)



    def __getitem__(self, idx):
        ecg_waveform = self.ecg_data[idx]           # (12, 5000) from memmap
        tabular_vec = self.tabular_features[idx]    # (72,)
        labels = self.labels[idx]              # (2,) — [event, time]
        patient_id = self.patient_ids[idx]         # scalar

        sample = {
            'ecg_waveform': torch.tensor(ecg_waveform, dtype=torch.float32),
            'tabular': torch.tensor(tabular_vec, dtype=torch.float32),
            'eventStatus': torch.tensor(int(labels[0]), dtype=torch.long),
            'followUpTime': torch.tensor(labels[1], dtype=torch.float32),
            'patient_id': torch.tensor(patient_id, dtype=torch.long),
        }
        return sample