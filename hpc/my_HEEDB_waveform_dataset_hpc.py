"""
Waveform dataset for the CNN baseline.

Loads raw 12-lead ECG waveforms from a memory-mapped file
(heedb_full_waveform.memmap) instead of pre-computed embeddings.

Returns the same batch dict keys as HEEDBDataset so the
existing training harness works without modification:
    'ecg_waveform'  -> (12, 5000)  float32  [raw waveform, NOT embedding]
    'tabular'       -> (72,)       float32
    'eventStatus'   -> scalar      long
    'followUpTime'  -> scalar      float32
    'patient_id'    -> scalar      long
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np


class HEEDBWaveformDataset(Dataset):
    """
    Dataset backed by a numpy memmap for memory-efficient access to
    187,694 x 12 x 5000 float32 ECG waveforms (~45 GB on disk).
    """

    def __init__(self, data_dir):
        print(f"Initializing waveform dataset from {data_dir}...")
        self.data_dir = data_dir
        self.n_leads = 12
        self.ecg_output_length = 5000

        # Labels and metadata (small enough to fit in RAM)
        self.tabular_features = np.load(os.path.join(data_dir, 'heedb_tabular_features.npy'))
        self.labels = np.load(os.path.join(data_dir, 'heedb_labels.npy'))
        self.patient_ids = np.load(
            os.path.join(data_dir, 'heedb_patient_ids.npy'), allow_pickle=True
        )
        self.patient_ids = self.patient_ids.astype(np.int32)

        # Pre-load all waveforms into RAM as float32 to avoid any precision loss.
        # 187694 x 12 x 5000 x float32 = ~45 GB — requires >=64 GB RAM allocation.
        # Loaded in chunks to keep peak memory under control.
        n_samples = len(self.labels)
        waveform_shape = (n_samples, self.n_leads, self.ecg_output_length)
        memmap_path = os.path.join(data_dir, 'heedb_full_waveform.memmap')

        print(f"Pre-loading {n_samples} waveforms as float32 (~{n_samples * 12 * 5000 * 4 / 1e9:.1f} GB)...")
        memmap_data = np.memmap(memmap_path, dtype=np.float32, mode='r', shape=waveform_shape)
        self.ecg_data = np.empty(waveform_shape, dtype=np.float32)

        chunk_size = 10000
        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            self.ecg_data[i:end] = memmap_data[i:end]
            if (i // chunk_size) % 5 == 0:
                print(f"  Loaded {end}/{n_samples} samples...")
        del memmap_data  # release memmap

        print(
            f"Dataset initialization complete. "
            f"{len(self)} samples, waveform shape per sample: "
            f"({self.n_leads}, {self.ecg_output_length}), dtype=float32"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ecg_waveform = self.ecg_data[idx]           # (12, 5000) from memmap
        tabular_vec  = self.tabular_features[idx]    # (72,)
        labels       = self.labels[idx]              # (2,) — [event, time]
        patient_id   = self.patient_ids[idx]         # scalar

        sample = {
            'ecg_waveform': torch.tensor(ecg_waveform, dtype=torch.float32),
            'tabular':      torch.tensor(tabular_vec,   dtype=torch.float32),
            'eventStatus':  torch.tensor(int(labels[0]), dtype=torch.long),
            'followUpTime': torch.tensor(labels[1],     dtype=torch.float32),
            'patient_id':   torch.tensor(patient_id,    dtype=torch.long),
        }
        return sample
