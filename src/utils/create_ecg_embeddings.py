import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.ecg_fm import ECGFM_Encoder
from src.classes.heedb_dataloader import HEEDBDataset
import os


device = "cuda" if torch.cuda.is_available() else "cpu"


data_dir = "./data/harvard-emory-dataset"
checkpoint_path = "./checkpoints/mimic_iv_ecg_physionet_pretrained.pt"
fairseq_path = "./fairseq-signals"


dataset = HEEDBDataset(data_dir, mode="waveform")

dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)

ecg_fm = ECGFM_Encoder(checkpoint_path, fairseq_path).to(device)
ecg_fm.eval()

all_embeddings = []

with torch.no_grad():
    for batch in tqdm(dataloader):
        waveform = batch['ecg_waveform'].to(device)
        embedding = ecg_fm(waveform)
        all_embeddings.append(embedding.cpu().numpy())

all_embeddings = np.concatenate(all_embeddings, axis=0)
np.save(os.path.join(data_dir, 'heedb_embeddings.npy'), all_embeddings)
print(f"Embeddings shape: {all_embeddings.shape}")
