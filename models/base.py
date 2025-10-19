"""
Base classes for MOS prediction models.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchaudio

class BasePredictor(nn.Module):
    """Base class for all MOS predictors."""
    def __init__(self):
        super(BasePredictor, self).__init__()
    
    def forward(self, wavs, texts):
        raise NotImplementedError

class MyDataset(Dataset):
    """Dataset class for MOS prediction."""
    def __init__(self, wavdir, mos_list):
        self.mos_overall_lookup = { }
        self.mos_coherence_lookup = { }
        f = open(mos_list, 'r')
        for line in f:
            parts = line.strip().split(',')
            wavname = parts[0]  # 'audiomos2025-track1-S002_P044.wav'
            mos_overall = float(parts[1])
            mos_coherence = float(parts[2])
            self.mos_overall_lookup[wavname] = mos_overall
            self.mos_coherence_lookup[wavname] = mos_coherence

        self.wavdir = wavdir
        self.wavnames = sorted(self.mos_overall_lookup.keys())
    
    def __getitem__(self, idx):
        wavname = self.wavnames[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        if wav.size(1) > 480000:    # 16khz*30s
            wav = wav[:,:480000]
        overall_score = self.mos_overall_lookup[wavname]
        coherence_score = self.mos_coherence_lookup[wavname]
        return wav, overall_score, coherence_score, wavname
    
    def __len__(self):
        return len(self.wavnames)

    def collate_fn(self, batch):
        wavs, overall_scores, coherence_scores, wavnames = zip(*batch)
        wavs = list(wavs)
        max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)
        output_wavs = torch.stack(output_wavs, dim=0)
        overall_scores = torch.stack([torch.tensor(x) for x in list(overall_scores)], dim=0)
        coherence_scores = torch.stack([torch.tensor(x) for x in list(coherence_scores)], dim=0)
        
        return output_wavs, overall_scores, coherence_scores, wavnames 