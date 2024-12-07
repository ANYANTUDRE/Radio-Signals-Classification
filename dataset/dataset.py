import torch
from torch.utils.data import Dataset 
import numpy as np


class SpecDataset(Dataset):
    def __init__(self, df, augs = None):
        self.df = df
        self.augs = augs
        
        label_map = {
            'Squiggle': 0,
            'Narrowbanddrd': 1,
            'Noises': 2,
            'Narrowband': 3
        }
        self.df.loc[:, 'labels'] = self.df.labels.map(label_map)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_pixels = np.array(row[:8192], dtype=np.float64)
        
        image = np.resize(image_pixels, (64, 128, 1)) # (h, w, c)
        label = np.array(row.labels, dtype=np.int64)
        
        image = torch.Tensor(image).permute(2, 0, 1)# (c, h, w)
        
        if self.augs != None:
            image = self.augs(image)
            
        return image.float(), label