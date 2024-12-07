import config
from . import trainer_api
from dataset.dataset import SpecDataset
from models.classifier import SpecModel
from utils.utils import get_train_tranform

import torch
import pandas as pd
from torch.utils.data import DataLoader 


# load datasets and dataset loaders
df_train = pd.read_csv(config['TRAIN_CSV'])
df_valid = pd.read_csv(config['VALID_CSV'])


trainloader = DataLoader(
    SpecDataset(df_train, augs=get_train_tranform()), 
    batch_size=config['BATCH_SIZE'], 
    shuffle=True
)

validloader = DataLoader(
    SpecDataset(df_valid), 
    batch_size=config['BATCH_SIZE']
)

# load model
model = SpecModel()

# training
optimizer = torch.optim.Adam(model.parameters(), lr=config['LR'])
trainer_api.fit(model, trainloader, validloader, optimizer)