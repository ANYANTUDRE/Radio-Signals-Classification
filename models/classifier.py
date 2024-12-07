import config

import torch
import timm


class SpecModel(nn.Module):
    def __init__(self):
        super(SpecModel, self).__init__()
        
        self.net = timm.create_model(config["MODEL_NAME"], num_classes = 4, pretrained = True, in_chans=1)
        
    def forward(self, images, labels = None):
        logits = self.net(images)
        
        if labels != None:
            loss = nn.CrossEntropyLoss()
            return logits, loss(logits, labels)
        
        return logits