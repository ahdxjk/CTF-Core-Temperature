"""Direct future-temperature readouts for the existing two causal encoders."""
import torch
from torch import nn
from torch.nn import functional as F
from multikernel_backbone_model_v1 import MultikernelBackbone
from channel_bank400_v1 import ExpandedTCN

class DirectMSCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=MultikernelBackbone("MK357")
        self.encoder.readout=nn.Identity()
        self.readout=nn.Linear(90,1)
    def forward(self,x,activity):
        z=self.encoder.encode(x)
        return self.readout(torch.cat((z,activity),dim=-1)).squeeze(-1)

class DirectHE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=ExpandedTCN(14,"Direct400")
        self.encoder.backbone.out=nn.Identity()
        self.readout=nn.Linear(66,1)
    def forward(self,x,activity):
        base,extra=x[...,:4],x[...,4:]
        z=self.encoder.backbone.embed(base.flatten(2).transpose(1,2))
        z=z+self.encoder.extra(extra.flatten(2)).transpose(1,2)
        z=self.encoder.backbone.blocks(F.silu(z)).transpose(1,2)
        return self.readout(torch.cat((z,activity),dim=-1)).squeeze(-1)

def build(kind):
    return DirectMSCA() if kind=="MSCA" else DirectHE()

