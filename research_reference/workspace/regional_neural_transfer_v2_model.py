"""Additional held-value/age stem; all legacy neural files stay immutable."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from multikernel_backbone_model_v1 import MultikernelBackbone, WIDTH, HEADS, CHANNELS
from channel_bank400_v1 import ExpandedTCN
from run_raw_tcn_all_v1 import TCN
from fair_peer_controls_v2 import Recurrent


class HeldAgeMultikernelBackbone(MultikernelBackbone):
    """Only the input projection changes: 22x5 ->88, versus legacy22x3 ->88.

    Slots: held_z, fresh, log1p(age_in_actual_minutes), ever, installed.
    Uninstalled/never-observed held values and ages are zero. The fresh bit is
    never manufactured by carrying a last value forward.
    """
    def __init__(self):
        super().__init__('MK357')
        self.sensor_mix=nn.Linear(CHANNELS*5,WIDTH)

    def encode(self,x):
        if x.ndim!=4 or x.shape[2:]!=(CHANNELS,5) or x.size(1)==0:
            raise ValueError('Expected [batch,minute,22,5]')
        installed=x[...,4]>0
        ever=installed&(x[...,3]>0)
        fresh=ever&(x[...,1]>0)
        held=torch.where(ever,x[...,0],0.)
        age=torch.where(ever,x[...,2],0.)
        slots=torch.stack((held,fresh.to(x.dtype),age,ever.to(x.dtype),installed.to(x.dtype)),dim=-1)
        features=self.sensor_mix(slots.flatten(2)).transpose(1,2)
        branches=features.split([size for _,size in self.groups],dim=1)
        local=torch.cat([conv(F.pad(branch,(kernel-1,0)))
                         for branch,(kernel,_),conv in zip(branches,self.groups,self.convs)],dim=1)
        tokens=self.token_norm(F.silu(local.transpose(1,2)))
        batch,steps=tokens.shape[:2];head_size=WIDTH//HEADS
        qkv=self.qkv(tokens).reshape(batch,steps,3,HEADS,head_size)
        query,key,value=qkv.permute(2,0,3,1,4).unbind(0)
        scores=(query@key.transpose(-1,-2))/math.sqrt(head_size)
        minute=torch.arange(steps,device=x.device);lag=minute[:,None]-minute[None,:]
        log_lag=torch.log1p(lag.clamp_min(0).to(tokens.dtype))
        scores=scores+self.log_lag_slope[:,None,None]*log_lag
        attention=scores.masked_fill(lag<0,-torch.inf).softmax(dim=-1)
        mixed=(attention@value).transpose(1,2).reshape(batch,steps,WIDTH)
        attended=self.attention_norm(tokens+self.dropout(self.output_projection(mixed)))
        return self.output_norm(attended+self.dropout(self.feed_forward(attended)))


def input_kind(setting):
    if setting.startswith('MSCA_HeldAge'):return 'HeldAge'
    if setting.startswith('MSCA_'):return 'Fresh'
    if setting.startswith('HE_'):return 'HE'
    if setting in ('TCN_Native','GRU_Native'):return 'Raw4'
    raise ValueError(setting)


def build_setting(setting):
    kind=input_kind(setting)
    if kind=='HeldAge':return HeldAgeMultikernelBackbone()
    if kind=='Fresh':return MultikernelBackbone('MK357')
    if kind=='HE':return ExpandedTCN(5,'Direct400')
    if setting=='TCN_Native':return TCN(20,64)
    if setting=='GRU_Native':return Recurrent('GRU',20)
    raise ValueError(setting)


def structure(setting,model):
    result={'setting':setting,'parameters':sum(p.numel() for p in model.parameters()),'input_kind':input_kind(setting)}
    if setting.startswith('MSCA'):
        result.update(width=model.width,kernels=[list(x) for x in model.groups],
                      stem_input=model.sensor_mix.in_features,heads=HEADS,
                      causal_attention=True,per_time_layernorm=True)
    return result
