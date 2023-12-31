"""
Source: https://github.com/ezhang7423/eztorchtils/blob/main/eztorchtils/utils/model_wrappers.py
"""

from copy import deepcopy

import torch
from torch import nn


class EMA(nn.Module):
    """Model Exponential Moving Average V2 from timm"""

    def __init__(self, model, decay=0.9999):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(
            model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)