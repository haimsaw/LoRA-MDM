import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from utils.misc import wrapped_getattr

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'


    def forward(self, x, timesteps, y=None):
        cond_mode = self.model.cond_mode
        assert cond_mode in ['text', 'action']
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        out = self.model(x, timesteps, y)
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))

    
    def __getattr__(self, name, default=None):
        # this method is reached only if name is not in self.__dict__.
        return wrapped_getattr(self, name, default=None)
