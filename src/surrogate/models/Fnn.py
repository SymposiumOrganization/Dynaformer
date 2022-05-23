import torch
from pytorch_lightning import LightningModule

import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from pytorch_lightning import LightningModule
from .BaseModel import BaseModel

import numpy as np
import pdb


class FNN(BaseModel):
    """Feedforward neural network baseline"""


    def __init__(self, number_of_layers=5,number_of_states=8,state_len = 50, hid_dim=1200, current_dim=1, lr=1e-3, requires_normalization= False, variable_current=False, cfg=None):
        super(FNN, self).__init__()
        """
        number_of_layers: number of layers in the network
        number_of_states: hidden dimension of the sensor points (usually 8)
        state_len: length of the state vector (usually 50)
        compress_states: whether to compress the state vector to one dimension before passing it to the main network
        current_dim: the dimension of the current vector (usually 1)
        using_current: whether to use the current vector as input (if False, current_dim is ignored)
        """
        
        #input_dim = 1 + current_dim + 1 
      
        
        # if using_current:
        #     current_dim = current_dim
        # else:
        #     current_dim = 0
        # self.state_encoder = nn.Linear(3,1)
        if variable_current:
            assert current_dim == 20
        assert  cfg.min_length == cfg.max_length, "Context length must the constant for fnn"
        initial_dim = cfg.min_length * 3 + current_dim + 1 #voltage/current context + current + query point

        # It consists of a number of fully connected layers with relu activation repeated number_of_layers times
        self.layers=[nn.Linear(initial_dim, hid_dim), nn.ReLU()]
        for _ in range(number_of_layers-1):
            self.layers+=[nn.Linear(hid_dim, hid_dim), nn.ReLU()]
        self.main_networks = nn.Sequential(*(self.layers + [nn.Linear(hid_dim, 1)]))
        self.cfg = cfg
        self.loss_func = torch.nn.MSELoss(reduction='mean') 
        self.lr = lr
        self.variable_current=variable_current
        #self.using_current=using_current
        self.save_hyperparameters()

    def forward(self, context, current, query_point):
        x_func = context  #input states on consistent support points (N * state_len)
        x_loc_1 = query_point # query point [1]
        x_loc_2 = current # current
        #if self.compress_states:
        #x_func = self.state_encoder(x_func)
        x_func_flat = x_func.flatten(start_dim=1)

        #if self.using_current:
        x_loc = torch.cat([x_loc_1,x_loc_2],axis=1)
        # else:
        #     x_loc=x_loc_1

        # Concatenate everything and feed it to the main network
        x = torch.cat([x_func_flat,x_loc],axis=1)
        x = self.main_networks(x)
        return x
       