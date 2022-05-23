import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from pytorch_lightning import LightningModule
from .BaseModel import BaseModel

import numpy as np
import pdb



class DeepONet(BaseModel):
    """DeepONet torch implementation according to https://www.nature.com/articles/s42256-021-00302-5.pdf."""

    def __init__(self, layer_sizes_branch=[2000,2000,2000,500],layer_sizes_trunk=[2,2000,2000,500],activation="relu",initial_layer_sizes=[3,1],using_current=True,lr=1e-3,variable_current=False,cfg=None):
        super(DeepONet, self).__init__()
        if isinstance(activation, dict):
            self.activation_branch = activation["branch"]
            self.activation_trunk = activation["trunk"]
        else:
            self.activation_branch = self.activation_trunk = activation

        self.initial_layer_sizes=initial_layer_sizes
        if self.initial_layer_sizes != None:
            self.initial=FNNBlock(initial_layer_sizes, None)
        self.branch = FNNBlock(layer_sizes_branch, self.activation_branch)
        self.trunk = FNNBlock(layer_sizes_trunk, self.activation_trunk)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.activation_trunk = torch.nn.ReLU()

        self.loss_func = torch.nn.MSELoss(reduction='mean')
        self.lr = lr
        self.using_current=using_current
        self.variable_current=variable_current
        self.cfg =cfg
        self.save_hyperparameters()

    def forward(self, context, current, query_point):
        x_func = context  #input states on consistent support points (N * state_len)
        x_loc_1 = query_point # query point [1]
        x_loc_2 = current # current   
        if self.using_current:
            x_loc = torch.cat([x_loc_1,x_loc_2],axis=1)
        else:
            x_loc=x_loc_1
        if self.initial_layer_sizes != None:
            x_func=self.initial(x_func).squeeze(dim=2)
        else:
            x_func=x_func.squeeze(dim=2)
        x_func = self.branch(x_func)
        x_loc = self.activation_trunk(self.trunk(x_loc))
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = torch.einsum("bi,bi->b", x_func, x_loc)#/np.sqrt(x_func.shape[1])  #### normalization
        x=x.unsqueeze(1)
        x=x+self.bias
        return x

    



class FNNBlock(LightningModule):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes, activation):
        super(FNNBlock, self).__init__()
        self.inputs = None
        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'swish':
            self.activation = torch.nn.SiLU()
        elif activation == None:
            self.activation=None
        else:
            raise NotImplementedError            
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

    def forward(self, x_input):
        x = x_input
        for linear in self.linears[:-1]:
            if self.activation==None:
                x = linear(x)
            else:
                x = self.activation(linear(x))
        x = self.linears[-1](x)
        return x


# if __name__ == "__main__":
#     net = FNNBlock([5,5,5],'relu')



if __name__ == "__main__":
    ss=np.load('data/states_te.npy')
    ts=np.load('data/times_te.npy')
    net = DeepONet([200,5,9],[1,5,9],'relu',[8,5,1])
    x=(torch.Tensor(ss[:1,:,:]), torch.Tensor(ts[:1]))
    print(net(x))
    print(net(x).shape)


