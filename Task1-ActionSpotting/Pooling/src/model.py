
import __future__

import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from netvlad import NetVLAD


class Model(nn.Module):
    def __init__(self, weights=None, input_size=512, num_classes=3, chunk_size=240, framerate=2, pool="NetVLAD"):
        """
        INPUT: a Tensor of shape (batch_size,chunk_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """

        super(Model, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.framerate = framerate
        self.pool = pool

        if self.pool == "MAX":
            self.pool_layer = nn.MaxPool1d(chunk_size, stride=1)
            self.fc = nn.Linear(input_size, self.num_classes+1)

        elif self.pool == "NetVLAD":
            self.pool_layer = NetVLAD(num_clusters=64, dim=512,
                                      normalize_input=True, vladv2=False)
            self.fc = nn.Linear(input_size*64, self.num_classes+1)

        self.drop = nn.Dropout(p=0.4)
        self.sigm = nn.Sigmoid()

        self.load_weights(weights=weights)

    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))

    def forward(self, inputs):
        # input_shape: (batch,frames,dim_features)

        # Temporal pooling operation
        if self.pool == "MAX":
            inputs = inputs.permute((0, 2, 1))
            inputs_pooled = self.pool_layer(inputs)
            inputs_pooled = inputs_pooled.squeeze(-1)

        elif self.pool == "NetVLAD":
            inputs = inputs.unsqueeze(-1)
            inputs = inputs.permute((0, 2, 1, 3))
            inputs_pooled = self.pool_layer(inputs)

        # Extra FC layer with dropout and sigmoid activation
        output = self.sigm(self.fc(self.drop(inputs_pooled)))

        return output
