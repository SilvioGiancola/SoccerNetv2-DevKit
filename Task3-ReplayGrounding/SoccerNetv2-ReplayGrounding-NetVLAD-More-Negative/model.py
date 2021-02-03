
import __future__

import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from netvlad import NetVLAD




class Model(nn.Module):
	def __init__(self, weights=None, input_size=512, chunk_size=240, dim_capsule=16, receptive_field=80, framerate=2, unsimilar_action=0.2, pooling="max" ,replay_size=40 ):
		"""
		INPUT: a Tensor of the form (batch_size,1,chunk_size,input_size)
		OUTPUTS:    1. The segmentation of the form (batch_size,chunk_size,num_classes)
					2. The action spotting of the form (batch_size,num_detections,2+num_classes)
		"""

		super(Model, self).__init__()

		self.load_weights(weights=weights)
		self.unsimilar_action=unsimilar_action
		self.pooling=pooling
		self.input_size = input_size
		self.num_classes = 1
		self.dim_capsule = dim_capsule
		self.receptive_field = receptive_field
		self.num_detections = 1
		self.chunk_size = chunk_size
		self.framerate = framerate
		self.replay_size=replay_size
		self.cluster=16
		# receptive_field2=10

		

		# NetVLAD
		self.netvald_pool = NetVLAD(num_clusters=self.cluster, dim=self.input_size,normalize_input=True, vladv2=False)

		# Cosine Similarity
		self.Cos=nn.CosineSimilarity(dim=1, eps=1e-6)

		# Confidence branch
		self.conv_pre_conf = nn.Conv2d(in_channels=self.cluster, out_channels=128, kernel_size=(self.input_size,1))
		self.conv_conf = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(1,1))

	def load_weights(self, weights=None):
		if(weights is not None):
			print("=> loading checkpoint '{}'".format(weights))
			checkpoint = torch.load(weights)
			self.load_state_dict(checkpoint['state_dict'],strict=False)
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(weights, checkpoint['epoch']))


	def forward(self, inputs,inputs_r,mask):
		
		out_x_pooled=self.netvald_pool(inputs.transpose(1,3).float())

		out_r_pooled=self.netvald_pool(inputs_r.transpose(1,3).float())
	
		sim_out=self.Cos(out_x_pooled,out_r_pooled)
		out=torch.zeros((sim_out.shape[0],1,2))
		out[:,:,0]=torch.max((sim_out.view(sim_out.shape[0],1).float()-self.unsimilar_action),torch.zeros((sim_out.shape[0],1)).cuda())

		pre_out=F.relu(self.conv_pre_conf(out_x_pooled.view(out_x_pooled.shape[0],self.cluster,self.input_size,1)))
		out[:,:,1]=torch.sigmoid(self.conv_conf(pre_out).view(sim_out.shape[0],1))
		return out

		
