
import __future__

import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import EdgeConv, DynamicEdgeConv
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.nn import global_max_pool


class ContextAwareModel(nn.Module):
    def __init__(self, num_classes=3, args=None):
        """
        INPUT: a Tensor of the form (batch_size,1,chunk_size,input_size)
        OUTPUTS:    1. The segmentation of the form (batch_size,chunk_size,num_classes)
                    2. The action spotting of the form (batch_size,num_detections,2+num_classes)
        """

        super(ContextAwareModel, self).__init__()

        self.args = args
        self.load_weights(weights=args.load_weights)

        self.input_size = args.num_features
        self.num_classes = num_classes
        self.dim_capsule = args.dim_capsule
        self.receptive_field = args.receptive_field*args.framerate
        self.num_detections = args.num_detections
        self.chunk_size = args.chunk_size*args.framerate
        self.framerate = args.framerate
        self.representation_width = args.dim_representation_w
        self.representation_height = args.dim_representation_h
        self.representation_channels = args.dim_representation_c

        self.pyramid_size_1 = int(np.ceil(self.receptive_field/7))
        self.pyramid_size_2 = int(np.ceil(self.receptive_field/3))
        self.pyramid_size_3 = int(np.ceil(self.receptive_field/2))
        self.pyramid_size_4 = int(np.ceil(self.receptive_field))
        self.pyramide_pool_size_h = int((((self.representation_height-4)/4)-4)/2)
        self.pyramide_pool_size_w = int((((self.representation_width-4)/4)-4)/2)

        # -------------------------------
        # Initialize the feature backbone
        # -------------------------------

        if args.backbone_feature is not None:
            if args.backbone_feature == "2DConv" and args.backbone_player is not None:
                self.init_2DConv(multiplier=1*self.args.feature_multiplier)
            elif args.backbone_feature == "2DConv" and args.backbone_player is None:
                self.init_2DConv(multiplier=2*self.args.feature_multiplier)

        # -------------------------------
        # Initialize the player backbone
        # -------------------------------

        if args.backbone_player is not None:
            if args.backbone_player == "3DConv" and args.backbone_feature is not None:
                self.init_3DConv(multiplier=1*self.args.feature_multiplier)
            elif args.backbone_player == "3DConv" and args.backbone_feature is None:
                self.init_3DConv(multiplier=2*self.args.feature_multiplier)
            elif "GCN" in self.args.backbone_player and args.backbone_feature is not None:
                self.init_GCN(multiplier=1*self.args.feature_multiplier)
            elif "GCN" in self.args.backbone_player and args.backbone_feature is None:
                self.init_GCN(multiplier=2*self.args.feature_multiplier)

        # -------------------
        # Segmentation module
        # -------------------

        self.kernel_seg_size = 3
        self.pad_seg = nn.ZeroPad2d((0,0,(self.kernel_seg_size-1)//2, self.kernel_seg_size-1-(self.kernel_seg_size-1)//2))
        self.conv_seg = nn.Conv2d(in_channels=152*self.args.feature_multiplier, out_channels=self.dim_capsule*self.num_classes, kernel_size=(self.kernel_seg_size,1))
        self.batch_seg = nn.BatchNorm2d(num_features=self.chunk_size, momentum=0.01,eps=0.001) 


        # -------------------
        # detection module
        # -------------------       
        self.max_pool_spot = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))
        self.kernel_spot_size = 3
        self.pad_spot_1 = nn.ZeroPad2d((0,0,(self.kernel_spot_size-1)//2, self.kernel_spot_size-1-(self.kernel_spot_size-1)//2))
        self.conv_spot_1 = nn.Conv2d(in_channels=self.num_classes*(self.dim_capsule+1), out_channels=32, kernel_size=(self.kernel_spot_size,1))
        self.max_pool_spot_1 = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))
        self.pad_spot_2 = nn.ZeroPad2d((0,0,(self.kernel_spot_size-1)//2, self.kernel_spot_size-1-(self.kernel_spot_size-1)//2))
        self.conv_spot_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(self.kernel_spot_size,1))
        self.max_pool_spot_2 = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))

        # Confidence branch
        self.conv_conf = nn.Conv2d(in_channels=16*(self.chunk_size//8-1), out_channels=self.num_detections*2, kernel_size=(1,1))

        # Class branch
        self.conv_class = nn.Conv2d(in_channels=16*(self.chunk_size//8-1), out_channels=self.num_detections*self.num_classes, kernel_size=(1,1))
        self.softmax = nn.Softmax(dim=-1)


    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))

    def forward(self, inputs, representation_inputs):

        # -----------------------------------
        # Feature input (chunks of the video)
        # -----------------------------------
        # input_shape: (batch,channel,frames,dim_features)
        #print("Input size: ", inputs.size())

        
        concatenation = None
        r_concatenation = None

        if self.args.backbone_feature == "2DConv":
            concatenation = self.forward_2DConv(inputs)

        if self.args.backbone_player == "3DConv":
            r_concatenation = self.forward_3DConv(representation_inputs)
        elif "GCN" in self.args.backbone_player:
            r_concatenation = self.forward_GCN(inputs, representation_inputs)
        
        if r_concatenation is not None and concatenation is not None:
            if self.args.with_dropout == 0 or not self.training:
                full_concatenation = torch.cat((concatenation, r_concatenation), 1)
            elif self.args.with_dropout > 0 and self.training:
                random_number = torch.rand(1, device=concatenation.device)
                if (random_number < self.args.with_dropout/2)[0]:
                    concatenation = concatenation * torch.zeros(concatenation.shape, device=concatenation.device, dtype = torch.float )
                elif (random_number < self.args.with_dropout)[0]:
                    r_concatenation = r_concatenation * torch.zeros(r_concatenation.shape, device=r_concatenation.device, dtype = torch.float )
                full_concatenation = torch.cat((concatenation, r_concatenation), 1)


            # full_concatenation = torch.cat((concatenation, r_concatenation), 1)
        elif r_concatenation is None and concatenation is not None:
            full_concatenation = concatenation
        elif r_concatenation is not None and concatenation is None:
            full_concatenation = r_concatenation
        #print("full_concatenation size: ", full_concatenation.size())
        #print(torch.cuda.memory_allocated()/10**9)

        # -------------------
        # Segmentation module
        # -------------------

        conv_seg = self.conv_seg(self.pad_seg(full_concatenation))
        #print("Conv_seg size: ", conv_seg.size())
        #print(torch.cuda.memory_allocated()/10**9)

        conv_seg_permuted = conv_seg.permute(0,2,3,1)
        #print("Conv_seg_permuted size: ", conv_seg_permuted.size())
        #print(torch.cuda.memory_allocated()/10**9)

        conv_seg_reshaped = conv_seg_permuted.view(conv_seg_permuted.size()[0],conv_seg_permuted.size()[1],self.dim_capsule,self.num_classes)
        #print("Conv_seg_reshaped size: ", conv_seg_reshaped.size())
        #print(torch.cuda.memory_allocated()/10**9)


        #conv_seg_reshaped_permuted = conv_seg_reshaped.permute(0,3,1,2)
        #print("Conv_seg_reshaped_permuted size: ", conv_seg_reshaped_permuted.size())

        conv_seg_norm = torch.sigmoid(self.batch_seg(conv_seg_reshaped))
        #print("Conv_seg_norm: ", conv_seg_norm.size())
        #print(torch.cuda.memory_allocated()/10**9)


        #conv_seg_norm_permuted = conv_seg_norm.permute(0,2,3,1)
        #print("Conv_seg_norm_permuted size: ", conv_seg_norm_permuted.size())

        output_segmentation = torch.sqrt(torch.sum(torch.square(conv_seg_norm-0.5), dim=2)*4/self.dim_capsule)
        #print("Output_segmentation size: ", output_segmentation.size())
        #print(torch.cuda.memory_allocated()/10**9)


        # ---------------
        # Spotting module
        # ---------------

        # Concatenation of the segmentation score to the capsules
        output_segmentation_reverse = 1-output_segmentation
        #print("Output_segmentation_reverse size: ", output_segmentation_reverse.size())
        #print(torch.cuda.memory_allocated()/10**9)

        output_segmentation_reverse_reshaped = output_segmentation_reverse.unsqueeze(2)
        #print("Output_segmentation_reverse_reshaped size: ", output_segmentation_reverse_reshaped.size())
        #print(torch.cuda.memory_allocated()/10**9)


        output_segmentation_reverse_reshaped_permutted = output_segmentation_reverse_reshaped.permute(0,3,1,2)
        #print("Output_segmentation_reverse_reshaped_permutted size: ", output_segmentation_reverse_reshaped_permutted.size())
        #print(torch.cuda.memory_allocated()/10**9)

        concatenation_2 = torch.cat((conv_seg, output_segmentation_reverse_reshaped_permutted), dim=1)
        #print("Concatenation_2 size: ", concatenation_2.size())
        #print(torch.cuda.memory_allocated()/10**9)

        conv_spot = self.max_pool_spot(F.relu(concatenation_2))
        #print("Conv_spot size: ", conv_spot.size())
        #print(torch.cuda.memory_allocated()/10**9)

        conv_spot_1 = F.relu(self.conv_spot_1(self.pad_spot_1(conv_spot)))
        #print("Conv_spot_1 size: ", conv_spot_1.size())
        #print(torch.cuda.memory_allocated()/10**9)

        conv_spot_1_pooled = self.max_pool_spot_1(conv_spot_1)
        #print("Conv_spot_1_pooled size: ", conv_spot_1_pooled.size())
        #print(torch.cuda.memory_allocated()/10**9)

        conv_spot_2 = F.relu(self.conv_spot_2(self.pad_spot_2(conv_spot_1_pooled)))
        #print("Conv_spot_2 size: ", conv_spot_2.size())
        #print(torch.cuda.memory_allocated()/10**9)

        conv_spot_2_pooled = self.max_pool_spot_2(conv_spot_2)
        #print("Conv_spot_2_pooled size: ", conv_spot_2_pooled.size())
        #print(torch.cuda.memory_allocated()/10**9)

        spotting_reshaped = conv_spot_2_pooled.view(conv_spot_2_pooled.size()[0],-1,1,1)
        #print("Spotting_reshape size: ", spotting_reshaped.size())
        #print(torch.cuda.memory_allocated()/10**9)

        # Confindence branch
        conf_pred = torch.sigmoid(self.conv_conf(spotting_reshaped).view(spotting_reshaped.shape[0],self.num_detections,2))
        #print("Conf_pred size: ", conf_pred.size())
        #print(torch.cuda.memory_allocated()/10**9)

        # Class branch
        conf_class = self.softmax(self.conv_class(spotting_reshaped).view(spotting_reshaped.shape[0],self.num_detections,self.num_classes))
        #print("Conf_class size: ", conf_class.size())
        #print(torch.cuda.memory_allocated()/10**9)

        output_spotting = torch.cat((conf_pred,conf_class),dim=-1)
        #print("Output_spotting size: ", output_spotting.size())
        #print(torch.cuda.memory_allocated()/10**9)


        return output_segmentation, output_spotting
        


    def init_2DConv(self, multiplier=1):

        # Base Convolutional Layers
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=64*multiplier, kernel_size=(1,self.input_size))
        self.conv_2 = nn.Conv2d(in_channels=64*multiplier, out_channels=16*multiplier, kernel_size=(1,1))

        # Temporal Pyramidal Module
        self.pad_p_1 = nn.ZeroPad2d((0,0,(self.pyramid_size_1-1)//2, self.pyramid_size_1-1-(self.pyramid_size_1-1)//2))
        self.pad_p_2 = nn.ZeroPad2d((0,0,(self.pyramid_size_2-1)//2, self.pyramid_size_2-1-(self.pyramid_size_2-1)//2))
        self.pad_p_3 = nn.ZeroPad2d((0,0,(self.pyramid_size_3-1)//2, self.pyramid_size_3-1-(self.pyramid_size_3-1)//2))
        self.pad_p_4 = nn.ZeroPad2d((0,0,(self.pyramid_size_4-1)//2, self.pyramid_size_4-1-(self.pyramid_size_4-1)//2))
        self.conv_p_1 = nn.Conv2d(in_channels=16*multiplier, out_channels=4*multiplier, kernel_size=(self.pyramid_size_1,1))
        self.conv_p_2 = nn.Conv2d(in_channels=16*multiplier, out_channels=8*multiplier, kernel_size=(self.pyramid_size_2,1))
        self.conv_p_3 = nn.Conv2d(in_channels=16*multiplier, out_channels=16*multiplier, kernel_size=(self.pyramid_size_3,1))
        self.conv_p_4 = nn.Conv2d(in_channels=16*multiplier, out_channels=32*multiplier, kernel_size=(self.pyramid_size_4,1))

    def init_3DConv(self, multiplier=1):

        # ---------------------
        # Representation branch
        # ---------------------
        # Base convolutional layers
        self.r_conv_1 = nn.Conv3d(in_channels=self.representation_channels, out_channels=16, kernel_size=(1,5,5))
        self.r_max_pool_1 = nn.MaxPool3d(kernel_size=(1,4,4))
        self.r_conv_2 = nn.Conv3d(in_channels=16, out_channels=16*multiplier, kernel_size=(1,5,5))
        self.r_max_pool_2 = nn.MaxPool3d(kernel_size=(1,2,2))

        # Temporal pyramidal module
        self.r_pad_p_1 = (0,0,0,0,(self.pyramid_size_1-1)//2, self.pyramid_size_1-1-(self.pyramid_size_1-1)//2)
        self.r_pad_p_2 = (0,0,0,0,(self.pyramid_size_2-1)//2, self.pyramid_size_2-1-(self.pyramid_size_2-1)//2)
        self.r_pad_p_3 = (0,0,0,0,(self.pyramid_size_3-1)//2, self.pyramid_size_3-1-(self.pyramid_size_3-1)//2)
        self.r_pad_p_4 = (0,0,0,0,(self.pyramid_size_4-1)//2, self.pyramid_size_4-1-(self.pyramid_size_4-1)//2)
        self.r_conv_p_1 = nn.Conv3d(in_channels=16*multiplier, out_channels=4*multiplier, kernel_size=(self.pyramid_size_1,1,1))
        self.r_conv_p_2 = nn.Conv3d(in_channels=16*multiplier, out_channels=8*multiplier, kernel_size=(self.pyramid_size_2,1,1))
        self.r_conv_p_3 = nn.Conv3d(in_channels=16*multiplier, out_channels=16*multiplier, kernel_size=(self.pyramid_size_3,1,1))
        self.r_conv_p_4 = nn.Conv3d(in_channels=16*multiplier, out_channels=32*multiplier, kernel_size=(self.pyramid_size_4,1,1))
        self.r_maxpool_p_1 = nn.MaxPool3d(kernel_size=(1,self.pyramide_pool_size_h,self.pyramide_pool_size_w))
        self.r_maxpool_p_2 = nn.MaxPool3d(kernel_size=(1,self.pyramide_pool_size_h,self.pyramide_pool_size_w))
        self.r_maxpool_p_3 = nn.MaxPool3d(kernel_size=(1,self.pyramide_pool_size_h,self.pyramide_pool_size_w))
        self.r_maxpool_p_4 = nn.MaxPool3d(kernel_size=(1,self.pyramide_pool_size_h,self.pyramide_pool_size_w))
        self.r_maxpool_p_0 = nn.MaxPool3d(kernel_size=(1,self.pyramide_pool_size_h,self.pyramide_pool_size_w))

    def init_GCN(self,multiplier=1):

        # ---------------------
        # Representation branch
        # ---------------------
        # Base convolutional layers
        input_channel = 8
        if self.args.calibration_confidence:
            input_channel = 9
        # print("input_channel", input_channel)
        
            
        if self.args.backbone_player == "GCN":
            self.r_conv_1 = GCNConv(input_channel, 8*multiplier)
            self.r_conv_2 = GCNConv(8*multiplier, 16*multiplier)
            self.r_conv_3 = GCNConv(16*multiplier, 32*multiplier)
            self.r_conv_4 = GCNConv(32*multiplier, 76*multiplier)

        elif self.args.backbone_player == "EdgeConvGCN":
            self.r_conv_1 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2*input_channel, 8*multiplier) ]))
            self.r_conv_2 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2*8*multiplier, 16*multiplier) ]))
            self.r_conv_3 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2*16*multiplier, 32*multiplier) ]))
            self.r_conv_4 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2*32*multiplier, 76*multiplier) ]))

        elif self.args.backbone_player == "DynamicEdgeConvGCN":
            self.r_conv_1 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2*input_channel, 8*multiplier) ]), k=3)
            self.r_conv_2 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2*8*multiplier, 16*multiplier) ]), k=3)
            self.r_conv_3 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2*16*multiplier, 32*multiplier) ]), k=3)
            self.r_conv_4 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2*32*multiplier, 76*multiplier) ]), k=3)

        elif "resGCN" in self.args.backbone_player:
            # hidden_channels=64, num_layers=28
            # input_channel = 6
            output_channel = 76*multiplier
            hidden_channels = 64
            self.num_layers = int(self.args.backbone_player.split("-")[-1])

            self.node_encoder = nn.Linear(input_channel, hidden_channels)
            self.edge_encoder = nn.Linear(input_channel, hidden_channels)
            self.layers = torch.nn.ModuleList()
            for i in range(1, self.num_layers + 1):
                conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                            t=1.0, learn_t=True, num_layers=2, norm='layer')
                norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)
                act = nn.ReLU(inplace=True)

                layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1,
                                    ckpt_grad=i % 3)
                self.layers.append(layer)

            self.lin = nn.Linear(hidden_channels, output_channel)



    def forward_2DConv(self, inputs):

        # -------------------------------------
        # Temporal Convolutional neural network
        # -------------------------------------

        # Base Convolutional Layers
        conv_1 = F.relu(self.conv_1(inputs))
        #print("Conv_1 size: ", conv_1.size())
        #print(torch.cuda.memory_allocated()/10**9)
        
        conv_2 = F.relu(self.conv_2(conv_1))
        #print("Conv_2 size: ", conv_2.size())
        #print(torch.cuda.memory_allocated()/10**9)


        # Temporal Pyramidal Module
        conv_p_1 = F.relu(self.conv_p_1(self.pad_p_1(conv_2)))
        #print("Conv_p_1 size: ", conv_p_1.size())
        #print(torch.cuda.memory_allocated()/10**9)
        conv_p_2 = F.relu(self.conv_p_2(self.pad_p_2(conv_2)))
        #print("Conv_p_2 size: ", conv_p_2.size())
        #print(torch.cuda.memory_allocated()/10**9)
        conv_p_3 = F.relu(self.conv_p_3(self.pad_p_3(conv_2)))
        #print("Conv_p_3 size: ", conv_p_3.size())
        #print(torch.cuda.memory_allocated()/10**9)
        conv_p_4 = F.relu(self.conv_p_4(self.pad_p_4(conv_2)))
        #print("Conv_p_4 size: ", conv_p_4.size())
        #print(torch.cuda.memory_allocated()/10**9)

        concatenation = torch.cat((conv_2,conv_p_1,conv_p_2,conv_p_3,conv_p_4),1)
        #print("Concatenation size: ", concatenation.size())
        #print(torch.cuda.memory_allocated()/10**9)
        
        return concatenation

    def forward_3DConv(self,representation_inputs):

        # --------------------
        # Representation input
        # --------------------
        #print("Representation input size: ", representation_inputs.size())

        #Base convolutional Layers

        r_conv_1 = F.relu(self.r_conv_1(representation_inputs))
        #print("r_conv_1 size: ", r_conv_1.size())
        #print(torch.cuda.memory_allocated()/10**9)
        r_conv_1_pooled = self.r_max_pool_1(r_conv_1)
        #print("r_conv_1_pooled size: ", r_conv_1_pooled.size())
        #print(torch.cuda.memory_allocated()/10**9)

        r_conv_2 = F.relu(self.r_conv_2(r_conv_1_pooled))
        #print("r_conv_2 size: ", r_conv_2.size())
        #print(torch.cuda.memory_allocated()/10**9)
        r_conv_2_pooled = self.r_max_pool_2(r_conv_2)
        #print("r_conv_2_pooled size: ", r_conv_2_pooled.size())
        #print(torch.cuda.memory_allocated()/10**9)

        # Temporal Pyramidal Module
        r_conv_p_1 = self.r_maxpool_p_1(F.relu(self.r_conv_p_1(F.pad(r_conv_2_pooled, self.r_pad_p_1, "constant", 0)))).squeeze(-1)
        #print("r_conv_p_1 size: ", r_conv_p_1.size())
        #print(torch.cuda.memory_allocated()/10**9)
        r_conv_p_2 = self.r_maxpool_p_2(F.relu(self.r_conv_p_2(F.pad(r_conv_2_pooled, self.r_pad_p_2, "constant", 0)))).squeeze(-1)
        #print("r_conv_p_2 size: ", r_conv_p_2.size())
        #print(torch.cuda.memory_allocated()/10**9)
        r_conv_p_3 = self.r_maxpool_p_3(F.relu(self.r_conv_p_3(F.pad(r_conv_2_pooled, self.r_pad_p_3, "constant", 0)))).squeeze(-1)
        #print("r_conv_p_3 size: ", r_conv_p_3.size())
        #print(torch.cuda.memory_allocated()/10**9)
        r_conv_p_4 = self.r_maxpool_p_4(F.relu(self.r_conv_p_4(F.pad(r_conv_2_pooled, self.r_pad_p_4, "constant", 0)))).squeeze(-1)
        #print("r_conv_p_4 size: ", r_conv_p_4.size())
        #print(torch.cuda.memory_allocated()/10**9)
        r_conv_2_pooled_pooled = self.r_maxpool_p_4(r_conv_2_pooled).squeeze(-1)
        #print("r_conv_2_pooled_pooled size: ", r_conv_2_pooled_pooled.size())
        #print(torch.cuda.memory_allocated()/10**9)

        r_concatenation = torch.cat((r_conv_2_pooled_pooled,r_conv_p_1,r_conv_p_2,r_conv_p_3,r_conv_p_4),1)
        #print("r_concatenation size: ", r_concatenation.size())
        #print(torch.cuda.memory_allocated()/10**9)

        return r_concatenation

    def forward_GCN(self, inputs, representation_inputs):

        BS, _, T, C = inputs.shape

        # --------------------
        # Representation input -> GCN
        # --------------------
        #print("Representation input size: ", representation_inputs.size())

        #Base convolutional Layers
        x, edge_index, batch = representation_inputs.x, representation_inputs.edge_index, representation_inputs.batch
        edge_attr = representation_inputs.edge_attr
        # batch_list = batch.tolist()

        batch_unique = list(set(batch.tolist()))
        batch_max = max(batch.tolist())
        # print("batch", len(batch_unique), batch_max, len(batch.tolist()))
        # list1 = batch_unique
        # list2 = [i for i in range(max(batch.tolist()))]
        # print("additional", set(list1).difference(list2))
        # print("missing", set(list2).difference(list1))
        if self.args.backbone_player == "GCN" or self.args.backbone_player == "EdgeConvGCN":
            x = F.relu(self.r_conv_1(x, edge_index))
            x = F.relu(self.r_conv_2(x, edge_index))
            x = F.relu(self.r_conv_3(x, edge_index))
            x = F.relu(self.r_conv_4(x, edge_index))
        elif "DynamicEdgeConvGCN" in self.args.backbone_player: #EdgeConvGCN or DynamicEdgeConvGCN
            x = F.relu(self.r_conv_1(x, batch))
            x = F.relu(self.r_conv_2(x, batch))
            x = F.relu(self.r_conv_3(x, batch))
            x = F.relu(self.r_conv_4(x, batch))
        elif "resGCN" in self.args.backbone_player: #EdgeConvGCN or DynamicEdgeConvGCN
            x = self.node_encoder(x)
            # edge_attr = self.edge_encoder(edge_attr)

            # x = self.layers[0].conv(x, edge_index, edge_attr)
            x = self.layers[0].conv(x, edge_index)

            for layer in self.layers[1:]:
                # x = layer(x, edge_index, edge_attr)
                x = layer(x, edge_index)

            x = self.layers[0].act(self.layers[0].norm(x))
            x = F.dropout(x, p=0.1, training=self.training)

            x = self.lin(x)
        # print("before max_pool", x.shape)
        x = global_max_pool(x, batch) 
        # print("after max_pool", x.shape)
        # print(batch)
        # BS = inputs.shape[0]

        # magic fix with zero padding
        expected_size = BS* T
        x = torch.cat([x, torch.zeros(expected_size-x.shape[0], x.shape[1]).to(x.device)], 0)

        x = x.reshape(BS, T, x.shape[1]) #BSxTxFS
        x = x.permute((0,2,1)) #BSxFSxT
        x = x.unsqueeze(-1) #BSxFSxTx1
        r_concatenation = x

        return r_concatenation