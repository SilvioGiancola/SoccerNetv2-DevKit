
import __future__

import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import EdgeConv
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
                if args.with_resnet > 0:
                    if args.with_dense:
                        self.init_Dense(multiplier=1*self.args.feature_multiplier)
                    else:
                        self.init_2DConv_copy(multiplier=1*self.args.feature_multiplier)
                else:
                    self.init_3DConv(multiplier=1*self.args.feature_multiplier)
            elif args.backbone_player == "3DConv" and args.backbone_feature is None:
                if args.with_resnet > 0:
                    if args.with_dense:
                        self.init_Dense(multiplier=2*self.args.feature_multiplier)
                    else:
                        self.init_2DConv_copy(multiplier=2*self.args.feature_multiplier)
                else:
                    self.init_3DConv(multiplier=2*self.args.feature_multiplier)

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
            if self.args.with_resnet > 0:
                if self.args.with_dense:
                    r_concatenation = self.forward_Dense(representation_inputs)
                else:
                    r_concatenation = self.forward_2DConv_copy(representation_inputs)
            else:
                r_concatenation = self.forward_3DConv(representation_inputs)
        
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

        elif r_concatenation is None and concatenation is not None:
            full_concatenation = concatenation
        elif r_concatenation is not None and concatenation is None:
            full_concatenation = r_concatenation

        # -------------------
        # Segmentation module
        # -------------------

        conv_seg = self.conv_seg(self.pad_seg(full_concatenation))
        

        conv_seg_permuted = conv_seg.permute(0,2,3,1)

        conv_seg_reshaped = conv_seg_permuted.view(conv_seg_permuted.size()[0],conv_seg_permuted.size()[1],self.dim_capsule,self.num_classes)



        conv_seg_norm = torch.sigmoid(self.batch_seg(conv_seg_reshaped))


        output_segmentation = torch.sqrt(torch.sum(torch.square(conv_seg_norm-0.5), dim=2)*4/self.dim_capsule)


        # ---------------
        # Spotting module
        # ---------------

        # Concatenation of the segmentation score to the capsules
        output_segmentation_reverse = 1-output_segmentation

        output_segmentation_reverse_reshaped = output_segmentation_reverse.unsqueeze(2)

        output_segmentation_reverse_reshaped_permutted = output_segmentation_reverse_reshaped.permute(0,3,1,2)

        concatenation_2 = torch.cat((conv_seg, output_segmentation_reverse_reshaped_permutted), dim=1)

        conv_spot = self.max_pool_spot(F.relu(concatenation_2))

        conv_spot_1 = F.relu(self.conv_spot_1(self.pad_spot_1(conv_spot)))

        conv_spot_1_pooled = self.max_pool_spot_1(conv_spot_1)

        conv_spot_2 = F.relu(self.conv_spot_2(self.pad_spot_2(conv_spot_1_pooled)))

        conv_spot_2_pooled = self.max_pool_spot_2(conv_spot_2)

        spotting_reshaped = conv_spot_2_pooled.view(conv_spot_2_pooled.size()[0],-1,1,1)

        # Confidence branch
        conf_pred = torch.sigmoid(self.conv_conf(spotting_reshaped).view(spotting_reshaped.shape[0],self.num_detections,2))

        # Class branch
        conf_class = self.softmax(self.conv_class(spotting_reshaped).view(spotting_reshaped.shape[0],self.num_detections,self.num_classes))

        output_spotting = torch.cat((conf_pred,conf_class),dim=-1)


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

    def init_2DConv_copy(self, multiplier=1):

        # Base Convolutional Layers
        if self.args.with_resnet == 4:
            self.conv_1_copy = nn.Conv2d(in_channels=1, out_channels=64*multiplier, kernel_size=(1,1792))
        else:
            self.conv_1_copy = nn.Conv2d(in_channels=1, out_channels=64*multiplier, kernel_size=(1,self.input_size))
        self.conv_2_copy = nn.Conv2d(in_channels=64*multiplier, out_channels=16*multiplier, kernel_size=(1,1))

        # Temporal Pyramidal Module
        self.pad_p_1_copy = nn.ZeroPad2d((0,0,(self.pyramid_size_1-1)//2, self.pyramid_size_1-1-(self.pyramid_size_1-1)//2))
        self.pad_p_2_copy = nn.ZeroPad2d((0,0,(self.pyramid_size_2-1)//2, self.pyramid_size_2-1-(self.pyramid_size_2-1)//2))
        self.pad_p_3_copy = nn.ZeroPad2d((0,0,(self.pyramid_size_3-1)//2, self.pyramid_size_3-1-(self.pyramid_size_3-1)//2))
        self.pad_p_4_copy = nn.ZeroPad2d((0,0,(self.pyramid_size_4-1)//2, self.pyramid_size_4-1-(self.pyramid_size_4-1)//2))
        self.conv_p_1_copy = nn.Conv2d(in_channels=16*multiplier, out_channels=4*multiplier, kernel_size=(self.pyramid_size_1,1))
        self.conv_p_2_copy = nn.Conv2d(in_channels=16*multiplier, out_channels=8*multiplier, kernel_size=(self.pyramid_size_2,1))
        self.conv_p_3_copy = nn.Conv2d(in_channels=16*multiplier, out_channels=16*multiplier, kernel_size=(self.pyramid_size_3,1))
        self.conv_p_4_copy = nn.Conv2d(in_channels=16*multiplier, out_channels=32*multiplier, kernel_size=(self.pyramid_size_4,1))

    def init_Dense(self, multiplier=1):
        if self.args.with_resnet == 4:
            self.dense_1 = nn.Conv2d(in_channels=1, out_channels=76*multiplier, kernel_size=(1,1792))
        else:
            self.dense_1 = nn.Conv2d(in_channels=1, out_channels=76*multiplier, kernel_size=(1,self.input_size))

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


    def forward_2DConv(self, inputs):

        # -------------------------------------
        # Temporal Convolutional neural network
        # -------------------------------------

        # Base Convolutional Layers
        conv_1 = F.relu(self.conv_1(inputs))
        
        conv_2 = F.relu(self.conv_2(conv_1))


        # Temporal Pyramidal Module
        conv_p_1 = F.relu(self.conv_p_1(self.pad_p_1(conv_2)))
        conv_p_2 = F.relu(self.conv_p_2(self.pad_p_2(conv_2)))
        conv_p_3 = F.relu(self.conv_p_3(self.pad_p_3(conv_2)))
        conv_p_4 = F.relu(self.conv_p_4(self.pad_p_4(conv_2)))

        concatenation = torch.cat((conv_2,conv_p_1,conv_p_2,conv_p_3,conv_p_4),1)
        
        return concatenation

    def forward_Dense(self, inputs):

        # Base Convolutional Layers
        concatenation = F.relu(self.dense_1(inputs))
        
        return concatenation

    def forward_2DConv_copy(self, inputs):

        # -------------------------------------
        # Temporal Convolutional neural network
        # -------------------------------------

        # Base Convolutional Layers
        conv_1 = F.relu(self.conv_1_copy(inputs))
        
        conv_2 = F.relu(self.conv_2_copy(conv_1))


        # Temporal Pyramidal Module
        conv_p_1 = F.relu(self.conv_p_1_copy(self.pad_p_1_copy(conv_2)))
        conv_p_2 = F.relu(self.conv_p_2_copy(self.pad_p_2_copy(conv_2)))
        conv_p_3 = F.relu(self.conv_p_3_copy(self.pad_p_3_copy(conv_2)))
        conv_p_4 = F.relu(self.conv_p_4_copy(self.pad_p_4_copy(conv_2)))

        concatenation = torch.cat((conv_2,conv_p_1,conv_p_2,conv_p_3,conv_p_4),1)
        
        return concatenation

    def forward_3DConv(self,representation_inputs):

        # --------------------
        # Representation input
        # --------------------
        #print("Representation input size: ", representation_inputs.size())

        #Base convolutional Layers

        r_conv_1 = F.relu(self.r_conv_1(representation_inputs))
        r_conv_1_pooled = self.r_max_pool_1(r_conv_1)

        r_conv_2 = F.relu(self.r_conv_2(r_conv_1_pooled))
        r_conv_2_pooled = self.r_max_pool_2(r_conv_2)

        # Temporal Pyramidal Module
        r_conv_p_1 = self.r_maxpool_p_1(F.relu(self.r_conv_p_1(F.pad(r_conv_2_pooled, self.r_pad_p_1, "constant", 0)))).squeeze(-1)

        r_conv_p_2 = self.r_maxpool_p_2(F.relu(self.r_conv_p_2(F.pad(r_conv_2_pooled, self.r_pad_p_2, "constant", 0)))).squeeze(-1)

        r_conv_p_3 = self.r_maxpool_p_3(F.relu(self.r_conv_p_3(F.pad(r_conv_2_pooled, self.r_pad_p_3, "constant", 0)))).squeeze(-1)

        r_conv_p_4 = self.r_maxpool_p_4(F.relu(self.r_conv_p_4(F.pad(r_conv_2_pooled, self.r_pad_p_4, "constant", 0)))).squeeze(-1)

        r_conv_2_pooled_pooled = self.r_maxpool_p_4(r_conv_2_pooled).squeeze(-1)


        r_concatenation = torch.cat((r_conv_2_pooled_pooled,r_conv_p_1,r_conv_p_2,r_conv_p_3,r_conv_p_4),1)


        return r_concatenation