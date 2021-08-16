from torch.utils.data import Dataset

import numpy as np
import random
# import pandas as pd
import os
import time


from tqdm import tqdm
# import utils

import torch

import logging
import json

from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from config.classes import EVENT_DICTIONARY_V2, K_V2, EVENT_DICTIONARY_V2_VISUAL, K_V2_VISUAL, EVENT_DICTIONARY_V2_NONVISUAL, K_V2_NONVISUAL

from preprocessing import oneHotToShifts, getTimestampTargets, getChunks_anchors, unproject_image_point, meter2radar

from torch_geometric.data import Data
from torch_geometric.data import Batch
import copy
import cv2

from torchvision import transforms
from efficientnet_pytorch import EfficientNet

class SoccerNetClips(Dataset):
    def __init__(self, path, split="train", args=None): 

        self.path = path
        self.args = args
        self.tiny = args.tiny
        self.listGames = getListGames(split)[0:self.tiny]
        self.features = args.features
        self.chunk_size = args.chunk_size*args.framerate
        self.receptive_field = args.receptive_field*args.framerate
        self.chunks_per_epoch = args.chunks_per_epoch
        self.framerate = args.framerate

        if self.args.class_split is None:
            self.dict_event = EVENT_DICTIONARY_V2
            self.K_parameters = K_V2*args.framerate 
            self.num_classes = 17
        elif self.args.class_split == "visual":
            self.dict_event = EVENT_DICTIONARY_V2_VISUAL
            self.K_parameters = K_V2_VISUAL*args.framerate 
            self.num_classes = 8
        elif self.args.class_split == "nonvisual":
            self.dict_event = EVENT_DICTIONARY_V2_NONVISUAL
            self.K_parameters = K_V2_NONVISUAL*args.framerate 
            self.num_classes = 9
        self.labels="Labels-v2.json"
        self.num_detections =args.num_detections
        self.split=split

        # Bounding boxes  
        self.bbox_predictions = "player_boundingbox_maskrcnn.json"
        self.representation_width = args.dim_representation_w
        self.representation_height = args.dim_representation_h
        self.representation_channel = args.dim_representation_c

        # Calibration 
        self.calibration_predictions = "field_calib_ccbv.json"
        self.calibration_threshold = 0.75
        if self.args.teacher:
            self.calibration_predictions = "HQ_25_teacher_calibration.json"
            self.calibration_threshold = 0.45
            print("Using the teacher calibration")
        self.size_radar_point = args.dim_representation_player
        self.dim_terrain = (68,105,self.representation_channel)
        self.dim_terrain_representation = (int(0.95*self.representation_height), int(0.95*self.representation_width), self.representation_channel)
        self.max_dist_player = args.dist_graph_player
        if self.args.calibration_field:
            self.radar_image = cv2.imread("src/config/radar.png", cv2.IMREAD_COLOR)
            if self.args.teacher:
                self.radar_image = cv2.imread("src/config/model-radar-mini.png", cv2.IMREAD_COLOR)
            self.radar_image = cv2.resize(self.radar_image, (self.dim_terrain_representation[1], self.dim_terrain_representation[0]),interpolation=cv2.INTER_CUBIC)

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[split], verbose=False)

        # Load the ResNet Model if required
        if self.args.with_resnet == 18:
            model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
            self.resnet = torch.nn.Sequential(*list(model.children())[:-1]).to("cuda").eval()
            self.resnet_transforms = transforms.Compose([transforms.ToPILImage(),transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            if self.args.calibration_physic:
                #self.features_resnet = "radar_resnet_18.npy"
                self.features_resnet = "radar_resnet_18_physic.npy"
                if self.args.teacher:
                    self.features_resnet = "radar_resnet_18_teacher_physic.npy"

            else:
                self.features_resnet = "radar_resnet_18_composite.npy"
                if self.args.teacher:
                    self.features_resnet = "radar_resnet_18_teacher_composite.npy"

        if self.args.with_resnet == 34:
            model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
            self.resnet = torch.nn.Sequential(*list(model.children())[:-1]).to("cuda").eval()
            self.resnet_transforms = transforms.Compose([transforms.ToPILImage(),transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            if self.args.calibration_physic:
                #self.features_resnet = "radar_resnet_18.npy"
                self.features_resnet = "radar_resnet_34_physic.npy"
                if self.args.teacher:
                    self.features_resnet = "radar_resnet_34_teacher_physic.npy"
            else:
                self.features_resnet = "radar_resnet_34_composite.npy"
                if self.args.teacher:
                    self.features_resnet = "radar_resnet_34_teacher_composite.npy"

        if self.args.with_resnet == 4:
            self.resnet = EfficientNet.from_pretrained('efficientnet-b4').to("cuda").eval()
            self.pooling = torch.nn.AdaptiveAvgPool2d(output_size=1).to("cuda").eval()
            self.resnet_transforms = transforms.Compose([transforms.ToPILImage(),transforms.Resize(224), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            if self.args.calibration_physic:
                #self.features_resnet = "radar_resnet_18.npy"
                self.features_resnet = "radar_efficientnet_b4_physic.npy"
                if self.args.teacher:
                    self.features_resnet = "radar_efficientnet_b4_teacher_physic.npy"
            else:
                self.features_resnet = "radar_efficientnet_b4_composite.npy"
                if self.args.teacher:
                    self.features_resnet = "radar_efficientnet_b4_teacher_composite.npy"

        def f_alpha(n):
            a = np.linspace(1/((n+1)//2), 1, (n+1)//2)
            return np.concatenate([a, np.flip(a[:-1])])
        self.trail_alpha = f_alpha(2*self.args.with_trail+1)
        print("alpha trail: ", self.trail_alpha)

        logging.info("Pre-compute clips")

        self.game_feats = list()
        self.game_labels = list()
        self.game_bbox = list()
        self.game_anchors = list()
        for i in np.arange(self.num_classes+1):
            self.game_anchors.append(list())

        game_counter = 0
        for game in tqdm(self.listGames):
            # Load features
            feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
            feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))

            # Load labels
            labels = json.load(open(os.path.join(self.path, game, self.labels)))

            label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
            label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))


            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = self.framerate * ( seconds + 60 * minutes ) 

                if event not in self.dict_event:
                    continue
                label = self.dict_event[event]



                if half == 1:
                    frame = min(frame, feat_half1.shape[0]-1)
                    label_half1[frame][label] = 1

                if half == 2:
                    frame = min(frame, feat_half2.shape[0]-1)
                    label_half2[frame][label] = 1

            shift_half1 = oneHotToShifts(label_half1, self.K_parameters.cpu().numpy())
            shift_half2 = oneHotToShifts(label_half2, self.K_parameters.cpu().numpy())

            anchors_half1 = getChunks_anchors(shift_half1, game_counter, self.K_parameters.cpu().numpy(), self.chunk_size, self.receptive_field)

            game_counter = game_counter+1

            anchors_half2 = getChunks_anchors(shift_half2, game_counter, self.K_parameters.cpu().numpy(), self.chunk_size, self.receptive_field)

            game_counter = game_counter+1

            # Saving the bounding boxes
            file_half1 = open(os.path.join(self.path, game, "1_" + self.bbox_predictions))
            bbox_half1 = json.load(file_half1)
            file_half1.close()
            file_half2 = open(os.path.join(self.path, game, "2_" + self.bbox_predictions))
            bbox_half2 = json.load(file_half2)
            file_half2.close()

            # Saving the calibration
            file_c_half1 = open(os.path.join(self.path, game, "1_" + self.calibration_predictions))
            calibration_half1 = json.load(file_c_half1)
            file_c_half1.close()
            file_c_half2 = open(os.path.join(self.path, game, "2_" + self.calibration_predictions))
            calibration_half2 = json.load(file_c_half2)
            file_c_half2.close()


            representation_half1 = None
            representation_half2 = None
            if self.args.backbone_player == "3DConv" and self.args.with_resnet >0 and os.path.exists(os.path.join(self.path, game, "1_" + self.features_resnet)) and os.path.exists(os.path.join(self.path, game, "2_" + self.features_resnet)):
                representation_half1 = np.load(os.path.join(self.path, game, "1_" + self.features_resnet))
                representation_half2 = np.load(os.path.join(self.path, game, "2_" + self.features_resnet))
            elif self.args.backbone_player == "3DConv":
                representation_half1 = np.zeros((feat_half1.shape[0], self.representation_height, self.representation_width, self.representation_channel), dtype=np.uint8)
                bbox_predictions = bbox_half1
                ratio_width = bbox_predictions["size"][2]/self.representation_width
                ratio_height = bbox_predictions["size"][1]/self.representation_height
                for i, bbox in enumerate(bbox_predictions["predictions"][0:feat_half1.shape[0]]):
                    if self.args.calibration:
                        confidence = calibration_half1["predictions"][i][0]["confidence"]
                        if confidence < self.calibration_threshold:
                            continue
                        homography = calibration_half1["predictions"][i][0]["homography"]
                        homography = np.reshape(homography, (3,3))
                        homography = homography/homography[2,2]
                        homography = np.linalg.inv(homography)
                        # Draw the field lines
                        if self.args.calibration_field:
                            if self.args.calibration_physic:
                                representation_half1[i,int(0.025*self.representation_height): int(0.025*self.representation_height) + self.radar_image.shape[0], int(0.025*self.representation_width): int(0.025*self.representation_width) + self.radar_image.shape[1], 2] =self.radar_image[:,:,2]
                            else:
                                representation_half1[i,int(0.025*self.representation_height): int(0.025*self.representation_height) + self.radar_image.shape[0], int(0.025*self.representation_width): int(0.025*self.representation_width) + self.radar_image.shape[1]] =self.radar_image

                        # Draw the calibration cones
                        if self.args.calibration_cone:
                            frame_top_left_projected = unproject_image_point(homography, np.array([0,0,1]))
                            frame_top_right_projected = unproject_image_point(homography, np.array([calibration_half1["size"][2],0,1]))
                            frame_bottom_left_projected = unproject_image_point(homography, np.array([0,calibration_half1["size"][1],1]))
                            frame_bottom_right_projected = unproject_image_point(homography, np.array([calibration_half1["size"][2],calibration_half1["size"][1],1]))
                            frame_top_left_radar = meter2radar(frame_top_left_projected, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                            frame_top_right_radar = meter2radar(frame_top_right_projected, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                            frame_bottom_left_radar = meter2radar(frame_bottom_left_projected, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                            frame_bottom_right_radar = meter2radar(frame_bottom_right_projected, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))

                            pts = np.array([frame_top_left_radar[0:2], frame_top_right_radar[0:2], frame_bottom_right_radar[0:2], frame_bottom_left_radar[0:2]], np.int32)


                            if self.args.calibration_physic:
                                image_polygone = cv2.fillPoly(np.copy(representation_half1[i]), [pts], (255,255,255))
                                representation_half1[i,:,:,1] = image_polygone[:,:,1]
                            else:
                                representation_half1[i] = cv2.polylines(representation_half1[i], [pts], True, (255,255,255), 1)

                        # Draw the players
                        if self.args.with_trail > 0:
                            bbox_trail = bbox_predictions["predictions"][max(0,i-self.args.with_trail):min(i+self.args.with_trail+1,feat_half1.shape[0])]
                            for j, bbox_trail_i in enumerate(bbox_trail):
                                for (box, color) in zip(bbox_trail_i["bboxes"], bbox_trail_i["colors"]):
                                    projection_point = np.array([int((box[0]+box[2])/2), box[3], 1])
                                    projected_point = unproject_image_point(homography, projection_point)
                                    radar_point = meter2radar(projected_point, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                                    if radar_point[0]+self.size_radar_point//2 < self.representation_width and radar_point[0]-self.size_radar_point//2 >=0 and radar_point[1]+self.size_radar_point//2 < self.representation_height and radar_point[1]-self.size_radar_point//2 >= 0:
                                        if args.calibration_physic:
                                            representation_half1[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,0] = np.uint8(int(255*self.trail_alpha[j]))
                                        else:
                                            representation_half1[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,0] = np.uint8(int(color[0]*self.trail_alpha[j]))
                                            representation_half1[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,1] = np.uint8(int(color[1]*self.trail_alpha[j]))
                                            representation_half1[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,2] = np.uint8(int(color[2]*self.trail_alpha[j]))


                        for box, color in zip(bbox["bboxes"], bbox["colors"]):
                            projection_point = np.array([int((box[0]+box[2])/2), box[3], 1])
                            projected_point = unproject_image_point(homography, projection_point)
                            radar_point = meter2radar(projected_point, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                            if radar_point[0]+self.size_radar_point//2 < self.representation_width and radar_point[0]-self.size_radar_point//2 >=0 and radar_point[1]+self.size_radar_point//2 < self.representation_height and radar_point[1]-self.size_radar_point//2 >= 0:
                                if args.calibration_physic:
                                    representation_half1[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,0] = 255
                                else:
                                    representation_half1[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,0] = np.uint8(int(color[0]))
                                    representation_half1[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,1] = np.uint8(int(color[1]))
                                    representation_half1[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,2] = np.uint8(int(color[2]))

                    else:
                        for box, color in zip(bbox["bboxes"], bbox["colors"]):
                            x_top_left = min(max(0,int(box[0]/ratio_width)),self.representation_width-1)
                            x_bottom_right = min(max(0,int(box[2]/ratio_width)),self.representation_width-1)
                            y_top_left = min(max(0,int(box[1]/ratio_height)),self.representation_height-1)
                            y_bottom_right = min(max(0,int(box[3]/ratio_height)),self.representation_height-1)
                            if x_bottom_right > x_top_left and y_bottom_right > y_top_left:
                                representation_half1[i,y_top_left:y_bottom_right, x_top_left:x_bottom_right,0] = np.uint8(int(color[0]))
                                representation_half1[i,y_top_left:y_bottom_right, x_top_left:x_bottom_right,1] = np.uint8(int(color[1]))
                                representation_half1[i,y_top_left:y_bottom_right, x_top_left:x_bottom_right,2] = np.uint8(int(color[2]))
                #Visualization
                #for i, frame in enumerate(representation_half1):
                #    cv2.imwrite("outputs/test/"+str(i)+".png", frame)
                #ygfdtffr

                if self.args.with_resnet > 0:
                    if self.args.with_resnet == 4:
                        representation_half1_np = np.zeros((feat_half1.shape[0],1792))
                    else:
                        representation_half1_np = np.zeros(feat_half1.shape)
                    representation_half1 = torch.Tensor(representation_half1).float()
                    representation_half1 = representation_half1.permute(0,3,1,2)
                    for i, img in tqdm(enumerate(representation_half1)):
                        img = img/255
                        preprocessed_img = self.resnet_transforms(img)
                        with torch.no_grad():
                            if self.args.with_resnet == 4:
                                output = self.resnet.extract_features(preprocessed_img.to("cuda").unsqueeze(0))
                                output = self.pooling(output)
                            else:
                                output = self.resnet(preprocessed_img.to("cuda").unsqueeze(0))
                            
                        representation_half1_np[i] = output.to("cpu").numpy()[0,:,0,0]
                    representation_half1 = representation_half1_np.astype(float)


                    np.save(os.path.join(self.path, game, "1_" + self.features_resnet),representation_half1)
                    
                

                representation_half2 = np.zeros((feat_half2.shape[0], self.representation_height, self.representation_width, self.representation_channel), dtype=np.uint8)
                bbox_predictions = bbox_half2
                ratio_width = bbox_predictions["size"][2]/self.representation_width
                ratio_height = bbox_predictions["size"][1]/self.representation_height
                for i, bbox in enumerate(bbox_predictions["predictions"][0:feat_half2.shape[0]]):
                    if self.args.calibration:
                        confidence = calibration_half2["predictions"][i][0]["confidence"]
                        if confidence < self.calibration_threshold:
                            continue
                        homography = calibration_half2["predictions"][i][0]["homography"]
                        homography = np.reshape(homography, (3,3))
                        homography = homography/homography[2,2]
                        homography = np.linalg.inv(homography)
                        # Draw the field lines
                        if self.args.calibration_field:
                            if self.args.calibration_physic:
                                representation_half2[i,int(0.025*self.representation_height): int(0.025*self.representation_height) + self.radar_image.shape[0], int(0.025*self.representation_width): int(0.025*self.representation_width) + self.radar_image.shape[1],2 ] =self.radar_image[:,:,2]
                            else:
                                representation_half2[i,int(0.025*self.representation_height): int(0.025*self.representation_height) + self.radar_image.shape[0], int(0.025*self.representation_width): int(0.025*self.representation_width) + self.radar_image.shape[1] ] =self.radar_image

                        # Draw the calibration cones
                        if self.args.calibration_cone:
                            frame_top_left_projected = unproject_image_point(homography, np.array([0,0,1]))
                            frame_top_right_projected = unproject_image_point(homography, np.array([calibration_half2["size"][2],0,1]))
                            frame_bottom_left_projected = unproject_image_point(homography, np.array([0,calibration_half2["size"][1],1]))
                            frame_bottom_right_projected = unproject_image_point(homography, np.array([calibration_half2["size"][2],calibration_half2["size"][1],1]))
                            frame_top_left_radar = meter2radar(frame_top_left_projected, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                            frame_top_right_radar = meter2radar(frame_top_right_projected, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                            frame_bottom_left_radar = meter2radar(frame_bottom_left_projected, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                            frame_bottom_right_radar = meter2radar(frame_bottom_right_projected, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))

                            pts = np.array([frame_top_left_radar[0:2], frame_top_right_radar[0:2], frame_bottom_right_radar[0:2], frame_bottom_left_radar[0:2]], np.int32)
                            if self.args.calibration_physic:
                                image_polygone = cv2.fillPoly(np.copy(representation_half2[i]), [pts], (255,255,255))
                                representation_half2[i,:,:,1] = image_polygone[:,:,1]
                            else:
                                representation_half2[i] = cv2.polylines(representation_half2[i], [pts], True, (255,255,255), 1)

                        # Draw the players
                        if self.args.with_trail > 0:
                            bbox_trail = bbox_predictions["predictions"][max(0,i-self.args.with_trail):min(i+self.args.with_trail+1,feat_half2.shape[0])]
                            for j, bbox_trail_i in enumerate(bbox_trail):
                                for (box, color) in zip(bbox_trail_i["bboxes"], bbox_trail_i["colors"]):
                                    projection_point = np.array([int((box[0]+box[2])/2), box[3], 1])
                                    projected_point = unproject_image_point(homography, projection_point)
                                    radar_point = meter2radar(projected_point, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                                    if radar_point[0]+self.size_radar_point//2 < self.representation_width and radar_point[0]-self.size_radar_point//2 >=0 and radar_point[1]+self.size_radar_point//2 < self.representation_height and radar_point[1]-self.size_radar_point//2 >= 0:
                                        if args.calibration_physic:
                                            representation_half2[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,0] = np.uint8(int(255*self.trail_alpha[j]))
                                        else:
                                            representation_half2[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,0] = np.uint8(int(color[0])*self.trail_alpha[j])
                                            representation_half2[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,1] = np.uint8(int(color[1])*self.trail_alpha[j])
                                            representation_half2[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,2] = np.uint8(int(color[2])*self.trail_alpha[j])
                        for box, color in zip(bbox["bboxes"], bbox["colors"]):
                            projection_point = np.array([int((box[0]+box[2])/2), box[3], 1])
                            projected_point = unproject_image_point(homography, projection_point)
                            radar_point = meter2radar(projected_point, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                            if radar_point[0]+self.size_radar_point//2 < self.representation_width and radar_point[0]-self.size_radar_point//2 >=0 and radar_point[1]+self.size_radar_point//2 < self.representation_height and radar_point[1]-self.size_radar_point//2 >= 0:
                                if args.calibration_physic:
                                    representation_half2[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,0] = 255
                                else:
                                    representation_half2[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,0] = np.uint8(int(color[0]))
                                    representation_half2[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,1] = np.uint8(int(color[1]))
                                    representation_half2[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,2] = np.uint8(int(color[2]))

                    else:
                        for box, color in zip(bbox["bboxes"], bbox["colors"]):
                            x_top_left = min(max(0,int(box[0]/ratio_width)),self.representation_width-1)
                            x_bottom_right = min(max(0,int(box[2]/ratio_width)),self.representation_width-1)
                            y_top_left = min(max(0,int(box[1]/ratio_height)),self.representation_height-1)
                            y_bottom_right = min(max(0,int(box[3]/ratio_height)),self.representation_height-1)
                            if x_bottom_right > x_top_left and y_bottom_right > y_top_left:
                                representation_half2[i,y_top_left:y_bottom_right, x_top_left:x_bottom_right,0] = np.uint8(int(color[0]))
                                representation_half2[i,y_top_left:y_bottom_right, x_top_left:x_bottom_right,1] = np.uint8(int(color[1]))
                                representation_half2[i,y_top_left:y_bottom_right, x_top_left:x_bottom_right,2] = np.uint8(int(color[2]))

                if self.args.with_resnet > 0:
                    if self.args.with_resnet == 4:
                        representation_half2_np = np.zeros((feat_half2.shape[0],1792))
                    else:
                        representation_half2_np = np.zeros(feat_half2.shape)
                    representation_half2 = torch.Tensor(representation_half2).float()
                    representation_half2 = representation_half2.permute(0,3,1,2)
                    for i, img in tqdm(enumerate(representation_half2)):
                        img = img/255
                        preprocessed_img = self.resnet_transforms(img)
                        with torch.no_grad():
                            if self.args.with_resnet == 4:
                                output = self.resnet.extract_features(preprocessed_img.to("cuda").unsqueeze(0))
                                output = self.pooling(output)
                            else:
                                output = self.resnet(preprocessed_img.to("cuda").unsqueeze(0))
                        representation_half2_np[i] = output.to("cpu").numpy()[0,:,0,0]
                    representation_half2 = representation_half2_np.astype(float)
                        # Save it for next time
                    np.save(os.path.join(self.path, game, "2_" + self.features_resnet),representation_half2)

                # Visualization
                #for i, frame in enumerate(representation_half2):
                #    cv2.imwrite("outputs/test/"+str(i)+".png", frame)

            
      
            
            self.game_bbox.append(representation_half1)
            self.game_bbox.append(representation_half2)


            self.game_feats.append(feat_half1)
            self.game_feats.append(feat_half2)
            self.game_labels.append(shift_half1)
            self.game_labels.append(shift_half2)
            for anchor in anchors_half1:
                self.game_anchors[anchor[2]].append(anchor)
            for anchor in anchors_half2:
                self.game_anchors[anchor[2]].append(anchor)
            

    def __getitem__(self, index):

        # Retrieve the game index and the anchor
        class_selection = random.randint(0, self.num_classes)
        event_selection = random.randint(0, len(self.game_anchors[class_selection])-1)
        game_index = self.game_anchors[class_selection][event_selection][0]
        anchor = self.game_anchors[class_selection][event_selection][1]

        # Compute the shift for event chunks
        if class_selection < self.num_classes:
            shift = np.random.randint(-self.chunk_size+self.receptive_field, -self.receptive_field)
            start = anchor + shift
        # Compute the shift for non-event chunks
        else:
            start = random.randint(anchor[0], anchor[1]-self.chunk_size)
        if start < 0:
            start = 0
        if start+self.chunk_size >= self.game_feats[game_index].shape[0]:
            start = self.game_feats[game_index].shape[0]-self.chunk_size-1

        # Extract the clips
        clip_feat = self.game_feats[game_index][start:start+self.chunk_size]
        clip_labels = self.game_labels[game_index][start:start+self.chunk_size]

        # Put loss to zero outside receptive field
        clip_labels[0:int(np.ceil(self.receptive_field/2)),:] = -1
        clip_labels[-int(np.ceil(self.receptive_field/2)):,:] = -1

        # Get the spotting target
        clip_targets = getTimestampTargets(np.array([clip_labels]), self.num_detections)[0]

        clip_representation = None
        # Get the player representation
        if self.args.backbone_player == "3DConv":
            clip_representation = copy.deepcopy(self.game_bbox[game_index][start:start+self.chunk_size])
            clip_representation = clip_representation.astype(float)
            if self.args.with_resnet == 0:
                clip_representation = clip_representation/127-1
        elif self.args.backbone_player is None:
            clip_representation = np.zeros((clip_feat.shape[0], 1))

        return torch.from_numpy(clip_feat), torch.from_numpy(clip_labels), torch.from_numpy(clip_targets), torch.from_numpy(clip_representation)

    def __len__(self):
        return self.chunks_per_epoch

class SoccerNetClipsTesting(Dataset):
    def __init__(self, path, split="test", args=None):
        self.path = path
        self.args = args
        self.tiny = args.tiny
        self.listGames = getListGames(split)[:self.tiny]
        self.features = args.features
        self.chunk_size = args.chunk_size*args.framerate
        self.receptive_field = args.receptive_field*args.framerate
        self.framerate = args.framerate

        if self.args.class_split is None:
            self.dict_event = EVENT_DICTIONARY_V2
            self.K_parameters = K_V2*args.framerate 
            self.num_classes = 17
        elif self.args.class_split == "visual":
            self.dict_event = EVENT_DICTIONARY_V2_VISUAL
            self.K_parameters = K_V2_VISUAL*args.framerate 
            self.num_classes = 8
        elif self.args.class_split == "nonvisual":
            self.dict_event = EVENT_DICTIONARY_V2_NONVISUAL
            self.K_parameters = K_V2_NONVISUAL*args.framerate 
            self.num_classes = 9
        
        self.labels="Labels-v2.json"
        self.num_detections =args.num_detections
        self.split=split

        # Bounding boxes  
        self.bbox_predictions = "player_boundingbox_maskrcnn.json"
        self.representation_width = args.dim_representation_w
        self.representation_height = args.dim_representation_h
        self.representation_channel = args.dim_representation_c

        # Calibration 
        self.calibration_predictions = "field_calib_ccbv.json"
        self.calibration_threshold = 0.75
        if self.args.teacher:
            self.calibration_predictions = "HQ_25_teacher_calibration.json"
            self.calibration_threshold = 0.45
            print("Using the teacher calibration")
        self.size_radar_point = args.dim_representation_player
        self.dim_terrain = (68,105,self.representation_channel)
        self.dim_terrain_representation = (int(0.95*self.representation_height), int(0.95*self.representation_width), self.representation_channel)
        self.max_dist_player = args.dist_graph_player
        if self.args.calibration_field:
            self.radar_image = cv2.imread("src/config/radar.png", cv2.IMREAD_COLOR)
            if self.args.teacher:
                self.radar_image = cv2.imread("src/config/model-radar-mini.png", cv2.IMREAD_COLOR)
            self.radar_image = cv2.resize(self.radar_image, (self.dim_terrain_representation[1], self.dim_terrain_representation[0]),interpolation=cv2.INTER_CUBIC)

        # Load the ResNet Model if required
        if self.args.with_resnet == 18:
            model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
            self.resnet = torch.nn.Sequential(*list(model.children())[:-1]).to("cuda").eval()
            self.resnet_transforms = transforms.Compose([transforms.ToPILImage(),transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            if self.args.calibration_physic:
                #self.features_resnet = "radar_resnet_18.npy"
                self.features_resnet = "radar_resnet_18_physic.npy"
                if self.args.teacher:
                    self.features_resnet = "radar_resnet_18_teacher_physic.npy"
            else:
                self.features_resnet = "radar_resnet_18_composite.npy"
                if self.args.teacher:
                    self.features_resnet = "radar_resnet_18_teacher_composite.npy"

        if self.args.with_resnet == 34:
            model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
            self.resnet = torch.nn.Sequential(*list(model.children())[:-1]).to("cuda").eval()
            self.resnet_transforms = transforms.Compose([transforms.ToPILImage(),transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            if self.args.calibration_physic:
                #self.features_resnet = "radar_resnet_18.npy"
                self.features_resnet = "radar_resnet_34_physic.npy"
                if self.args.teacher:
                    self.features_resnet = "radar_resnet_34_teacher_physic.npy"
            else:
                self.features_resnet = "radar_resnet_34_composite.npy"
                if self.args.teacher:
                    self.features_resnet = "radar_resnet_34_teacher_composite.npy"

        if self.args.with_resnet == 4:
            self.resnet = EfficientNet.from_pretrained('efficientnet-b4').to("cuda").eval()
            self.pooling = torch.nn.AdaptiveAvgPool2d(output_size=1).to("cuda").eval()
            self.resnet_transforms = transforms.Compose([transforms.ToPILImage(),transforms.Resize(224), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            if self.args.calibration_physic:
                #self.features_resnet = "radar_resnet_18.npy"
                self.features_resnet = "radar_efficientnet_b4_physic.npy"
                if self.args.teacher:
                    self.features_resnet = "radar_efficientnet_b4_teacher_physic.npy"
            else:
                self.features_resnet = "radar_efficientnet_b4_composite.npy"
                if self.args.teacher:
                    self.features_resnet = "radar_efficientnet_b4_teacher_composite.npy"

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        if split == "challenge":
            downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], split=[split], verbose=False)
        else:       
            downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[split], verbose=False)

        def f_alpha(n):
            a = np.linspace(1/((n+1)//2), 1, (n+1)//2)
            return np.concatenate([a, np.flip(a[:-1])])
        self.trail_alpha = f_alpha(2*self.args.with_trail+1)
        print("alpha trail: ", self.trail_alpha)


    def __getitem__(self, index):

        def feats2clip(feats, stride, clip_length):

            idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
            idxs = []
            for i in torch.arange(0, clip_length):
                idxs.append(idx+i)
            idx = torch.stack(idxs, dim=1)

            idx = idx.clamp(0, feats.shape[0]-1)
            idx[-1] = torch.arange(clip_length)+feats.shape[0]-clip_length

            return feats[idx,:]
                    
        # Load features
        feat_half1 = np.load(os.path.join(self.path, self.listGames[index], "1_" + self.features))
        feat_half2 = np.load(os.path.join(self.path, self.listGames[index], "2_" + self.features))

        # Representation of the bounding boxes
        file_half1 = open(os.path.join(self.path, self.listGames[index], "1_" + self.bbox_predictions))
        bbox_half1 = json.load(file_half1)
        file_half1.close()
        file_half2 = open(os.path.join(self.path, self.listGames[index], "2_" + self.bbox_predictions))
        bbox_half2 = json.load(file_half2)
        file_half2.close()

        # Saving the calibration
        file_c_half1 = open(os.path.join(self.path, self.listGames[index], "1_" + self.calibration_predictions))
        calibration_half1 = json.load(file_c_half1)
        file_c_half1.close()
        file_c_half2 = open(os.path.join(self.path, self.listGames[index], "2_" + self.calibration_predictions))
        calibration_half2 = json.load(file_c_half2)
        file_c_half2.close()

        clip_representation_half1 = None
        clip_representation_half2 = None
        if self.args.backbone_player == "3DConv" and self.args.with_resnet >0 and os.path.exists(os.path.join(self.path, self.listGames[index], "1_" + self.features_resnet)) and os.path.exists(os.path.join(self.path, self.listGames[index], "2_" + self.features_resnet)):
            clip_representation_half1 = np.load(os.path.join(self.path, self.listGames[index], "1_" + self.features_resnet))
            clip_representation_half2 = np.load(os.path.join(self.path, self.listGames[index], "2_" + self.features_resnet))

            clip_representation_half1 = feats2clip(torch.from_numpy(clip_representation_half1), 
                            stride=self.chunk_size-self.receptive_field, 
                            clip_length=self.chunk_size)

            clip_representation_half2 = feats2clip(torch.from_numpy(clip_representation_half2), 
                            stride=self.chunk_size-self.receptive_field, 
                            clip_length=self.chunk_size)
            
        elif self.args.backbone_player == "3DConv":
            if self.args.calibration:
                clip_representation_half1 = np.zeros((feat_half1.shape[0], self.representation_height, self.representation_width, self.representation_channel), dtype = np.uint8)
                clip_representation_half2 = np.zeros((feat_half2.shape[0], self.representation_height, self.representation_width, self.representation_channel), dtype = np.uint8)
                    
            else:
                clip_representation_half1 = np.zeros((feat_half1.shape[0], self.representation_height, self.representation_width, self.representation_channel))
                clip_representation_half2 = np.zeros((feat_half2.shape[0], self.representation_height, self.representation_width, self.representation_channel))
                             
            ratio_width = bbox_half1["size"][2]/self.representation_width
            ratio_height = bbox_half1["size"][1]/self.representation_height
            for i, bbox in enumerate(bbox_half1["predictions"][0:feat_half1.shape[0]]):
                if self.args.calibration:
                    confidence = calibration_half1["predictions"][i][0]["confidence"]
                    if confidence < self.calibration_threshold:
                        continue
                    homography = calibration_half1["predictions"][i][0]["homography"]
                    homography = np.reshape(homography, (3,3))
                    homography = homography/homography[2,2]
                    homography = np.linalg.inv(homography)
                    # Draw the field lines
                    if self.args.calibration_field:
                        if self.args.calibration_physic:
                            clip_representation_half1[i,int(0.025*self.representation_height): int(0.025*self.representation_height) + self.radar_image.shape[0], int(0.025*self.representation_width): int(0.025*self.representation_width) + self.radar_image.shape[1], 2 ] =self.radar_image[:,:,2]
                        else:
                            clip_representation_half1[i,int(0.025*self.representation_height): int(0.025*self.representation_height) + self.radar_image.shape[0], int(0.025*self.representation_width): int(0.025*self.representation_width) + self.radar_image.shape[1] ] =self.radar_image

                    # Draw the calibration cones
                    if self.args.calibration_cone:
                        frame_top_left_projected = unproject_image_point(homography, np.array([0,0,1]))
                        frame_top_right_projected = unproject_image_point(homography, np.array([calibration_half1["size"][2],0,1]))
                        frame_bottom_left_projected = unproject_image_point(homography, np.array([0,calibration_half1["size"][1],1]))
                        frame_bottom_right_projected = unproject_image_point(homography, np.array([calibration_half1["size"][2],calibration_half1["size"][1],1]))
                        frame_top_left_radar = meter2radar(frame_top_left_projected, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                        frame_top_right_radar = meter2radar(frame_top_right_projected, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                        frame_bottom_left_radar = meter2radar(frame_bottom_left_projected, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                        frame_bottom_right_radar = meter2radar(frame_bottom_right_projected, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))

                        pts = np.array([frame_top_left_radar[0:2], frame_top_right_radar[0:2], frame_bottom_right_radar[0:2], frame_bottom_left_radar[0:2]], np.int32)
                        if self.args.calibration_physic:
                            image_polygone = cv2.fillPoly(np.copy(clip_representation_half1[i]), [pts], (255,255,255))
                            clip_representation_half1[i,:,:,1] = image_polygone[:,:,1]
                        else:
                            clip_representation_half1[i] = cv2.polylines(clip_representation_half1[i], [pts], True, (255,255,255), 1)

                    # Draw the players
                    if self.args.with_trail > 0:
                        bbox_trail = bbox_half1["predictions"][max(0,i-self.args.with_trail):min(i+self.args.with_trail+1,feat_half1.shape[0])]
                        for j, bbox_trail_i in enumerate(bbox_trail):
                            for (box, color) in zip(bbox_trail_i["bboxes"], bbox_trail_i["colors"]):
                                projection_point = np.array([int((box[0]+box[2])/2), box[3], 1])
                                projected_point = unproject_image_point(homography, projection_point)
                                radar_point = meter2radar(projected_point, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                                if radar_point[0]+self.size_radar_point//2 < self.representation_width and radar_point[0]-self.size_radar_point//2 >=0 and radar_point[1]+self.size_radar_point//2 < self.representation_height and radar_point[1]-self.size_radar_point//2 >= 0:
                                    if self.args.calibration_physic:
                                        clip_representation_half1[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,0] = np.uint8(int(255*self.trail_alpha[j]))
                                    else:
                                        clip_representation_half1[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,0] = np.uint8(int(color[0]*self.trail_alpha[j]))
                                        clip_representation_half1[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,1] = np.uint8(int(color[1]*self.trail_alpha[j]))
                                        clip_representation_half1[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,2] = np.uint8(int(color[2]*self.trail_alpha[j]))
                    for box, color in zip(bbox["bboxes"], bbox["colors"]):
                        projection_point = np.array([int((box[0]+box[2])/2), box[3], 1])
                        projected_point = unproject_image_point(homography, projection_point)
                        radar_point = meter2radar(projected_point, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                        if radar_point[0]+self.size_radar_point//2 < self.representation_width and radar_point[0]-self.size_radar_point//2 >=0 and radar_point[1]+self.size_radar_point//2 < self.representation_height and radar_point[1]-self.size_radar_point//2 >= 0:
                            if self.args.calibration_physic:
                                clip_representation_half1[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,0] = 255
                            else:
                                clip_representation_half1[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,0] = np.uint8(int(color[0]))
                                clip_representation_half1[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,1] = np.uint8(int(color[1]))
                                clip_representation_half1[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,2] = np.uint8(int(color[2]))
                
                else:
                    for box, color in zip(bbox["bboxes"], bbox["colors"]):
                        x_top_left = min(max(0,int(box[0]/ratio_width)),self.representation_width-1)
                        x_bottom_right = min(max(0,int(box[2]/ratio_width)),self.representation_width-1)
                        y_top_left = min(max(0,int(box[1]/ratio_height)),self.representation_height-1)
                        y_bottom_right = min(max(0,int(box[3]/ratio_height)),self.representation_height-1)
                        color_np = np.array([(color[0]/127)-1, (color[1]/127)-1, (color[2]/127)-1])
                        if x_bottom_right > x_top_left and y_bottom_right > y_top_left:
                            clip_representation_half1[i,y_top_left:y_bottom_right, x_top_left:x_bottom_right] = color_np

            ratio_width = bbox_half2["size"][2]/self.representation_width
            ratio_height = bbox_half2["size"][1]/self.representation_height
            for i, bbox in enumerate(bbox_half2["predictions"][0:feat_half2.shape[0]]):
                if self.args.calibration:
                    confidence = calibration_half2["predictions"][i][0]["confidence"]
                    if confidence < self.calibration_threshold:
                        continue
                    homography = calibration_half2["predictions"][i][0]["homography"]
                    homography = np.reshape(homography, (3,3))
                    homography = homography/homography[2,2]
                    homography = np.linalg.inv(homography)
                    # Draw the field lines
                    if self.args.calibration_field:
                        if self.args.calibration_physic:
                            clip_representation_half2[i,int(0.025*self.representation_height): int(0.025*self.representation_height) + self.radar_image.shape[0], int(0.025*self.representation_width): int(0.025*self.representation_width) + self.radar_image.shape[1],2 ] =self.radar_image[:,:,2]
                        else:
                            clip_representation_half2[i,int(0.025*self.representation_height): int(0.025*self.representation_height) + self.radar_image.shape[0], int(0.025*self.representation_width): int(0.025*self.representation_width) + self.radar_image.shape[1] ] =self.radar_image

                    # Draw the calibration cones
                    if self.args.calibration_cone:
                        frame_top_left_projected = unproject_image_point(homography, np.array([0,0,1]))
                        frame_top_right_projected = unproject_image_point(homography, np.array([calibration_half2["size"][2],0,1]))
                        frame_bottom_left_projected = unproject_image_point(homography, np.array([0,calibration_half2["size"][1],1]))
                        frame_bottom_right_projected = unproject_image_point(homography, np.array([calibration_half2["size"][2],calibration_half2["size"][1],1]))
                        frame_top_left_radar = meter2radar(frame_top_left_projected, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                        frame_top_right_radar = meter2radar(frame_top_right_projected, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                        frame_bottom_left_radar = meter2radar(frame_bottom_left_projected, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                        frame_bottom_right_radar = meter2radar(frame_bottom_right_projected, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))

                        pts = np.array([frame_top_left_radar[0:2], frame_top_right_radar[0:2], frame_bottom_right_radar[0:2], frame_bottom_left_radar[0:2]], np.int32)
                        if self.args.calibration_physic:
                            image_polygone = cv2.fillPoly(np.copy(clip_representation_half2[i]), [pts], (255,255,255))
                            clip_representation_half2[i,:,:,1] = image_polygone[:,:,1]
                        else:
                            clip_representation_half2[i] = cv2.polylines(clip_representation_half2[i], [pts], True, (255,255,255), 1)

                    # Draw the players
                    if self.args.with_trail > 0:
                        bbox_trail = bbox_half2["predictions"][max(0,i-self.args.with_trail):min(i+self.args.with_trail+1,feat_half2.shape[0])]
                        for j, bbox_trail_i in enumerate(bbox_trail):
                            for (box, color) in zip(bbox_trail_i["bboxes"], bbox_trail_i["colors"]):
                                projection_point = np.array([int((box[0]+box[2])/2), box[3], 1])
                                projected_point = unproject_image_point(homography, projection_point)
                                radar_point = meter2radar(projected_point, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                                if radar_point[0]+self.size_radar_point//2 < self.representation_width and radar_point[0]-self.size_radar_point//2 >=0 and radar_point[1]+self.size_radar_point//2 < self.representation_height and radar_point[1]-self.size_radar_point//2 >= 0:
                                    if self.args.calibration_physic:
                                        clip_representation_half2[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,0] = np.uint8(int(255*self.trail_alpha[j]))
                                    else:
                                        clip_representation_half2[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,0] = np.uint8(int(color[0]*self.trail_alpha[j]))
                                        clip_representation_half2[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,1] = np.uint8(int(color[1]*self.trail_alpha[j]))
                                        clip_representation_half2[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,2] = np.uint8(int(color[2]*self.trail_alpha[j]))

                    for box, color in zip(bbox["bboxes"], bbox["colors"]):
                        projection_point = np.array([int((box[0]+box[2])/2), box[3], 1])
                        projected_point = unproject_image_point(homography, projection_point)
                        radar_point = meter2radar(projected_point, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                        if radar_point[0]+self.size_radar_point//2 < self.representation_width and radar_point[0]-self.size_radar_point//2 >=0 and radar_point[1]+self.size_radar_point//2 < self.representation_height and radar_point[1]-self.size_radar_point//2 >= 0:
                            if self.args.calibration_physic:
                                clip_representation_half2[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,0] = 255
                            else:
                                clip_representation_half2[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,0] = np.uint8(int(color[0]))
                                clip_representation_half2[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,1] = np.uint8(int(color[1]))
                                clip_representation_half2[i,radar_point[1]-self.size_radar_point//2:radar_point[1]+self.size_radar_point//2, radar_point[0]-self.size_radar_point//2:radar_point[0]+self.size_radar_point//2,2] = np.uint8(int(color[2]))
                else:
                    for box, color in zip(bbox["bboxes"], bbox["colors"]):
                        x_top_left = min(max(0,int(box[0]/ratio_width)),self.representation_width-1)
                        x_bottom_right = min(max(0,int(box[2]/ratio_width)),self.representation_width-1)
                        y_top_left = min(max(0,int(box[1]/ratio_height)),self.representation_height-1)
                        y_bottom_right = min(max(0,int(box[3]/ratio_height)),self.representation_height-1)
                        color_np = np.array([(color[0]/127)-1, (color[1]/127)-1, (color[2]/127)-1])
                        if x_bottom_right > x_top_left and y_bottom_right > y_top_left:
                            clip_representation_half2[i,y_top_left:y_bottom_right, x_top_left:x_bottom_right] = color_np

            

            if self.args.with_resnet > 0:
                clip_representation_half1_np = np.zeros(feat_half1.shape)
                clip_representation_half1 = torch.Tensor(clip_representation_half1).float()
                clip_representation_half1 = clip_representation_half1.permute(0,3,1,2)
                for i, img in tqdm(enumerate(clip_representation_half1)):
                    img=img/255
                    preprocessed_img = self.resnet_transforms(img)
                    with torch.no_grad():
                        output = self.resnet(preprocessed_img.to("cuda").unsqueeze(0))
                    clip_representation_half1_np[i] = output.to("cpu").numpy()[0,:,0,0]
                clip_representation_half1 = clip_representation_half1_np.astype(float)

                np.save(os.path.join(self.path, self.listGames[index], "1_" + self.features_resnet),clip_representation_half1)

                clip_representation_half2_np = np.zeros(feat_half2.shape)
                clip_representation_half2 = torch.Tensor(clip_representation_half2).float()
                clip_representation_half2 = clip_representation_half2.permute(0,3,1,2)
                for i, img in tqdm(enumerate(clip_representation_half2)):
                    img=img/255
                    preprocessed_img = self.resnet_transforms(img)
                    with torch.no_grad():
                        output = self.resnet(preprocessed_img.to("cuda").unsqueeze(0))
                    clip_representation_half2_np[i] = output.to("cpu").numpy()[0,:,0,0]
                clip_representation_half2 = clip_representation_half2_np.astype(float)
                    # Save it for next time
                np.save(os.path.join(self.path, self.listGames[index], "2_" + self.features_resnet),clip_representation_half2)

            elif self.args.calibration:
                clip_representation_half1 = (clip_representation_half1.astype(float)/127)-1
                clip_representation_half2 = (clip_representation_half2.astype(float)/127)-1

            clip_representation_half1 = feats2clip(torch.from_numpy(clip_representation_half1), 
                            stride=self.chunk_size-self.receptive_field, 
                            clip_length=self.chunk_size)

            clip_representation_half2 = feats2clip(torch.from_numpy(clip_representation_half2), 
                            stride=self.chunk_size-self.receptive_field, 
                            clip_length=self.chunk_size)


        label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
        label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))


        # Load labels
        if os.path.exists(os.path.join(self.path, self.listGames[index], self.labels)):
            labels = json.load(open(os.path.join(self.path, self.listGames[index], self.labels)))

            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = self.framerate * ( seconds + 60 * minutes ) 

                if event not in self.dict_event:
                    continue
                label = self.dict_event[event]

                value = 1
                if "visibility" in annotation.keys():
                    if annotation["visibility"] == "not shown":
                        value = -1

                if half == 1:
                    frame = min(frame, feat_half1.shape[0]-1)
                    label_half1[frame][label] = value

                if half == 2:
                    frame = min(frame, feat_half2.shape[0]-1)
                    label_half2[frame][label] = value


        feat_half1 = feats2clip(torch.from_numpy(feat_half1), 
                        stride=self.chunk_size-self.receptive_field, 
                        clip_length=self.chunk_size)

        feat_half2 = feats2clip(torch.from_numpy(feat_half2), 
                        stride=self.chunk_size-self.receptive_field, 
                        clip_length=self.chunk_size)

        
        if self.args.backbone_player is None:
            clip_representation_half1 = torch.zeros(feat_half1.shape[0], 1)
            clip_representation_half2 = torch.zeros(feat_half2.shape[0], 1)

        return feat_half1, feat_half2, torch.from_numpy(label_half1), torch.from_numpy(label_half2), clip_representation_half1, clip_representation_half2

    def __len__(self):
        return len(self.listGames)


