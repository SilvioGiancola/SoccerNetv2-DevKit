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
        # self.dict_event = EVENT_DICTIONARY_V2
        # self.num_classes = 17
        self.labels="Labels-v2.json"
        # self.K_parameters = K_V2*args.framerate 
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
        downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[split], verbose=False, randomized=True)
        downloader.downloadGames(files=[f"1_{self.bbox_predictions}", f"2_{self.bbox_predictions}"], split=[split], verbose=False, randomized=True)
        downloader.downloadGames(files=[f"1_{self.calibration_predictions}", f"2_{self.calibration_predictions}"], split=[split], verbose=False, randomized=True)


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
            if self.args.backbone_player == "3DConv":
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
                            representation_half1[i,int(0.025*self.representation_height): int(0.025*self.representation_height) + self.radar_image.shape[0], int(0.025*self.representation_width): int(0.025*self.representation_width) + self.radar_image.shape[1] ] =self.radar_image

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

                            representation_half1[i] = cv2.polylines(representation_half1[i], [pts], True, (255,255,255), 1)

                        # Draw the players
                        for box, color in zip(bbox["bboxes"], bbox["colors"]):
                            projection_point = np.array([int((box[0]+box[2])/2), box[3], 1])
                            projected_point = unproject_image_point(homography, projection_point)
                            radar_point = meter2radar(projected_point, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                            if radar_point[0]+self.size_radar_point//2 < self.representation_width and radar_point[0]-self.size_radar_point//2 >=0 and radar_point[1]+self.size_radar_point//2 < self.representation_height and radar_point[1]-self.size_radar_point//2 >= 0:
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

                # Visualization
                #for i, frame in enumerate(representation_half1):
                #    cv2.imwrite("outputs/test/"+str(i)+".png", frame)
                

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

                            representation_half2[i] = cv2.polylines(representation_half2[i], [pts], True, (255,255,255), 1)

                        # Draw the players
                        for box, color in zip(bbox["bboxes"], bbox["colors"]):
                            projection_point = np.array([int((box[0]+box[2])/2), box[3], 1])
                            projected_point = unproject_image_point(homography, projection_point)
                            radar_point = meter2radar(projected_point, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                            if radar_point[0]+self.size_radar_point//2 < self.representation_width and radar_point[0]-self.size_radar_point//2 >=0 and radar_point[1]+self.size_radar_point//2 < self.representation_height and radar_point[1]-self.size_radar_point//2 >= 0:
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


                # Visualization
                #for i, frame in enumerate(representation_half2):
                #    cv2.imwrite("outputs/test/"+str(i)+".png", frame)

            elif "GCN" in self.args.backbone_player:
                representation_half1 = [] #np.zeros((feat_half1.shape[0], self.representation_height, self.representation_width, self.representation_channel), dtype=np.uint8)
                bbox_predictions = bbox_half1

                # for i, bbox in enumerate(bbox_predictions["predictions"][0:feat_half1.shape[0]]):
                for i in range(feat_half1.shape[0]):
                    Poses = []
                    Edges = []
                    Features = []


                    if i < len(bbox_predictions["predictions"]):
                        bbox = bbox_predictions["predictions"][i]

                        homography = None
                        frame_center_projected = [0,0,1]
                        feat_confidence = 0

                        if self.args.calibration:
                            confidence = calibration_half1["predictions"][i][0]["confidence"]

                            if self.args.calibration_confidence:
                                feat_confidence = confidence
                            if confidence >= self.calibration_threshold:
                                homography = calibration_half1["predictions"][i][0]["homography"]
                                homography = np.reshape(homography, (3,3))
                                homography = homography/homography[2,2]
                                homography = np.linalg.inv(homography)
                                if self.args.calibration_cone:
                                    frame_center_projected = unproject_image_point(homography, np.array([calibration_half1["size"][2]/2,calibration_half1["size"][1]/2,1]))

                        # Features.append( np.array([0,0,0,0]))
                        for i_box, (box, color) in enumerate(zip(bbox["bboxes"], bbox["colors"])):
                            # given a box
                            # get its position (center box)
                            if self.args.calibration and homography is None:
                                continue
                            this_pos = None
                            if self.args.calibration:
                                projection_point = np.array([int((box[0]+box[2])/2), box[3], 1])
                                this_pos = unproject_image_point(homography, projection_point)[:2]
                            else:
                                this_pos = np.array([(box[0] + box[2])/2,  (box[3] + box[3])/2])

                            # loop over the other bxes
                            for j_other_pos, other_pos in enumerate(Poses):
                                if self.args.calibration:
                                    dist = np.linalg.norm(other_pos - this_pos)
                                    if dist < self.max_dist_player:
                                        # continue
                                        Edges.append([i_box, j_other_pos])
                                        Edges.append([j_other_pos, i_box])

                            # keep track of all poses
                            Poses.append(this_pos)

                            # define node feture
                            color = color[:3]# RGB only
                            color[0] = color[0]/127-1
                            color[1] = color[1]/127-1
                            color[2] = color[2]/127-1
                            color.append((box[0] - box[2]) * (box[1] - box[3])/10000) # add area
                            color.append(this_pos[0]/50)# add pos
                            color.append(this_pos[1]/50)# add pos
                            color.append(frame_center_projected[0]/50)# add pos
                            color.append(frame_center_projected[1]/50)# add pos
                            if self.args.calibration_confidence:
                                color.append(feat_confidence)
                            # color.append((box[0] + box[2])/2)# add pos
                            # color.append((box[1] + box[3])/2)# add pos
                            Features.append(color)

                    if (len(Poses) == 0 and len(Features)==0):
                        # print(i, "hello there!")
                        # Edges = []
                        Poses.append( np.array([0,0]) )
                        # Features = []
                        if self.args.calibration_confidence:
                            Features.append( np.array([0,0,0,0,0,0,0,0,0]))
                        else:
                            Features.append( np.array([0,0,0,0,0,0,0,0]))
                        # Features.append( np.array([0,0,0,0,0,0,0,0]))

                    edge_index = torch.tensor(Edges, dtype=torch.long)
                    x = torch.tensor(Features, dtype=torch.float)
                    # data = MyData(x=x, edge_index=edge_index.t().contiguous(), frame=i)
                    data = Data(x=x, edge_index=edge_index.t().contiguous())
                    representation_half1.append(data)
                    

                representation_half2 = [] #np.zeros((feat_half1.shape[0], self.representation_height, self.representation_width, self.representation_channel), dtype=np.uint8)
                bbox_predictions = bbox_half2

                # for i, bbox in enumerate(bbox_predictions["predictions"][0:feat_half2.shape[0]]):
                for i in range(feat_half2.shape[0]):
                    Poses = []
                    Edges = []
                    Features = []


                    if i < len(bbox_predictions["predictions"]):
                        bbox = bbox_predictions["predictions"][i]

                        homography = None
                        frame_center_projected = [0,0,1]
                        feat_confidence=0
                        if self.args.calibration:
                            confidence = calibration_half2["predictions"][i][0]["confidence"]

                            if self.args.calibration_confidence:
                                feat_confidence = confidence
                            if confidence > self.calibration_threshold:
                                homography = calibration_half2["predictions"][i][0]["homography"]
                                homography = np.reshape(homography, (3,3))
                                homography = homography/homography[2,2]
                                homography = np.linalg.inv(homography)
                                if self.args.calibration_cone:
                                    frame_center_projected = unproject_image_point(homography, np.array([calibration_half1["size"][2]/2,calibration_half1["size"][1]/2,1]))

                        for i_box, (box, color) in enumerate(zip(bbox["bboxes"], bbox["colors"])):
                            # given a box
                            # get its position (center box)
                            if self.args.calibration and homography is None:
                                continue
                            this_pos = None
                            if self.args.calibration:
                                projection_point = np.array([int((box[0]+box[2])/2), box[3], 1])
                                this_pos = unproject_image_point(homography, projection_point)[:2]
                            else:
                                this_pos = np.array([(box[0] + box[2])/2,  (box[3] + box[3])/2])

                            # loop over the other bxes
                            for j_other_pos, other_pos in enumerate(Poses):
                                if self.args.calibration:
                                    dist = np.linalg.norm(other_pos - this_pos)
                                    if dist < self.max_dist_player:
                                        # continue
                                        Edges.append([i_box, j_other_pos])
                                        Edges.append([j_other_pos, i_box])

                            # keep track of all poses
                            Poses.append(this_pos)

                            # define node feture
                            color = color[:3]# RGB only
                            color[0] = color[0]/127-1
                            color[1] = color[1]/127-1
                            color[2] = color[2]/127-1
                            color.append((box[0] - box[2]) * (box[1] - box[3])/10000) # add area
                            color.append(this_pos[0]/50)# add pos
                            color.append(this_pos[1]/50)# add pos
                            color.append(frame_center_projected[0]/50)# add pos
                            color.append(frame_center_projected[1]/50)# add pos

                            if self.args.calibration_confidence:
                                color.append(feat_confidence)
                            # color.append((box[0] + box[2])/2)# add pos
                            # color.append((box[1] + box[3])/2)# add pos
                            # print("color",color)
                            Features.append(color)

                    if len(Poses) == 0 and len(Features)==0:
                        # print(i, "hello there!")
                        # Edges = []
                        Poses.append( np.array([0,0]) )
                        # Features = []

                        if self.args.calibration_confidence:
                            Features.append( np.array([0,0,0,0,0,0,0,0,0]))
                        else:
                            Features.append( np.array([0,0,0,0,0,0,0,0]))
                        # Features.append( np.array([0,0,0,0,0,0,0,0]))



                    edge_index = torch.tensor(Edges, dtype=torch.long)
                    x = torch.tensor(Features, dtype=torch.float)
                    
                    # if x.shape[0] == 0:
                        # print(x.shape)
                    # data = MyData(x=x, edge_index=edge_index.t().contiguous(), frame=i)
                    # if x.shape[0] == 0:
                    data = Data(x=x, edge_index=edge_index.t().contiguous())
                        # print(data)
                    representation_half2.append(data)
                    

                if not len(representation_half1) == len(feat_half1):
                    print("representation_half1 diff length than feat_half1")
                if not len(representation_half2) == len(feat_half2):
                    print("representation_half2 diff length than feat_half2")
      

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
            clip_representation = (clip_representation.astype(float)/127)-1
        elif "GCN" in self.args.backbone_player:
            clip_representation = copy.deepcopy(self.game_bbox[game_index][start:start+self.chunk_size])
            if not len(clip_representation) == len(clip_feat):
                print("clip_representation diff length than clip_feat")
            return torch.from_numpy(clip_feat), torch.from_numpy(clip_labels), torch.from_numpy(clip_targets), clip_representation
        elif self.args.backbone_player is None:
            clip_representation = np.zeros((clip_feat.shape[0], 1))

        return torch.from_numpy(clip_feat), torch.from_numpy(clip_labels), torch.from_numpy(clip_targets), torch.from_numpy(clip_representation)

    def __len__(self):
        return self.chunks_per_epoch

def collateGCN(list_of_examples):
    # data_list = [x[0] for x in list_of_examples]
    # tensors = [x[1] for x in list_of_examples]
    
    return torch.stack([x[0] for x in list_of_examples], dim=0), \
            torch.stack([x[1] for x in list_of_examples], dim=0), \
            torch.stack([x[2] for x in list_of_examples], dim=0), \
            Batch.from_data_list([x for b in list_of_examples for x in b[3] ])

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
        # self.dict_event = EVENT_DICTIONARY_V2
        # self.num_classes = 17
        self.labels="Labels-v2.json"
        # self.K_parameters = K_V2*args.framerate
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
        if split == "challenge":
            downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], split=[split], verbose=False, randomized=True)
        else:       
            downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[split], verbose=False, randomized=True)

        downloader.downloadGames(files=[f"1_{self.bbox_predictions}", f"2_{self.bbox_predictions}"], split=[split], verbose=False, randomized=True)
        downloader.downloadGames(files=[f"1_{self.calibration_predictions}", f"2_{self.calibration_predictions}"], split=[split], verbose=False, randomized=True)



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

        if self.args.backbone_player == "3DConv":
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

                        clip_representation_half1[i] = cv2.polylines(clip_representation_half1[i], [pts], True, (255,255,255), 1)

                    # Draw the players
                    for box, color in zip(bbox["bboxes"], bbox["colors"]):
                        projection_point = np.array([int((box[0]+box[2])/2), box[3], 1])
                        projected_point = unproject_image_point(homography, projection_point)
                        radar_point = meter2radar(projected_point, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                        if radar_point[0]+self.size_radar_point//2 < self.representation_width and radar_point[0]-self.size_radar_point//2 >=0 and radar_point[1]+self.size_radar_point//2 < self.representation_height and radar_point[1]-self.size_radar_point//2 >= 0:
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

                        clip_representation_half2[i] = cv2.polylines(clip_representation_half2[i], [pts], True, (255,255,255), 1)

                    # Draw the players
                    for box, color in zip(bbox["bboxes"], bbox["colors"]):
                        projection_point = np.array([int((box[0]+box[2])/2), box[3], 1])
                        projected_point = unproject_image_point(homography, projection_point)
                        radar_point = meter2radar(projected_point, self.dim_terrain, (self.representation_height, self.representation_width, self.representation_channel))
                        if radar_point[0]+self.size_radar_point//2 < self.representation_width and radar_point[0]-self.size_radar_point//2 >=0 and radar_point[1]+self.size_radar_point//2 < self.representation_height and radar_point[1]-self.size_radar_point//2 >= 0:
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

            if self.args.calibration:
                clip_representation_half1 = (clip_representation_half1.astype(float)/127)-1
                clip_representation_half2 = (clip_representation_half2.astype(float)/127)-1

            clip_representation_half1 = feats2clip(torch.from_numpy(clip_representation_half1), 
                            stride=self.chunk_size-self.receptive_field, 
                            clip_length=self.chunk_size)

            clip_representation_half2 = feats2clip(torch.from_numpy(clip_representation_half2), 
                            stride=self.chunk_size-self.receptive_field, 
                            clip_length=self.chunk_size)



        elif "GCN" in self.args.backbone_player:
            clip_representation_half1 = [] #np.zeros((feat_half1.shape[0], self.representation_height, self.representation_width, self.representation_channel), dtype=np.uint8)
            bbox_predictions = bbox_half1

            # for i, bbox in enumerate(bbox_predictions["predictions"][0:feat_half1.shape[0]]):
            for i in range(feat_half1.shape[0]):
                Poses = []
                Edges = []
                Features = []
                if i < len(bbox_predictions["predictions"]):
                    
                    bbox = bbox_predictions["predictions"][i]
                    # Features.append( np.array([0,0,0,0]))
                    homography = None
                    frame_center_projected = [0,0,1]
                    feat_confidence = 0
                    if self.args.calibration:
                        confidence = calibration_half1["predictions"][i][0]["confidence"]
                        if self.args.calibration_confidence:
                            feat_confidence = confidence
                        if confidence > self.calibration_threshold:
                            homography = calibration_half1["predictions"][i][0]["homography"]
                            homography = np.reshape(homography, (3,3))
                            homography = homography/homography[2,2]
                            homography = np.linalg.inv(homography)


                            # frame_top_left_projected = unproject_image_point(homography, np.array([0,0,1]))
                            # frame_top_right_projected = unproject_image_point(homography, np.array([calibration_half1["size"][2],0,1]))
                            # frame_bottom_left_projected = unproject_image_point(homography, np.array([0,calibration_half1["size"][1],1]))
                            if self.args.calibration_cone:
                                frame_center_projected = unproject_image_point(homography, np.array([calibration_half1["size"][2]/2,calibration_half1["size"][1]/2,1]))

                    for i_box, (box, color) in enumerate(zip(bbox["bboxes"], bbox["colors"])):
                        # given a box
                        # get its position (center box)
                        if self.args.calibration and homography is None:
                            continue
                        this_pos = None
                        if self.args.calibration:
                            projection_point = np.array([int((box[0]+box[2])/2), box[3], 1])
                            this_pos = unproject_image_point(homography, projection_point)[:2]
                        else:
                            # continue
                            this_pos = np.array([(box[0] + box[2])/2,  (box[3] + box[3])/2])

                        # loop over the other bxes
                        for j_other_pos, other_pos in enumerate(Poses):
                            if self.args.calibration:
                                dist = np.linalg.norm(other_pos - this_pos)
                                if dist < self.max_dist_player:
                                    # continue
                                    Edges.append([i_box, j_other_pos])
                                    Edges.append([j_other_pos, i_box])

                        # keep track of all poses
                        Poses.append(this_pos)

                        # define node feture
                        color = color[:3]# RGB only
                        color[0] = color[0]/127-1
                        color[1] = color[1]/127-1
                        color[2] = color[2]/127-1
                        color.append((box[0] - box[2]) * (box[1] - box[3])/10000) # add area
                        color.append(this_pos[0]/50)# add pos
                        color.append(this_pos[1]/50)# add pos
                        color.append(frame_center_projected[0]/50)# add pos
                        color.append(frame_center_projected[1]/50)# add pos
                        if self.args.calibration_confidence:
                            color.append(feat_confidence)
                        # color.append((box[0] + box[2])/2)# add pos
                        # color.append((box[1] + box[3])/2)# add pos
                        Features.append(color)

                if len(Poses) == 0 and len(Features)==0:
                    # print(i, "hello there!")
                    # Edges = []
                    Poses.append( np.array([0,0]) )
                    # Features = []

                    if self.args.calibration_confidence:
                        Features.append( np.array([0,0,0,0,0,0,0,0,0]))
                    else:
                        Features.append( np.array([0,0,0,0,0,0,0,0]))
                    # Features.append( np.array([0,0,0,0,0,0,0,0,0]))

                edge_index = torch.tensor(Edges, dtype=torch.long)
                x = torch.tensor(Features, dtype=torch.float)
                # data = MyData(x=x, edge_index=edge_index.t().contiguous(), frame=i)
                data = Data(x=x, edge_index=edge_index.t().contiguous())
                clip_representation_half1.append(data)
                

            clip_representation_half2 = [] #np.zeros((feat_half1.shape[0], self.representation_height, self.representation_width, self.representation_channel), dtype=np.uint8)
            bbox_predictions = bbox_half2

            # for i, bbox in enumerate(bbox_predictions["predictions"][0:feat_half2.shape[0]]):
            for i in range(feat_half2.shape[0]):
                Poses = []
                Edges = []
                Features = []
                if i < len(bbox_predictions["predictions"]):
                    bbox = bbox_predictions["predictions"][i]
                    homography = None
                    frame_center_projected = [0,0,1]
                    feat_confidence=0
                    if self.args.calibration:
                        confidence = calibration_half2["predictions"][i][0]["confidence"]
                        if self.args.calibration_confidence:
                            feat_confidence = confidence
                        if confidence > self.calibration_threshold:
                            homography = calibration_half2["predictions"][i][0]["homography"]
                            homography = np.reshape(homography, (3,3))
                            homography = homography/homography[2,2]
                            homography = np.linalg.inv(homography)
                            if self.args.calibration_cone:
                                frame_center_projected = unproject_image_point(homography, np.array([calibration_half1["size"][2]/2,calibration_half1["size"][1]/2,1]))

                    for i_box, (box, color) in enumerate(zip(bbox["bboxes"], bbox["colors"])):
                        # given a box
                        # get its position (center box)
                        if self.args.calibration and homography is None:
                            continue
                        this_pos = None
                        if self.args.calibration:
                            projection_point = np.array([int((box[0]+box[2])/2), box[3], 1])
                            this_pos = unproject_image_point(homography, projection_point)[:2]
                        else:
                            # continue
                            this_pos = np.array([(box[0] + box[2])/2,  (box[3] + box[3])/2])

                        # loop over the other bxes
                        for j_other_pos, other_pos in enumerate(Poses):
                            if self.args.calibration:
                                dist = np.linalg.norm(other_pos - this_pos)
                                if dist < self.max_dist_player:
                                    # continue
                                    Edges.append([i_box, j_other_pos])
                                    Edges.append([j_other_pos, i_box])

                        # keep track of all poses
                        Poses.append(this_pos)

                        # define node feture
                        color = color[:3]# RGB only
                        color[0] = color[0]/127-1
                        color[1] = color[1]/127-1
                        color[2] = color[2]/127-1
                        color.append((box[0] - box[2]) * (box[1] - box[3])/10000) # add area
                        color.append(this_pos[0]/50)# add pos
                        color.append(this_pos[1]/50)# add pos
                        color.append(frame_center_projected[0]/50)# add pos
                        color.append(frame_center_projected[1]/50)# add pos
                        if self.args.calibration_confidence:
                            color.append(feat_confidence)
                        # color.append((box[0] + box[2])/2)# add pos
                        # color.append((box[1] + box[3])/2)# add pos
                        Features.append(color)

                if len(Poses) == 0 and len(Features)==0:
                    # print(i, "hello there!")
                    # Edges = []
                    Poses.append( np.array([0,0]) )
                    # Features = []
                    if self.args.calibration_confidence:
                        Features.append( np.array([0,0,0,0,0,0,0,0,0]))
                    else:
                        Features.append( np.array([0,0,0,0,0,0,0,0]))



                edge_index = torch.tensor(Edges, dtype=torch.long)
                x = torch.tensor(Features, dtype=torch.float)
                
                # if x.shape[0] == 0:
                    # print(x.shape)
                # data = MyData(x=x, edge_index=edge_index.t().contiguous(), frame=i)
                # if x.shape[0] == 0:
                data = Data(x=x, edge_index=edge_index.t().contiguous())
                    # print(data)
                clip_representation_half2.append(data)
                

            if not len(clip_representation_half1) == len(feat_half1):
                print("representation_half1 diff length than feat_half1")
            if not len(clip_representation_half2) == len(feat_half2):
                print("representation_half2 diff length than feat_half2")


            def feats2clip_PTgeo(feats, stride, clip_length):

                idx = torch.arange(start=0, end=len(feats)-1, step=stride)
                idxs = []
                for i in torch.arange(0, clip_length):
                    idxs.append(idx+i)
                idx = torch.stack(idxs, dim=1)

                idx = idx.clamp(0, len(feats)-1)
                idx[-1] = torch.arange(clip_length)+len(feats)-clip_length

                # print(idx.tolist())
                out = []
                for id_i in idx.tolist():
                    # print(id_i)
                    out.append([feats[i] for i in id_i])
                return out

            clip_representation_half1 = feats2clip_PTgeo(clip_representation_half1, 
                        stride=self.chunk_size-self.receptive_field, 
                        clip_length=self.chunk_size)

            clip_representation_half2 = feats2clip_PTgeo(clip_representation_half2, 
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

def collateGCNTesting(list_of_examples):
    # data_list = [x[0] for x in list_of_examples]
    # tensors = [x[1] for x in list_of_examples]
    
    return torch.stack([x[0] for x in list_of_examples], dim=0), \
            torch.stack([x[1] for x in list_of_examples], dim=0), \
            torch.stack([x[2] for x in list_of_examples], dim=0), \
            torch.stack([x[3] for x in list_of_examples], dim=0), \
            Batch.from_data_list([c for b in list_of_examples for x in b[4] for c in x ]), \
            Batch.from_data_list([c for b in list_of_examples for x in b[5] for c in x ])

if __name__ == "__main__":
    print("hello")
    # dataset_Train = SoccerNetClips(path=args.SoccerNet_path, split="train", args=args)
