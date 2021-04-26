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
from config.classes import  EVENT_DICTIONARY_V2

from preprocessing import getTimestampTargets



class SoccerNetReplayClips(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split="train", 
                framerate=2, chunk_size=240, receptive_field=80, chunks_per_epoch=6000):
        self.path = path
        self.listGames = getListGames(split)
        self.features = features
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.chunks_per_epoch = chunks_per_epoch
        self.dict_event = EVENT_DICTIONARY_V2
        self.num_action_classes = 17
        self.labels_actions="Labels-v2.json"
        self.labels_replays="Labels-cameras.json"

        #logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        downloader.downloadGames(files=[self.labels_actions, self.labels_replays,
                                        f"1_{self.features}", f"2_{self.features}"], split=[split], verbose=False)


        logging.info("Pre-compute clips")

        clip_feats = []
        clip_labels = []

        self.game_feats = list()
        self.replay_labels = list()
        self.replay_anchors = list()
        self.game_anchors = list()


        game_counter = 0
        for game in tqdm(self.listGames):
            # Load the features
            feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
            feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))



            # load the replay labels
            labels_replays = json.load(open(os.path.join(self.path, game, self.labels_replays)))
            previous_timestamp = 0

            anchors_replay_half1 = list()
            anchors_replay_half2 = list()

            for annotation in labels_replays["annotations"]:

                time = annotation["gameTime"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = framerate * ( seconds + 60 * minutes ) 


                if not "link" in annotation:
                    previous_timestamp = frame
                    continue

                event = annotation["link"]["label"]

                if not event in self.dict_event or int(annotation["link"]["half"]) != half:
                    previous_timestamp = frame
                    continue

                if previous_timestamp == frame:
                    previous_timestamp = frame
                    continue                    

                time_event = annotation["link"]["time"]
                minutes_event = int(time_event[0:2])
                seconds_event = int(time_event[3:])
                frame_event = framerate * ( seconds_event + 60 * minutes_event ) 

                label = self.dict_event[event]
                if half == 1:
                    anchors_replay_half1.append([game_counter,previous_timestamp,frame,frame_event,label])

                if half == 2:
                    anchors_replay_half2.append([game_counter+1,previous_timestamp,frame,frame_event,label])



                previous_timestamp = frame


            # Load action labels
            labels_actions = json.load(open(os.path.join(self.path, game, self.labels_actions)))

            anchors_half1 = list()
            anchors_half2 = list()

            for annotation in labels_actions["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = framerate * ( seconds + 60 * minutes ) 

                if event not in self.dict_event:
                    continue
                label = self.dict_event[event]

                if half == 1:
                    frame = min(frame, feat_half1.shape[0]-1)
                    anchors_half1.append([game_counter,frame,label])

                if half == 2:
                    frame = min(frame, feat_half2.shape[0]-1)
                    anchors_half2.append([game_counter+1,frame,label])

            self.game_feats.append(feat_half1)
            self.game_feats.append(feat_half2)

            self.game_anchors.append(list())
            for i in np.arange(self.num_action_classes):
                self.game_anchors[-1].append(list())
            for anchor in anchors_half1:
                self.game_anchors[-1][anchor[2]].append(anchor)

            self.game_anchors.append(list())
            for i in np.arange(self.num_action_classes):
                self.game_anchors[-1].append(list())
            for anchor in anchors_half2:
                self.game_anchors[-1][anchor[2]].append(anchor)

            for anchor in anchors_replay_half1:
                self.replay_anchors.append(anchor)
            for anchor in anchors_replay_half2:
                self.replay_anchors.append(anchor)
            game_counter = game_counter+2



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels for the segmentation.
            clip_targets (np.array): clip of targets for the spotting.
        """



        # First retrieve a replay with its action
        replay_anchor = self.replay_anchors[index]
        game_index = replay_anchor[0]
        replay_sequence_start = replay_anchor[1]
        replay_sequence_stop = replay_anchor[2]
        event_anchor = replay_anchor[3]
        event_anchor_class = replay_anchor[4]

        TSE_labels = np.arange(self.game_feats[game_index].shape[0])-event_anchor


        # Load the replay chunk
        # [REPLAY_LOADING] THE FOLLOWING SET OF LINES COULD AND SHOULD BE CHANGED
        replay_clip = np.zeros((self.chunk_size,self.game_feats[game_index].shape[-1]),dtype=self.game_feats[game_index].dtype)
        # Make sure that it is not > chunk_size
        replay_chunk_small = self.game_feats[game_index][replay_sequence_start:min(replay_sequence_stop,replay_sequence_start+self.chunk_size)]
        replay_size = len(replay_chunk_small)
        fill_start = 0
        while fill_start + replay_size < self.chunk_size:
            replay_clip[fill_start:fill_start+replay_size] = replay_chunk_small
            fill_start += replay_size
        """
        replay_clip[0:replay_size] = replay_chunk_small
        """

        selection = np.random.randint(0, 2)


        start = 0
        # Load a positive chunk
        if selection == 0:
            shift = np.random.randint(-self.chunk_size+self.receptive_field, -self.receptive_field)
            start = event_anchor + shift

        # Load a negative chunk
        if selection == 1:
            list_events = self.game_anchors[game_index][event_anchor_class]
            selection_2 = np.random.randint(0, 2)
            # Take one of the same class
            if len(list_events) > 0 and selection_2 == 0:
                event_selection = random.randint(0, len(list_events)-1)
                anchor = list_events[event_selection][1]
                shift = np.random.randint(-self.chunk_size+self.receptive_field, -self.receptive_field)
                start = anchor + shift
            # Take one randomly from the game
            else:
                start = random.randint(0, self.game_feats[game_index].shape[0]-1)

        # Make sure that the chunk does not go out of the video.
        if start < 0:
            start = 0
        if start+self.chunk_size >= self.game_feats[game_index].shape[0]:
            start = self.game_feats[game_index].shape[0]-self.chunk_size-1



        clip = self.game_feats[game_index][start:start+self.chunk_size]
        clip_TSE = TSE_labels[start:start+self.chunk_size]
        clip_target = getTimestampTargets(clip_TSE)

        clip_stacked = np.stack([clip, replay_clip], axis=0)

        return torch.from_numpy(clip_stacked), torch.from_numpy(np.expand_dims(clip_TSE, axis=-1)), torch.from_numpy(clip_target)



    def __len__(self):
        return len(self.replay_anchors)


class SoccerNetReplayClipsTesting(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split="test", 
                framerate=2, chunk_size=240, receptive_field=80):
        self.path = path
        self.listGames = getListGames(split)
        self.features = features
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.dict_event = EVENT_DICTIONARY_V2
        self.num_action_classes = 17
        self.labels_actions="Labels-v2.json"
        self.labels_replays="Labels-cameras.json"
        if split=="challenge":
            self.labels_replays="Labels-replays.json"
        self.framerate = framerate

        #logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        downloader.downloadGames(files=[self.labels_actions, self.labels_replays,
                                        f"1_{self.features}", f"2_{self.features}"], split=[split], verbose=False)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            feat_half2 (np.array): features for the 2nd half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
            label_half2 (np.array): labels (one-hot) for the 2nd half.
        """

        # Load features
        feat_half1 = np.load(os.path.join(self.path, self.listGames[index], "1_" + self.features))
        feat_half2 = np.load(os.path.join(self.path, self.listGames[index], "2_" + self.features))

        # load the replay labels
        labels_replays = json.load(open(os.path.join(self.path, self.listGames[index], self.labels_replays)))
        
        previous_timestamp = 0

        anchors_replay_half1 = list()
        anchors_replay_half2 = list()

        for annotation in labels_replays["annotations"]:

            time = annotation["gameTime"]

            half = int(time[0])

            minutes = int(time[-5:-3])
            seconds = int(time[-2::])
            frame = self.framerate * ( seconds + 60 * minutes ) 


            if not "link" in annotation:
                previous_timestamp = frame
                continue

            if previous_timestamp == frame:
                previous_timestamp = frame
                continue
            if split=="challenge":
                event="Replay"
                frame_event=0
            else:
                event = annotation["link"]["label"]
                if not event in self.dict_event or int(annotation["link"]["half"])!=half:
                    previous_timestamp = frame
                    continue

                     
                time_event = annotation["link"]["time"]
                minutes_event = int(time_event[0:2])
                seconds_event = int(time_event[3:])
                frame_event = self.framerate * ( seconds_event + 60 * minutes_event ) 


            label = self.dict_event[event]
            if half == 1:
                anchors_replay_half1.append([0,previous_timestamp,frame,frame_event,label])

            if half == 2:
                anchors_replay_half2.append([1,previous_timestamp,frame,frame_event,label])


            previous_timestamp = frame


        def feats2clip(feats, stride, clip_length, padding = "replicate_last"):

            if padding =="zeropad":
                print("beforepadding", feats.shape)
                pad = feats.shape[0] - int(feats.shape[0]/stride)*stride
                print("pad need to be", clip_length-pad)
                m = torch.nn.ZeroPad2d((0, 0, clip_length-pad, 0))
                feats = m(feats)
                print("afterpadding", feats.shape)
                # nn.ZeroPad2d(2)

            idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
            idxs = []
            for i in torch.arange(0, clip_length):
                idxs.append(idx+i)
            idx = torch.stack(idxs, dim=1)

            if padding=="replicate_last":
                idx = idx.clamp(0, feats.shape[0]-1)
                # Not replicate last, but take the clip closest to the end of the video
                idx[-1] = torch.arange(clip_length)+feats.shape[0]-clip_length

            return feats[idx,:]
            
        clip_replay_half1 = list()
        label_half1 = list()
        label_replay_half1 = list()
        replay_name_half1=list()
        for anchor in anchors_replay_half1:
            replay_sequence_start = anchor[1]
            replay_sequence_stop = anchor[2]
            event_anchor = anchor[3]

            #  [REPLAY_LOADING] THE FOLLOWING SET OF LINES COULD AND SHOULD BE CHANGED
            replay_clip = np.zeros((self.chunk_size,feat_half1.shape[-1]),dtype=feat_half1.dtype)
            # Make sure that it is not > chunk_size
            replay_chunk_small = feat_half1[replay_sequence_start:min(replay_sequence_stop,replay_sequence_start+self.chunk_size)]
            replay_size = len(replay_chunk_small)        
            fill_start = 0
            while fill_start + replay_size < self.chunk_size:
                replay_clip[fill_start:fill_start+replay_size] = replay_chunk_small
                fill_start += replay_size
            """
            replay_clip[0:replay_size] = replay_chunk_small
            """
            clip_replay_half1.append(replay_clip)
            
            label = np.zeros((feat_half1.shape[0],1))
            label[event_anchor] = 1
            label_half1.append(label)
            replay_name_half1.append(np.concatenate(([index],[anchor[1]],[anchor[2]]), axis=0, out=None))
            label_replay = np.zeros((feat_half1.shape[0],1))
            label_replay[replay_sequence_start:replay_sequence_stop] = 1
            label_replay_half1.append(label_replay)


        clip_replay_half2 = list()
        label_half2 = list()
        label_replay_half2 = list()
        replay_name_half2=list()
        for anchor in anchors_replay_half2:
            replay_sequence_start = anchor[1]
            replay_sequence_stop = anchor[2]
            event_anchor = anchor[3]

            #  [REPLAY_LOADING]  THE FOLLOWING SET OF LINES COULD AND SHOULD BE CHANGED
            replay_clip = np.zeros((self.chunk_size,feat_half2.shape[-1]),dtype=feat_half2.dtype)
            # Make sure that it is not > chunk_size
            replay_chunk_small = feat_half2[replay_sequence_start:min(replay_sequence_stop,replay_sequence_start+self.chunk_size)]
            replay_size = len(replay_chunk_small)
            fill_start = 0
            while fill_start + replay_size < self.chunk_size:
                replay_clip[fill_start:fill_start+replay_size] = replay_chunk_small
                fill_start += replay_size
            """
            replay_clip[0:replay_size] = replay_chunk_small
            """
            clip_replay_half2.append(replay_clip)
            
            label = np.zeros((feat_half2.shape[0],1))
            label[event_anchor] = 1
            label_half2.append(label)
            replay_name_half2.append(np.concatenate(([index],[anchor[1]],[anchor[2]]), axis=0, out=None))
            label_replay = np.zeros((feat_half2.shape[0],1))
            label_replay[replay_sequence_start:replay_sequence_stop] = 1
            label_replay_half2.append(label_replay)

        
        clip_replay_half1 = np.array(clip_replay_half1)
        clip_replay_half2 = np.array(clip_replay_half2)


        feat_half1 = feats2clip(torch.from_numpy(feat_half1), 
                        stride=self.chunk_size-self.receptive_field, 
                        clip_length=self.chunk_size)

        feat_half2 = feats2clip(torch.from_numpy(feat_half2), 
                        stride=self.chunk_size-self.receptive_field, 
                        clip_length=self.chunk_size)

        return feat_half1, feat_half2, torch.from_numpy(clip_replay_half1), torch.from_numpy(clip_replay_half2), label_half1, label_half2, label_replay_half1, label_replay_half2 ,replay_name_half1,replay_name_half2

    def __len__(self):
        return len(self.listGames)
