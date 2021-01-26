
import ffmpy # Require pip install ffmpy
import os
import json
import configparser

import multiprocessing
import signal
import time
import SoccerNet
from SoccerNet.utils import getListGames

    

def convert_video(input_path, output_path):
    

    try:
        file = input_path.split("/")[-1]
        folder = os.path.dirname(input_path)

        config = configparser.RawConfigParser()
        config.sections()
        config.read(folder + '/video.ini')
            
        start = int(float(config[file]['start_time_second']))
        stop = int(float(config[file]['duration_second']))

    except Exception as e:
        print ("----->   video.ini error in :", input_path, e)


    try:

        ff = ffmpy.FFmpeg(
             inputs={input_path: ""},
             outputs={output_path: '-r 25 -vf scale=-1:224 -ss ' + str(start) + ' -t ' + str(stop)})

        if(not os.path.exists(os.path.dirname(output_path))):
            os.makedirs(os.path.dirname(output_path))
        print(ff.cmd)
        ff.run()
    except Exception as e:
        print ("----->   ffmpeg error with :", input_path ,e)
    

def elaborateVideo(args, half):
    min_size = 50*1024 #in kB
    max_size = 200*1024 #in kB

    RootFolder = args["RootFolder"]
    NewRootFolder = args["NewRootFolder"]
    Game = args["Game"]          
        
    start_time = time.time()
            
    Half_crop224_MKV_file = os.path.join(NewRootFolder, Game, str(half) + ".mkv")
    Half_MKV_file = os.path.join(RootFolder, Game, str(half) + "_HQ.mkv")
    Half_MP4_file = os.path.join(RootFolder, Game, str(half) + "_HQ.mp4")
    Half_TS_file =  os.path.join(RootFolder, Game, str(half) + "_HQ.ts")

    if (not os.path.exists(Half_crop224_MKV_file)):     
        print(Half_crop224_MKV_file)
        if (os.path.exists(Half_MKV_file)):   convert_video(Half_MKV_file, Half_crop224_MKV_file)
        elif (os.path.exists(Half_MP4_file)): convert_video(Half_MP4_file, Half_crop224_MKV_file)
        elif (os.path.exists(Half_TS_file)):  convert_video(Half_TS_file, Half_crop224_MKV_file)
        print("  done", time.time() - start, "s\n")
    else:
        size_kB = os.path.getsize(Half_crop224_MKV_file) // 1024
        if (size_kB<min_size) or (size_kB>max_size):  

            if (os.path.exists(Half_MKV_file)):   print(Half_MKV_file)
            elif (os.path.exists(Half_MP4_file)):   print(Half_MP4_file)
            elif (os.path.exists(Half_TS_file)):   print(Half_TS_file)
            print("  Already Exist  /!\ size looks strange /!\ ", size_kB , "kB!!\n")


        if (size_kB<1000) :  
            os.remove(Half_crop224_MKV_file)

def elaborateGame(args):

    elaborateVideo(args, "1")
    elaborateVideo(args, "2")




RootFolder = "/media/giancos/Football/SoccerNet_test/"
NewRootFolder = "/media/giancos/Football/SoccerNet_test224/"


start = time.time()
cnt = 0
args_list = []

import random
game_list = getListGames(["test"], task="spotting")
game_list = random.sample(game_list, len(game_list)) # shuffle games to run multiple instances of conversions
for game in game_list:
    args = {}
    args["RootFolder"] = RootFolder 
    args["NewRootFolder"] = NewRootFolder
    args["Game"] = game
    args_list.append(args)
    elaborateGame(args)

