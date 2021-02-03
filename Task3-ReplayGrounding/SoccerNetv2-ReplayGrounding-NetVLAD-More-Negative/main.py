import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from dataset import SoccerNetReplayClips, SoccerNetReplayClipsTesting
from model import Model
from train import trainer, test
from loss import SymmetricContextAwareLoss, ReplayGroundingSpottingLoss
from config.classes import K_MATRIX

# for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def main(args):


    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))


    # create dataset
    if not args.test_only:

        dataset_Train = SoccerNetReplayClips(path=args.SoccerNet_path, features=args.features, split="train", framerate=args.framerate, chunk_size=args.chunk_size*args.framerate, receptive_field=args.receptive_field*args.framerate, chunks_per_epoch=args.chunks_per_epoch, replay_size=args.replay_size, hard_negative_weight=args.hard_negative_weight, random_negative_weight=args.random_negative_weight, replay_negative_weight=args.replay_negative_weight, loop=args.loop)
        dataset_Valid = SoccerNetReplayClips(path=args.SoccerNet_path, features=args.features, split="valid", framerate=args.framerate, chunk_size=args.chunk_size*args.framerate, receptive_field=args.receptive_field*args.framerate, chunks_per_epoch=args.chunks_per_epoch,  replay_size=args.replay_size, hard_negative_weight=args.hard_negative_weight, random_negative_weight=args.random_negative_weight, replay_negative_weight=args.replay_negative_weight, loop=args.loop)
        dataset_Valid_metric  = SoccerNetReplayClipsTesting(path=args.SoccerNet_path, features=args.features, split="valid", framerate=args.framerate, chunk_size=args.chunk_size*args.framerate, receptive_field=args.receptive_field*args.framerate, replay_size=args.replay_size)
    dataset_Test  = SoccerNetReplayClipsTesting(path=args.SoccerNet_path, features=args.features, split="test", framerate=args.framerate, chunk_size=args.chunk_size*args.framerate, receptive_field=args.receptive_field*args.framerate, replay_size=args.replay_size)


    # create model  
    model = Model(weights=args.load_weights, input_size=args.num_features, chunk_size=args.chunk_size*args.framerate, dim_capsule=args.dim_capsule, receptive_field=args.receptive_field*args.framerate, framerate=args.framerate, unsimilar_action=args.unsimilar_action, pooling=args.pooling ,replay_size=args.replay_size ).cuda()
    logging.info(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))

    # create dataloader
    if not args.test_only:
        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.max_num_worker, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(dataset_Valid,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True)

        val_metric_loader = torch.utils.data.DataLoader(dataset_Valid_metric,
            batch_size=1, shuffle=False,
            num_workers=1, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset_Test,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)

    # training parameters
    if not args.test_only:
        
        criterion_spotting = ReplayGroundingSpottingLoss(lambda_coord=args.lambda_coord, lambda_noobj=args.lambda_noobj)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, 
                                        betas=(0.9, 0.999), eps=1e-07, 
                                        weight_decay=0, amsgrad=False)


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)

        # start training
        trainer(train_loader, val_loader, val_metric_loader, test_loader, 
                model, optimizer, scheduler,  criterion_spotting, [args.loss_weight_segmentation, args.loss_weight_detection],
                model_name=args.model_name,
                max_epochs=args.max_epochs, evaluation_frequency=args.evaluation_frequency,
                annotation_path=args.SoccerNet_path,detection_path=args.detection_path, save_results=args.save_results)

    # For the best model only
    load_path=os.path.join("models", args.model_name, "model.pth.tar")
    if args.test_only:
        load_path=args.load_weights
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['state_dict'],strict=False)

    average_mAP = test(test_loader, model=model, model_name=args.model_name,split='test',annotation_path=args.SoccerNet_path,detection_path=args.detection_path,save_results=args.save_results)
    logging.info("Best Performance at end of training " + str(average_mAP))
    return average_mAP


if __name__ == '__main__':


    parser = ArgumentParser(description='context aware loss function', formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--SoccerNet_path',   required=False, type=str,   default="/media/giancos/Football/SoccerNet/",     help='Path for SoccerNet' )
    parser.add_argument('--save_results',   required=False, type=bool ,     help='Save the results to Detection_path if it is True' )
    parser.add_argument('--detection_path',   required=False, type=str,   default="/media/giancos/Football/SoccerNet/",     help='Path for Output_results' )
    parser.add_argument('--features',   required=False, type=str,   default="ResNET_PCA512.npy",     help='Video features' )
    parser.add_argument('--max_epochs',   required=False, type=int,   default=1000,     help='Maximum number of epochs' )
    parser.add_argument('--load_weights',   required=False, type=str,   default=None,     help='weights to load' )
    parser.add_argument('--model_name',   required=False, type=str,   default="CALF",     help='named of the model to save' )
    parser.add_argument('--test_only',   required=False, action='store_true',  help='Perform testing only' )
    # parser.add_argument('--feature', dest='feature', action='store_true')

    parser.add_argument('--loop', required=False, type=int,   default=1,     help='Number of negative samples for each replay' )
    parser.add_argument('--replay_size', required=False, type=int,   default=40,     help='Size of replay that is paasing to the model' )
    parser.add_argument('--hard_negative_weight', required=False, type=float,   default=0.5,     help='Weights for negatives from same class ' )
    parser.add_argument('--random_negative_weight', required=False, type=float,   default=0.5,     help='Weights for random negatives' )
    parser.add_argument('--replay_negative_weight', required=False, type=float,   default=0,     help='Weights for negative replays' )
    parser.add_argument('--unsimilar_action', required=False, type=float,   default=0.2,     help='minmum confident score to penalize the model' )
    parser.add_argument('--pooling', required=False, type=str,   default="MAX",     help='Pooling method ' )

    parser.add_argument('--num_features', required=False, type=int,   default=512,     help='Number of input features' )
    parser.add_argument('--chunks_per_epoch', required=False, type=int,   default=6000,     help='Number of chunks per epoch' )
    parser.add_argument('--evaluation_frequency', required=False, type=int,   default=20,     help='Number of chunks per epoch' )
    parser.add_argument('--dim_capsule', required=False, type=int,   default=16,     help='Dimension of the capsule network' )
    parser.add_argument('--framerate', required=False, type=int,   default=2,     help='Framerate of the input features' )
    parser.add_argument('--chunk_size', required=False, type=int,   default=120,     help='Size of the chunk (in seconds)' )
    parser.add_argument('--receptive_field', required=False, type=int,   default=40,     help='Temporal receptive field of the network (in seconds)' )
    parser.add_argument("--lambda_coord", required=False, type=float, default=5.0, help="Weight of the coordinates of the event in the detection loss")
    parser.add_argument("--lambda_noobj", required=False, type=float, default=0.5, help="Weight of the no object detection in the detection loss")
    parser.add_argument("--loss_weight_segmentation", required=False, type=float, default=0.002, help="Weight of the segmentation loss compared to the detection loss")
    parser.add_argument("--loss_weight_detection", required=False, type=float, default=1.0, help="Weight of the detection loss")
    # parser.add_argument("--outpool_size", required=False, type=float, default=30 help="the Size of adaptive Maxpool output")

    parser.add_argument('--batch_size', required=False, type=int,   default=1,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=1e-03, help='Learning Rate' )
    parser.add_argument('--LRe',       required=False, type=float,   default=1e-06, help='Learning Rate end' )
    parser.add_argument('--patience', required=False, type=int,   default=50,     help='Batch size' )

    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=4, help='number of worker to load data')

    parser.add_argument('--logging_dir',       required=False, type=str,   default="log", help='Where to log' )
    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')

    args = parser.parse_args()

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    os.makedirs(args.logging_dir, exist_ok=True)
    log_path = os.path.join(args.logging_dir, datetime.now().strftime('%Y-%m-%d %H-%M-%S.log'))
    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)


    start=time.time()
    logging.info('Starting main function')
    main(args)
    logging.info(f'Total Execution Time is {time.time()-start} seconds')
