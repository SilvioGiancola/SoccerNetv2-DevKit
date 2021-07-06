import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from dataset import SoccerNetClipsTesting
from model import ContextAwareModel
from train import test

# Fixing seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

def main(args):

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))


    dataset_Test = SoccerNetClipsTesting(path=args.video_path, features=args.features, framerate=args.framerate, chunk_size=args.chunk_size*args.framerate, receptive_field=args.receptive_field*args.framerate)


    # Create the deep learning model
    model = ContextAwareModel(weights=args.load_weights, input_size=args.num_features, num_classes=dataset_Test.num_classes, chunk_size=args.chunk_size*args.framerate, dim_capsule=args.dim_capsule, receptive_field=args.receptive_field*args.framerate, num_detections=dataset_Test.num_detections, framerate=args.framerate).cuda()
    # Logging information about the model
    logging.info(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))


    test_loader = torch.utils.data.DataLoader(dataset_Test,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)

    
    # Load the best model and compute its performance
    checkpoint = torch.load(os.path.join("models", args.model_name, "model.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'])

    test(test_loader, model=model, model_name=args.model_name, save_predictions=True)







if __name__ == '__main__':

    # Load the arguments
    parser = ArgumentParser(description='context aware loss function', formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--video_path',   required=True, type=str, help='Path to the SoccerNet-V2 dataset folder' )
    parser.add_argument('--features',   required=False, type=str,   default="ResNET_PCA512.npy",     help='Video features' )
    parser.add_argument('--max_epochs',   required=False, type=int,   default=1000,     help='Maximum number of epochs' )
    parser.add_argument('--load_weights',   required=False, type=str,   default=None,     help='weights to load' )
    parser.add_argument('--model_name',   required=False, type=str,   default="CALF",     help='named of the model to save' )
    parser.add_argument('--test_only',   required=False, action='store_true',  help='Perform testing only' )
    parser.add_argument('--challenge',   required=False, action='store_true',  help='Perform evaluations on the challenge set to produce json files' )

    parser.add_argument('--K_params', required=False, type=type(torch.FloatTensor),   default=None,     help='K_parameters' )
    parser.add_argument('--num_features', required=False, type=int,   default=512,     help='Number of input features' )
    parser.add_argument('--chunks_per_epoch', required=False, type=int,   default=18000,     help='Number of chunks per epoch' )
    parser.add_argument('--evaluation_frequency', required=False, type=int,   default=20,     help='Number of chunks per epoch' )
    parser.add_argument('--dim_capsule', required=False, type=int,   default=16,     help='Dimension of the capsule network' )
    parser.add_argument('--framerate', required=False, type=int,   default=2,     help='Framerate of the input features' )
    parser.add_argument('--chunk_size', required=False, type=int,   default=120,     help='Size of the chunk (in seconds)' )
    parser.add_argument('--receptive_field', required=False, type=int,   default=40,     help='Temporal receptive field of the network (in seconds)' )
    parser.add_argument("--lambda_coord", required=False, type=float, default=5.0, help="Weight of the coordinates of the event in the detection loss")
    parser.add_argument("--lambda_noobj", required=False, type=float, default=0.5, help="Weight of the no object detection in the detection loss")
    parser.add_argument("--loss_weight_segmentation", required=False, type=float, default=0.000367, help="Weight of the segmentation loss compared to the detection loss")
    parser.add_argument("--loss_weight_detection", required=False, type=float, default=1.0, help="Weight of the detection loss")

    parser.add_argument('--batch_size', required=False, type=int,   default=32,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=1e-03, help='Learning Rate' )
    parser.add_argument('--patience', required=False, type=int,   default=25,     help='Patience before reducing LR (ReduceLROnPlateau)' )

    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=4, help='number of worker to load data')

    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')

    args = parser.parse_args()


    # Logging information
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    os.makedirs(os.path.join("models", args.model_name), exist_ok=True)
    log_path = os.path.join("models", args.model_name,
                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    # Setup the GPU
    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)


    # Start the main training function
    start=time.time()
    logging.info('Starting main function')
    main(args)
    logging.info(f'Total Execution Time is {time.time()-start} seconds')
