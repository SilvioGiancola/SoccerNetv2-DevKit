import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from dataset import SoccerNet, SoccerNetClips, SoccerNetClipsTesting
from model import Model
from train import trainer, test
from loss import SegmentationLoss, SpottingLoss

# for reproducibility
torch.manual_seed(0)
np.random.seed(0)

def main(args, model_save_path):
    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))


    # create dataset
    # dataset_Train = SoccerNet(path=args.SoccerNet_path, split="train", version=args.version, framerate=args.framerate)
    # dataset_Valid = SoccerNet(path=args.SoccerNet_path, split="valid", version=args.version, framerate=args.framerate)
    # dataset_Test  = SoccerNet(path=args.SoccerNet_path, split="test", version=args.version, framerate=args.framerate)
    if not args.test_only:
        dataset_Train = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split="train", version=args.version, framerate=args.framerate, chunk_size=args.chunk_size*args.framerate, receptive_field=args.receptive_field*args.framerate)
        dataset_Valid = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split="valid", version=args.version, framerate=args.framerate, chunk_size=args.chunk_size*args.framerate, receptive_field=args.receptive_field*args.framerate)
    dataset_Test  = SoccerNetClipsTesting(path=args.SoccerNet_path, features=args.features, split="test", version=args.version, framerate=args.framerate, chunk_size=args.chunk_size*args.framerate, receptive_field=args.receptive_field*args.framerate, advanced_test=args.advanced_test)


    # create model  
    model = Model(weights=args.load_weights, chunk_size=args.chunk_size*args.framerate, dim_capsule=args.dim_capsule, receptive_field=args.receptive_field*args.framerate, num_detections=dataset_Test.num_detections, framerate=args.framerate).cuda()
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

    test_loader = torch.utils.data.DataLoader(dataset_Test,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)

    # training parameters
    if not args.test_only:
        criterion_segmentation = SegmentationLoss(K=dataset_Train.K_parameters)
        criterion_spotting = SpottingLoss(lambda_coord=args.lambda_coord, lambda_noobj=args.lambda_noobj)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, 
                                     betas=(0.9, 0.999), eps=1e-08, 
                                     weight_decay=0, amsgrad=False)
        if args.scheduler == "ExponentialDecay":
            scheduler = [args.LR, args.LR/1000]
        elif args.scheduler == "ReduceLRonPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)

        # start training
        trainer(train_loader, val_loader, test_loader, 
                model, optimizer, scheduler, [criterion_segmentation, criterion_spotting], [args.loss_weight_segmentation, args.loss_weight_detection],
                max_epochs=args.max_epochs,
                model_save_path=model_save_path)



    best_model_path = os.path.join(model_save_path, "model.pth.tar")
    # print("loding?")
    if os.path.exists(best_model_path):
        print(f"loading {best_model_path}")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['state_dict'])

    average_mAP = test(test_loader, model, "best",  model_save_path)
    logging.info("Best Performance at end of training " + str(average_mAP))

if __name__ == '__main__':


    parser = ArgumentParser(description='context aware loss function', formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--SoccerNet_path',   required=False, type=str,   default="path/to/SoccerNet/",     help='Path for SoccerNet' )
    parser.add_argument('--features',   required=False, type=str,   default="ResNET_PCA512.npy",     help='Video features' )
    parser.add_argument('--max_epochs',   required=False, type=int,   default=1000,     help='Maximum number of epochs' )
    parser.add_argument('--load_weights',   required=False, type=str,   default=None,     help='weights to load' )
    parser.add_argument('--model_name',   required=False, type=str,   default="CALF",     help='named of the model to save' )
    parser.add_argument('--test_only',   required=False, action='store_true',  help='Perform testing only' )
    parser.add_argument('--advanced_test',   required=False, type=str,   default="abrupt",    help='Perform testing only' )

    parser.add_argument('--version', required=False, type=int,   default=1,     help='Version of the dataset' )
    parser.add_argument('--num_features', required=False, type=int,   default=512,     help='Number of input features' )
    parser.add_argument('--dim_capsule', required=False, type=int,   default=16,     help='Dimension of the capsule network' )
    parser.add_argument('--framerate', required=False, type=int,   default=2,     help='Framerate of the input features' )
    parser.add_argument('--chunk_size', required=False, type=int,   default=120,     help='Size of the chunk (in seconds)' )
    parser.add_argument('--receptive_field', required=False, type=int,   default=40,     help='Temporal receptive field of the network (in seconds)' )
    parser.add_argument('--num_detections', required=False, type=int,   default=5,     help='Maximal number of detections per chunk' )
    parser.add_argument("--lambda_coord", required=False, type=float, default=5.0, help="Weight of the coordinates of the event in the detection loss")
    parser.add_argument("--lambda_noobj", required=False, type=float, default=0.5, help="Weight of the no object detection in the detection loss")
    parser.add_argument("--loss_weight_segmentation", required=False, type=float, default=0.002, help="Weight of the segmentation loss compared to the detection loss")
    parser.add_argument("--loss_weight_detection", required=False, type=float, default=1.0, help="Weight of the detection loss")
    parser.add_argument("--scheduler", required=False, type=str, default="ExponentialDecay", help="define scheduler")

    parser.add_argument('--batch_size', required=False, type=int,   default=1,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=1e-04, help='Learning Rate' )
    parser.add_argument('--patience', required=False, type=int,   default=25,     help='Batch size' )

    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=4, help='number of worker to load data')

    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')

    args = parser.parse_args()

    start_time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    model_save_path = os.path.join("models", args.model_name)
    os.makedirs(model_save_path, exist_ok=True)
    log_path = os.path.join(model_save_path, f"log.txt")

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

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
    main(args, model_save_path)
    logging.info(f'Total Execution Time is {time.time()-start} seconds')
