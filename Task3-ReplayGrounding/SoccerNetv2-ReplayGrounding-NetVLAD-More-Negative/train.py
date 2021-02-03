import logging
import os
import time
from tqdm import tqdm
import torch
import numpy as np
import math
from SoccerNet.Downloader import getListGames
from visualization import visualize
from SoccerNet.Evaluation.ReplayGrounding import evaluate, average_mAP, game_results_to_json


def trainer(train_loader,
            val_loader,
            val_metric_loader,
            test_loader,
            model,
            optimizer,
            scheduler,
            criterion,
            weights,
            model_name,
            max_epochs=1000,
            evaluation_frequency=20,
            outpool_size=20,
            annotation_path='',
            detection_path='',
            save_results=False
            ):

    logging.info("start training")

    best_loss = 9e99
    best_metric = -1

    for epoch in range(max_epochs):
        best_model_path = os.path.join("models", model_name, "model.pth.tar")


        # train for one epoch
        loss_training = train(
            train_loader,
            model,
            criterion,
            weights,
            optimizer,
            epoch + 1,
            train = True)

        # evaluate on validation set
        loss_validation = train(
            val_loader,
            model,
            criterion,
            weights,
            optimizer,
            epoch + 1,
            train = False)

        

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }
        os.makedirs(os.path.join("models", model_name), exist_ok=True)
        # torch.save(
        #     state,
        #     os.path.join("models", model_name,
        #                  "model_epoch" + str(epoch + 1) + ".pth.tar"))

        # remember best prec@1 and save checkpoint
        is_better = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)



        # Save the best model based on loss only if the evaluation frequency too long
        if is_better and evaluation_frequency > 50:
            torch.save(state, best_model_path)

        # Test the model on the validation set
        if epoch % evaluation_frequency == 0  and epoch !=0 :
            performance_validation = test(
                val_metric_loader,
                model, 
                model_name,
        "valid",
        annotation_path,
        detection_path, 
        save_results)

            logging.info("Validation performance at epoch " + str(epoch+1) + " -> " + str(performance_validation))

            is_better_metric = performance_validation > best_metric
            best_metric = max(performance_validation,best_metric)


            # Save the best model based on metric only if the evaluation frequency is short enough
            if is_better_metric and evaluation_frequency <= 50:
                torch.save(state, best_model_path)
                performance_test = test(
                    test_loader,
                    model, 
                    model_name,
        "test",
        annotation_path,
        detection_path, 
        save_results)

                logging.info("Test performance at epoch " + str(epoch+1) + " -> " + str(performance_test))

        if scheduler is not None:
            prevLR = optimizer.param_groups[0]['lr']
            scheduler.step(loss_validation)
            currLR = optimizer.param_groups[0]['lr']
            if (currLR is not prevLR and scheduler.num_bad_epochs == 0):
                logging.info("Plateau Reached!")

            if (prevLR < 2 * scheduler.eps and
                    scheduler.num_bad_epochs >= scheduler.patience):
                logging.info(
                    "Plateau Reached and no more reduction -> Exiting Loop")
                break
        else:
            current_learning_rate = optimizer.param_groups[0]['lr']
            new_learning_rate = current_learning_rate * 0.993116#- (scheduler[0]-scheduler[1])/max_epochs# * 0.993116
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_learning_rate

            print(new_learning_rate)

        """

        """
    return

def train(dataloader,
          model,
          criterion, 
          weights,
          optimizer,
          epoch,
          train=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # losses_segmentation = AverageMeter()
    losses_spotting = AverageMeter()

    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()
        
    end = time.time()
    loop=10
    
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=160) as t:
        for i, (feats_all, feats_replay_all, targets_all, replay_masks_all) in t: 
            # measure data loading time
            data_time.update(time.time() - end)
            for (feats, feats_replay, targets, replay_masks) in zip(feats_all, feats_replay_all, targets_all, replay_masks_all):
                
            # print(feats.size(),feats_replay.size())
                feats = feats.cuda()
                feats_replay = feats_replay.cuda()
                targets = targets.cuda().float()

                # compute output

                output_spotting = model(feats,feats_replay, replay_masks).cuda()

                # loss_segmentation = criterion[0](labels, output_segmentation) 
                # print(targets.shape,'targets')
                # print(targets)
                # print(output_spotting,'detection')
                # print(output_spotting.shape)
                loss_spotting = criterion(targets, output_spotting)

                loss = loss_spotting

                # measure accuracy and record loss
                losses.update(loss.item(), feats.size(0))
                # losses_segmentation.update(loss_segmentation.item(), feats.size(0))
                losses_spotting.update(loss_spotting.item(), feats.size(0))

                if train:
                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if train:
                desc = f'Train {epoch}: '
            else:
                desc = f'Evaluate {epoch}: '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            desc += f'Loss {losses.avg:.4e} '
            # desc += f'Loss Seg {losses_segmentation.avg:.4e} '
            desc += f'Loss Spot {losses_spotting.avg:.4e} '
            t.set_description(desc)

    return losses.avg


def test(dataloader,model, model_name,split,annotation_path,detection_path,save_results):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    spotting_grountruth = list()
    spotting_predictions = list()
    segmentation_predictions = list()
    replay_grountruth = list()

    chunk_size = model.chunk_size
    receptive_field = model.receptive_field

    model.eval()

    def timestamps2long(output_spotting, video_size, chunk_size, receptive_field):

        start = 0
        last = False
        receptive_field = receptive_field//2

        timestamps_long = torch.zeros([video_size,1], dtype = torch.float, device=output_spotting.device)-1


        for batch in np.arange(output_spotting.size()[0]):

            tmp_timestamps = torch.zeros([chunk_size,1], dtype = torch.float, device=output_spotting.device)-1
            
            tmp_timestamps[torch.floor(output_spotting[batch,0,1]*(chunk_size-1)).type(torch.int) , 0 ] = output_spotting[batch,0,0]

            # ------------------------------------------
            # Store the result of the chunk in the video
            # ------------------------------------------
            if video_size <= chunk_size:
                timestamps_long = tmp_timestamps[0:video_size]
                break

            # For the first chunk
            if start == 0:
                timestamps_long[0:chunk_size-receptive_field] = tmp_timestamps[0:chunk_size-receptive_field]

            # For the last chunk
            elif last:
                timestamps_long[start+receptive_field:start+chunk_size] = tmp_timestamps[receptive_field:]
                break

            # For every other chunk
            else:
                timestamps_long[start+receptive_field:start+chunk_size-receptive_field] = tmp_timestamps[receptive_field:chunk_size-receptive_field]
            
            # ---------------
            # Loop Management
            # ---------------

            # Update the index
            start += chunk_size - 2 * receptive_field
            # Check if we are at the last index of the game
            if start + chunk_size >= video_size:
                start = video_size - chunk_size 
                last = True
        return timestamps_long

    def batch2long(output_segmentation, video_size, chunk_size, receptive_field):

        start = 0
        last = False
        receptive_field = receptive_field//2

        segmentation_long = torch.zeros([video_size,1], dtype = torch.float, device=output_segmentation.device)


        for batch in np.arange(output_segmentation.size()[0]):

            tmp_segmentation = 1-output_segmentation[batch]


            # ------------------------------------------
            # Store the result of the chunk in the video
            # ------------------------------------------

            # For the first chunk
            if start == 0:
                segmentation_long[0:chunk_size-receptive_field] = tmp_segmentation[0:chunk_size-receptive_field]

            # For the last chunk
            elif last:
                segmentation_long[start+receptive_field:start+chunk_size] = tmp_segmentation[receptive_field:]
                break

            # For every other chunk
            else:
                segmentation_long[start+receptive_field:start+chunk_size-receptive_field] = tmp_segmentation[receptive_field:chunk_size-receptive_field]
            
            # ---------------
            # Loop Management
            # ---------------

            # Update the index
            start += chunk_size - 2 * receptive_field
            # Check if we are at the last index of the game
            if start + chunk_size >= video_size:
                start = video_size - chunk_size 
                last = True
        return segmentation_long

    end = time.time()

    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
        for i, (feat_half1, feat_half2, replay_half1, replay_half2, label_half1, label_half2, label_replay_half1, label_replay_half2, mask_replay_half1, mask_replay_half2,replay_name_half1,replay_name_half2) in t:
            data_time.update(time.time() - end)

            replay_half1 = replay_half1.cuda().squeeze(0)
            replay_half2 = replay_half2.cuda().squeeze(0)
            feat_half1 = feat_half1.cuda().squeeze(0)
            feat_half2 = feat_half2.cuda().squeeze(0)
            feat_half1=feat_half1.unsqueeze(1)
            feat_half2=feat_half2.unsqueeze(1)
  


            detection_half1 = list()
            replay_names_half1 = list()
            for replay, label, label_replay, mask_replay,replay_name in zip(replay_half1, label_half1, label_replay_half1,mask_replay_half1,replay_name_half1):
                label = label.float().squeeze(0)
                label_replay = label_replay.float().squeeze(0)
                output_spotting_half_1 = model( feat_half1,replay.unsqueeze(0),mask_replay)
                timestamp_long_half_1 = timestamps2long(output_spotting_half_1.cpu().detach(), label.size()[0], chunk_size, receptive_field)
                detection_half1.append(timestamp_long_half_1)
                
                replay_names_half1.append(replay_name)
                spotting_grountruth.append(label)
                spotting_predictions.append(timestamp_long_half_1)
                replay_grountruth.append(label_replay)

            detection_half2 = list()
            replay_names_half2 = list()
            for replay, label, label_replay, mask_replay, replay_name in zip(replay_half2, label_half2, label_replay_half2, mask_replay_half2,replay_name_half2):
                label = label.float().squeeze(0)
                label_replay = label_replay.float().squeeze(0)
                output_spotting_half_2 = model( feat_half2,replay.unsqueeze(0),mask_replay)
                timestamp_long_half_2 = timestamps2long(output_spotting_half_2.cpu().detach(), label.size()[0], chunk_size, receptive_field)
                detection_half2.append(timestamp_long_half_2)
                
                replay_names_half2.append(replay_name)
                spotting_grountruth.append(label)
                spotting_predictions.append(timestamp_long_half_2)
                replay_grountruth.append(label_replay)
            if save_results:
                game_results_to_json(detection_path,split,detection_half1,detection_half2,replay_names_half1,replay_names_half2,model.framerate,timestamp_long_half_1.shape[0],timestamp_long_half_2.shape[0])



        #visualize(spotting_grountruth ,spotting_predictions,segmentation_predictions, replay_grountruth)
        
        if not save_results:
            a_AP = average_mAP(spotting_grountruth, spotting_predictions, model.framerate)
            print("a-AP: ", a_AP)
        if save_results:
            results=evaluate(annotation_path,detection_path,"Detection-replays.json",split)
            print("a_AP: ",results["a_AP"])
    return results["a_AP"]



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count