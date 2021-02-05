import logging
import os
from metrics_visibility_fast import AverageMeter, average_mAP, NMS
import time
from tqdm import tqdm
import torch
import numpy as np
import math
from preprocessing import batch2long, timestamps2long
from json_io import predictions2json
from SoccerNet.Downloader import getListGames

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
            evaluation_frequency=20):

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

        # Remember best loss and save checkpoint
        is_better = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)



        # Save the best model based on loss only if the evaluation frequency too long
        if is_better and evaluation_frequency > max_epochs:
            torch.save(state, best_model_path)

        # Test the model on the validation set
        if epoch % evaluation_frequency == 0 and epoch != 0:
            performance_validation = test(
                val_metric_loader,
                model, 
                model_name)

            
            performance_validation = performance_validation[0]
            logging.info("Validation performance at epoch " + str(epoch+1) + " -> " + str(performance_validation))

            is_better_metric = performance_validation > best_metric
            best_metric = max(performance_validation,best_metric)


            # Save the best model based on metric only if the evaluation frequency is short enough
            if is_better_metric and evaluation_frequency <= max_epochs:
                torch.save(state, best_model_path)
                performance_test = test(
                    test_loader,
                    model, 
                    model_name, save_predictions=True)
                performance_test = performance_test[0]

                logging.info("Test performance at epoch " + str(epoch+1) + " -> " + str(performance_test))

        # Learning rate scheduler update
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
    losses_segmentation = AverageMeter()
    losses_spotting = AverageMeter()

    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()
        
    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=160) as t:
        for i, (feats, labels, targets) in t: 
            # measure data loading time
            data_time.update(time.time() - end)

    
            feats = feats.cuda()
            labels = labels.cuda().float()
            targets = targets.cuda().float()


            feats=feats.unsqueeze(1)

            # compute output
            output_segmentation, output_spotting = model(feats)

            loss_segmentation = criterion[0](labels, output_segmentation) 
            loss_spotting = criterion[1](targets, output_spotting)

            loss = weights[0]*loss_segmentation + weights[1]*loss_spotting

            # measure accuracy and record loss
            losses.update(loss.item(), feats.size(0))
            losses_segmentation.update(loss_segmentation.item(), feats.size(0))
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
            desc += f'Loss Seg {losses_segmentation.avg:.4e} '
            desc += f'Loss Spot {losses_spotting.avg:.4e} '
            t.set_description(desc)

    return losses.avg


def test(dataloader,model, model_name, save_predictions=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    spotting_grountruth = list()
    spotting_grountruth_visibility = list()
    spotting_predictions = list()
    segmentation_predictions = list()

    chunk_size = model.chunk_size
    receptive_field = model.receptive_field

    model.eval()

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
        for i, (feat_half1, feat_half2, label_half1, label_half2) in t:
            data_time.update(time.time() - end)

            feat_half1 = feat_half1.cuda().squeeze(0)
            label_half1 = label_half1.float().squeeze(0)
            feat_half2 = feat_half2.cuda().squeeze(0)
            label_half2 = label_half2.float().squeeze(0)


            feat_half1=feat_half1.unsqueeze(1)
            feat_half2=feat_half2.unsqueeze(1)

            # Compute the output
            output_segmentation_half_1, output_spotting_half_1 = model(feat_half1)
            output_segmentation_half_2, output_spotting_half_2 = model(feat_half2)


            timestamp_long_half_1 = timestamps2long(output_spotting_half_1.cpu().detach(), label_half1.size()[0], chunk_size, receptive_field)
            timestamp_long_half_2 = timestamps2long(output_spotting_half_2.cpu().detach(), label_half2.size()[0], chunk_size, receptive_field)
            segmentation_long_half_1 = batch2long(output_segmentation_half_1.cpu().detach(), label_half1.size()[0], chunk_size, receptive_field)
            segmentation_long_half_2 = batch2long(output_segmentation_half_2.cpu().detach(), label_half2.size()[0], chunk_size, receptive_field)

            spotting_grountruth.append(torch.abs(label_half1))
            spotting_grountruth.append(torch.abs(label_half2))
            spotting_grountruth_visibility.append(label_half1)
            spotting_grountruth_visibility.append(label_half2)
            spotting_predictions.append(timestamp_long_half_1)
            spotting_predictions.append(timestamp_long_half_2)
            segmentation_predictions.append(segmentation_long_half_1)
            segmentation_predictions.append(segmentation_long_half_2)


    # Transformation to numpy for evaluation
    targets_numpy = list()
    closests_numpy = list()
    detections_numpy = list()
    for target, detection in zip(spotting_grountruth_visibility,spotting_predictions):
        target_numpy = target.numpy()
        targets_numpy.append(target_numpy)
        detections_numpy.append(NMS(detection.numpy(), 20*model.framerate))
        closest_numpy = np.zeros(target_numpy.shape)-1
        #Get the closest action index
        for c in np.arange(target_numpy.shape[-1]):
            indexes = np.where(target_numpy[:,c] != 0)[0].tolist()
            if len(indexes) == 0 :
                continue
            indexes.insert(0,-indexes[0])
            indexes.append(2*closest_numpy.shape[0])
            for i in np.arange(len(indexes)-2)+1:
                start = max(0,(indexes[i-1]+indexes[i])//2)
                stop = min(closest_numpy.shape[0], (indexes[i]+indexes[i+1])//2)
                closest_numpy[start:stop,c] = target_numpy[indexes[i],c]
        closests_numpy.append(closest_numpy)

    # Save the predictions to the json format
    if save_predictions:
        list_game = getListGames(dataloader.dataset.split)
        for index in np.arange(len(list_game)):
            predictions2json(detections_numpy[index*2], detections_numpy[(index*2)+1],"outputs/", list_game[index], model.framerate)


    # Compute the performances
    a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = average_mAP(targets_numpy, detections_numpy, closests_numpy, model.framerate)
    
    print("Average mAP: ", a_mAP)
    print("Average mAP visible: ", a_mAP_visible)
    print("Average mAP unshown: ", a_mAP_unshown)
    print("Average mAP per class: ", a_mAP_per_class)
    print("Average mAP visible per class: ", a_mAP_per_class_visible)
    print("Average mAP unshown per class: ", a_mAP_per_class_unshown)

    return a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown