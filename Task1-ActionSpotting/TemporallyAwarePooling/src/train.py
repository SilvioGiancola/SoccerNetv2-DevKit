import logging
import os
import zipfile
import sys
import json
import time
from tqdm import tqdm
import torch
import numpy as np

import sklearn
import sklearn.metrics
from sklearn.metrics import average_precision_score
from SoccerNet.Evaluation.ActionSpotting import evaluate
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1




def trainer(train_loader,
            val_loader,
            val_metric_loader,
            model,
            optimizer,
            scheduler,
            criterion,
            model_name,
            max_epochs=1000,
            evaluation_frequency=20):

    logging.info("start training")

    best_loss = 9e99

    for epoch in range(max_epochs):
        best_model_path = os.path.join("models", model_name, "model.pth.tar")

        # train for one epoch
        loss_training = train(train_loader, model, criterion,
                              optimizer, epoch + 1, train=True)

        # evaluate on validation set
        loss_validation = train(
            val_loader, model, criterion, optimizer, epoch + 1, train=False)

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }
        os.makedirs(os.path.join("models", model_name), exist_ok=True)

        # remember best prec@1 and save checkpoint
        is_better = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)

        # Save the best model based on loss only if the evaluation frequency too long
        if is_better:
            torch.save(state, best_model_path)

        # Test the model on the validation set
        if epoch % evaluation_frequency == 0 and epoch != 0:
            performance_validation = test(
                val_metric_loader,
                model,
                model_name)

            logging.info("Validation performance at epoch " +
                         str(epoch+1) + " -> " + str(performance_validation))

        # Reduce LR on Plateau after patience reached
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
          optimizer,
          epoch,
          train=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, (feats, labels) in t:
            # measure data loading time
            data_time.update(time.time() - end)
            feats = feats.cuda()
            labels = labels.cuda()
            # compute output
            output = model(feats)

            # hand written NLL criterion
            loss = criterion(labels, output)

            # measure accuracy and record loss
            losses.update(loss.item(), feats.size(0))

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
            t.set_description(desc)

    return losses.avg


def test(dataloader, model, model_name):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()

    end = time.time()
    all_labels = []
    all_outputs = []
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, (feats, labels) in t:
            # measure data loading time
            data_time.update(time.time() - end)
            feats = feats.cuda()
            # labels = labels.cuda()

            # print(feats.shape)
            # feats=feats.unsqueeze(0)
            # print(feats.shape)

            # compute output
            output = model(feats)

            all_labels.append(labels.detach().numpy())
            all_outputs.append(output.cpu().detach().numpy())

            batch_time.update(time.time() - end)
            end = time.time()

            desc = f'Test (cls): '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            t.set_description(desc)

    AP = []
    for i in range(1, dataloader.dataset.num_classes+1):
        AP.append(average_precision_score(np.concatenate(all_labels)
                                          [:, i], np.concatenate(all_outputs)[:, i]))

    # t.set_description()
    # print(AP)
    mAP = np.mean(AP)
    print(mAP, AP)

    return mAP

def testSpotting(dataloader, model, model_name, overwrite=True, NMS_window=30, NMS_threshold=0.5):
    
    split = '_'.join(dataloader.dataset.split)
    # print(split)
    output_results = os.path.join("models", model_name, f"results_spotting_{split}.zip")
    output_folder = f"outputs_{split}"


    if not os.path.exists(output_results) or overwrite:
        batch_time = AverageMeter()
        data_time = AverageMeter()

        spotting_grountruth = list()
        spotting_grountruth_visibility = list()
        spotting_predictions = list()

        model.eval()

        count_visible = torch.FloatTensor([0.0]*dataloader.dataset.num_classes)
        count_unshown = torch.FloatTensor([0.0]*dataloader.dataset.num_classes)
        count_all = torch.FloatTensor([0.0]*dataloader.dataset.num_classes)

        end = time.time()
        with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
            for i, (game_ID, feat_half1, feat_half2, label_half1, label_half2) in t:
                data_time.update(time.time() - end)

                # Batch size of 1
                game_ID = game_ID[0]
                feat_half1 = feat_half1.squeeze(0)
                label_half1 = label_half1.float().squeeze(0)
                feat_half2 = feat_half2.squeeze(0)
                label_half2 = label_half2.float().squeeze(0)

                # Compute the output for batches of frames
                BS = 256
                timestamp_long_half_1 = []
                for b in range(int(np.ceil(len(feat_half1)/BS))):
                    start_frame = BS*b
                    end_frame = BS*(b+1) if BS * \
                        (b+1) < len(feat_half1) else len(feat_half1)
                    feat = feat_half1[start_frame:end_frame].cuda()
                    output = model(feat).cpu().detach().numpy()
                    timestamp_long_half_1.append(output)
                timestamp_long_half_1 = np.concatenate(timestamp_long_half_1)

                timestamp_long_half_2 = []
                for b in range(int(np.ceil(len(feat_half2)/BS))):
                    start_frame = BS*b
                    end_frame = BS*(b+1) if BS * \
                        (b+1) < len(feat_half2) else len(feat_half2)
                    feat = feat_half2[start_frame:end_frame].cuda()
                    output = model(feat).cpu().detach().numpy()
                    timestamp_long_half_2.append(output)
                timestamp_long_half_2 = np.concatenate(timestamp_long_half_2)


                timestamp_long_half_1 = timestamp_long_half_1[:, 1:]
                timestamp_long_half_2 = timestamp_long_half_2[:, 1:]

                spotting_grountruth.append(torch.abs(label_half1))
                spotting_grountruth.append(torch.abs(label_half2))
                spotting_grountruth_visibility.append(label_half1)
                spotting_grountruth_visibility.append(label_half2)
                spotting_predictions.append(timestamp_long_half_1)
                spotting_predictions.append(timestamp_long_half_2)

                batch_time.update(time.time() - end)
                end = time.time()

                desc = f'Test (spot.): '
                desc += f'Time {batch_time.avg:.3f}s '
                desc += f'(it:{batch_time.val:.3f}s) '
                desc += f'Data:{data_time.avg:.3f}s '
                desc += f'(it:{data_time.val:.3f}s) '
                t.set_description(desc)



                def get_spot_from_NMS(Input, window=60, thresh=0.0):

                    detections_tmp = np.copy(Input)
                    indexes = []
                    MaxValues = []
                    while(np.max(detections_tmp) >= thresh):

                        # Get the max remaining index and value
                        max_value = np.max(detections_tmp)
                        max_index = np.argmax(detections_tmp)
                        MaxValues.append(max_value)
                        indexes.append(max_index)
                        # detections_NMS[max_index,i] = max_value

                        nms_from = int(np.maximum(-(window/2)+max_index,0))
                        nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)))
                        detections_tmp[nms_from:nms_to] = -1

                    return np.transpose([indexes, MaxValues])

                framerate = dataloader.dataset.framerate
                get_spot = get_spot_from_NMS

                json_data = dict()
                json_data["UrlLocal"] = game_ID
                json_data["predictions"] = list()

                for half, timestamp in enumerate([timestamp_long_half_1, timestamp_long_half_2]):
                    for l in range(dataloader.dataset.num_classes):
                        spots = get_spot(
                            timestamp[:, l], window=NMS_window*framerate, thresh=NMS_threshold)
                        for spot in spots:
                            # print("spot", int(spot[0]), spot[1], spot)
                            frame_index = int(spot[0])
                            confidence = spot[1]
                            # confidence = predictions_half_1[frame_index, l]

                            seconds = int((frame_index//framerate)%60)
                            minutes = int((frame_index//framerate)//60)

                            prediction_data = dict()
                            prediction_data["gameTime"] = str(half+1) + " - " + str(minutes) + ":" + str(seconds)
                            if dataloader.dataset.version == 2:
                                prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[l]
                            else:
                                prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V1[l]
                            prediction_data["position"] = str(int((frame_index/framerate)*1000))
                            prediction_data["half"] = str(half+1)
                            prediction_data["confidence"] = str(confidence)
                            json_data["predictions"].append(prediction_data)
                
                os.makedirs(os.path.join("models", model_name, output_folder, game_ID), exist_ok=True)
                with open(os.path.join("models", model_name, output_folder, game_ID, "results_spotting.json"), 'w') as output_file:
                    json.dump(json_data, output_file, indent=4)


        def zipResults(zip_path, target_dir, filename="results_spotting.json"):            
            zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
            rootlen = len(target_dir) + 1
            for base, dirs, files in os.walk(target_dir):
                for file in files:
                    if file == filename:
                        fn = os.path.join(base, file)
                        zipobj.write(fn, fn[rootlen:])

        # zip folder
        zipResults(zip_path = output_results,
                target_dir = os.path.join("models", model_name, output_folder),
                filename="results_spotting.json")

    if split == "challenge": 
        print("Visit eval.ai to evalaute performances on Challenge set")
        return None
        
    results =  evaluate(SoccerNet_path=dataloader.dataset.path, 
                 Predictions_path=output_results,
                 split="test",
                 prediction_file="results_spotting.json", 
                 version=dataloader.dataset.version)

    return results