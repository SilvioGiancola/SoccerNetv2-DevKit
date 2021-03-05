import logging
import os
from metrics_fast import AverageMeter ,average_mAP,calculate_f1_score
import time
from tqdm import tqdm
import torch
import numpy as np
import math


def trainer(train_loader,
            val_loader,
            val_metric_loader,
            test_loader,
            model,
            optimizer,
            scheduler,
            criterion,
            weights,
            model_save_path,
            max_epochs=1000):

    logging.info("start training")

    evaluate_test_epoch = 20
    save_checkpoint_epoch = 100
    best_loss = 9e99
    best_perf = 0
    for epoch in range(max_epochs):
        best_model_path = os.path.join(model_save_path, "model.pth.tar")

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

        # mean_part_iou, f1_macro, f1_micro
        perf_validation ,_ , _= test(
            val_metric_loader, model, str(epoch + 1), model_save_path)
        # if (epoch + 1) % evaluate_test_epoch == 0 or epoch == 0:
        #     performance_test = test(test_loader, model, str(epoch + 1), model_save_path)

        #     logging.info("Performance at epoch " + str(epoch+1) + " -> " + str(performance_test))

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }
        # if (epoch + 1) % save_checkpoint_epoch == 0 or epoch == 0:
        #     torch.save(
        #         state,
        #         os.path.join(model_save_path, "model_epoch" + str(epoch + 1) + ".pth.tar"))

        # remember best prec@1 and save checkpoint
        is_better = loss_validation < best_loss
        is_better = perf_validation > best_perf
        best_loss = min(loss_validation, best_loss)
        best_perf = max(perf_validation, best_perf)

        # test the model
        if is_better:
            torch.save(state, best_model_path)

        if isinstance(scheduler, list): 

            current_learning_rate = optimizer.param_groups[0]['lr']
            new_learning_rate = current_learning_rate - (scheduler[0]-scheduler[1])/max_epochs# * 0.993116
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_learning_rate

        # print(new_learning_rate)
        # update the LR scheduler
        elif scheduler is not None:
            prevLR = optimizer.param_groups[0]['lr']
            scheduler.step(perf_validation)
            currLR = optimizer.param_groups[0]['lr']
            if (currLR is not prevLR and scheduler.num_bad_epochs == 0):
                logging.info("Plateau Reached!")

            if (prevLR < 2 * scheduler.eps and
                    scheduler.num_bad_epochs >= scheduler.patience):
                logging.info(
                    "Plateau Reached and no more reduction -> Exiting Loop")
                break

    performance_test = test(test_loader, model, str(epoch + 1), model_save_path)
    logging.info("Performance at end training " + str(epoch+1) + " -> " + str(performance_test))

    torch.save(state, os.path.join(model_save_path, "model_last.pth.tar"))
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
        
    nSamples = np.array([0]*dataloader.dataset.num_classes_sgementation)
    # with  as t:
    for i, (feat_half1, feat_half2, label_change_half1, label_change_half2,label_half1,label_half2) in enumerate(dataloader): 
        for feats, labels, targets in zip([feat_half1, feat_half2], [label_change_half1, label_change_half2],[label_half1,label_half2]):
            feats = feats.cuda()
            labels = labels.cuda().float()
            targets = targets.cuda().float()
            target_np = targets.max(dim=2)[1].squeeze(0).cpu().numpy()    
            nSamples = nSamples +  np.array([np.sum(target_np == cl, dtype=np.float32) for cl in range(dataloader.dataset.num_classes_sgementation)])
    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    normedWeights = torch.FloatTensor(normedWeights).cuda()
    # print(nSamples)
    # print(normedWeights)

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=160) as t:
        for i, (feat_half1, feat_half2, label_change_half1, label_change_half2, label_half1, label_half2) in t: 
            # if np.random.randint(100, size=1)<90:
            #      continue
            # measure data loading time
            data_time.update(time.time() - end)
            for feats, labels, targets in zip([feat_half1, feat_half2], [label_change_half1, label_change_half2],[label_half1,label_half2]):
                feats = feats.cuda()
                labels = labels.cuda().float()
                targets = targets.cuda().float()
                # print("feats", feats.shape)
                # print("labels", labels.shape)
                # print("targets", targets.shape)


                # target_np = targets.max(dim=2)[1].squeeze(0).cpu().numpy()    
                # nSamples = [np.sum(target_np == cl, dtype=np.float32) for cl in range(14)]
                # normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
                # normedWeights = torch.FloatTensor(normedWeights).cuda()
                # print(nSamples)
                # feats=feats.unsqueeze(1)

                # compute output
                output = model(feats)
                output = output.permute((0,2,1))
                output = output.reshape(-1,dataloader.dataset.num_classes_sgementation)
                targets_mse = targets.reshape(-1,dataloader.dataset.num_classes_sgementation)
                targets_ce = targets.max(dim=2)[1].squeeze(0)
                
                # print(output[output == 0])
                # targets = targets.reshape(-1,14)
                # print(output.shape)
                # print(targets.shape)
                # loss = torch.nn.CrossEntropyLoss()(output, targets)
                # import random
                # # criterion = torch.nn.CrossEntropyLoss(weight=normedWeights)
                # criterion = torch.nn.CrossEntropyLoss()
                # # criterion = torch.nn.MSELoss()
                # loss = 0
                # for cl in range(14):
                #     # print((targets == 0).nonzero().shape)
                #     k=10
                #     idx = torch.nonzero(targets_ce == cl, as_tuple=False).squeeze(1).cpu().numpy()
                #     if len(idx) < 1:
                #         # print(f"{cl} has no sample")
                #         continue
                #     idx = np.random.choice(idx,200)
                #     # idx = random.sample(idx,10)
                #     # print(idx)
                #     # perm = torch.randperm(len(idx))
                #     # idx = perm[:k]
                #     # print(idx)

                #     # print(idx.shape)
                #     # print(output[idx,:])
                #     # print(targets[idx])
                #     # print(output[idx,:].shape)
                #     # print(targets[idx].shape)
                #     loss += criterion(output[idx,:], targets_mse[idx,:])
                if criterion == "CE":
                    loss = torch.nn.CrossEntropyLoss()(output, targets_ce)
                # loss = torch.nn.CrossEntropyLoss(weight=normedWeights)(output, targets_ce)
                elif criterion == "MSE":
                    loss = torch.nn.MSELoss()(output, targets_mse)

                # loss_segmentation = criterion[0](labels, output_segmentation) 
                # loss_spotting = criterion[1](targets, output_spotting)

                # loss = weights[0]*loss_segmentation + weights[1]*loss_spotting

                # measure accuracy and record loss
                losses.update(loss.item(), feats.size(0))
                # losses_segmentation.update(loss_segmentation.item(), feats.size(0))
                # losses_spotting.update(loss_spotting.item(), feats.size(0))

                if train:
                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if train:
                    desc = f'Train {epoch:4d}: '
                else:
                    desc = f'Evaluate {epoch:4d}: '
                desc += f'Time {batch_time.avg:.3f}s '
                desc += f'(it:{batch_time.val:.3f}s) '
                desc += f'Data:{data_time.avg:.3f}s '
                desc += f'(it:{data_time.val:.3f}s) '
                desc += f'Loss {losses.avg:.4f} '
                # desc += f'(Seg:{losses_segmentation.avg:.4f}/'
                # desc += f'Spot:{losses_spotting.avg:.4f}) '
                t.set_description(desc)

    # logging.info(desc)
    return losses.avg


def test(dataloader, model, epoch, model_save_path):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    spotting_grountruth = list()
    spotting_predictions = list()
    segmentation_grountruth = list()
    segmentation_predictions = list()

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
            
            for i in np.arange(output_spotting.size()[1]):
                tmp_timestamps[torch.floor(output_spotting[batch,i,1]*(chunk_size-1)).type(torch.int)] = output_spotting[batch,i,0]

            # ------------------------------------------
            # Store the result of the chunk in the video
            # ------------------------------------------

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

        segmentation_long = torch.zeros([video_size,output_segmentation.size()[-1]], dtype = torch.float, device=output_segmentation.device)


        for batch in np.arange(output_segmentation.size()[0]):

            tmp_segmentation = torch.nn.functional.one_hot(torch.argmax(output_segmentation[batch], dim=-1), num_classes=output_segmentation.size()[-1])


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


    part_intersect = np.zeros(dataloader.dataset.num_classes_sgementation, dtype=np.float32)
    part_union = np.zeros(dataloader.dataset.num_classes_sgementation, dtype=np.float32)


    targets_numpy2 = list()
    detections_numpy2 = list()

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
        for i, (feat_half1, feat_half2, label_change_half1, label_change_half2,label_half1,label_half2) in t:
            # if np.random.randint(10, size=1)<8:
            #      continue
            data_time.update(time.time() - end)

            for feats, labels, targets in zip([feat_half1, feat_half2], [label_change_half1, label_change_half2],[label_half1,label_half2]):

                # feat_half1 = feat_half1.cuda().squeeze(0)
                # label_change_half1 = label_change_half1.float().squeeze(0)
                # feat_half2 = feat_half2.cuda().squeeze(0)
                # label_change_half2 = label_change_half2.float().squeeze(0)
                # label_half1 = label_half1.float().squeeze(0)
                # label_half2 = label_half2.float().squeeze(0)
                # feats = feats.cuda().squeeze(0)
                # labels = labels.float().squeeze(0)
                # targets = targets.float().squeeze(0)
                feats = feats.cuda()
                labels = labels.cuda().float()
                targets = targets.cuda().float()

                # feat_half1=feat_half1.unsqueeze(1)
                # feat_half2=feat_half2.unsqueeze(1)

            # Compute the output
                output = model(feats)
                output=output.permute((0,2,1))
                # print(output.shape)
                pred = output.max(dim=2)[1]
                # print(pred.shape)
                targets = targets.max(dim=2)[1]
                # print(targets.shape)
                pred_np = pred.squeeze(0).cpu().numpy()
                target_np = targets.squeeze(0).cpu().numpy()    

                for cl in range(dataloader.dataset.num_classes_sgementation):
                    cur_gt_mask = (target_np == cl)
                    cur_pred_mask = (pred_np == cl)

                    I = np.sum(np.logical_and(cur_gt_mask,cur_pred_mask), dtype=np.float32) + 1
                    U = np.sum(np.logical_or(cur_gt_mask,cur_pred_mask), dtype=np.float32) + 1

                    if U > 0:
                        part_intersect[cl] += I
                        part_union[cl] += U

    # for target2, detection2 in zip(segmentation_grountruth,segmentation_predictions):
                targets_numpy2.append(target_np)
                detections_numpy2.append(pred_np)
                        # cur_shape_iou_tot += I/U
                        # cur_shape_iou_cnt += 1.

    part_iou = np.divide(part_intersect, part_union)
    mean_part_iou = np.mean(part_iou)
    # print("mean_part_iou", mean_part_iou)
    # a_mAP = average_mAP(targets_numpy, detections_numpy, model.framerate)
    from sklearn.metrics import f1_score
    # print(targets_numpy2)
    # print(detections_numpy2)

    targets_numpy2 = np.concatenate(targets_numpy2)
    detections_numpy2 = np.concatenate(detections_numpy2)

    # print(targets_numpy2)
    # print(detections_numpy2)
    f1_macro=f1_score(targets_numpy2, detections_numpy2, average='macro')
    f1_micro=f1_score(targets_numpy2, detections_numpy2, average='micro')

    # f1_scores=calculate_f1_score(targets_numpy2,detections_numpy2)
    # print("Test mean_part_iou: ", mean_part_iou)
    # print("Test F1_scores: macro", f1_macro, "micro",f1_micro)

    return mean_part_iou, f1_macro, f1_micro

                    # cls_pred = str(pred_np[i])
            # output = model(feat_half2)

    #         timestamp_long_half_1 = timestamps2long(output_spotting_half_1.cpu().detach(), label_change_half1.size()[0], chunk_size, receptive_field)
    #         timestamp_long_half_2 = timestamps2long(output_spotting_half_2.cpu().detach(), label_change_half2.size()[0], chunk_size, receptive_field)
    #         segmentation_long_half_1 = batch2long(output_segmentation_half_1.cpu().detach(), label_half1.size()[0], chunk_size, receptive_field)
    #         segmentation_long_half_2 = batch2long(output_segmentation_half_2.cpu().detach(), label_half2.size()[0], chunk_size, receptive_field)

    #         spotting_grountruth.append(label_change_half1)
    #         spotting_grountruth.append(label_change_half2)
    #         spotting_predictions.append(timestamp_long_half_1)
    #         spotting_predictions.append(timestamp_long_half_2)
    #         segmentation_predictions.append(segmentation_long_half_1)
    #         segmentation_predictions.append(segmentation_long_half_2)
    #         segmentation_grountruth.append(label_half1)
    #         segmentation_grountruth.append(label_half2)

    
    # targets_numpy = list()
    # detections_numpy = list()
    # for target, detection in zip(spotting_grountruth,spotting_predictions):
    #     targets_numpy.append(target.numpy())
    #     detections_numpy.append(detection.numpy())


    # targets_numpy2 = list()
    # detections_numpy2 = list()
    # for target2, detection2 in zip(segmentation_grountruth,segmentation_predictions):
    #     targets_numpy2.append(target2.numpy())
    #     detections_numpy2.append(detection2.numpy())
    # np.save(os.path.join(model_save_path, f"spotting_groundtruth_{epoch}.npy"), targets_numpy)
    # np.save(os.path.join(model_save_path, f"spotting_predictions_{epoch}.npy"), detections_numpy)
    # np.save(os.path.join(model_save_path, f"segementation_groundtruth_{epoch}.npy"), targets_numpy2)
    # np.save(os.path.join(model_save_path, f"segementation_predictions_{epoch}.npy"), detections_numpy2)
    
    # a_mAP = average_mAP(targets_numpy, detections_numpy, model.framerate)
    # f1_scores=calculate_f1_score(targets_numpy2,detections_numpy2)
    # print("Test average-mAP: ", a_mAP)
    # print("Test F1_scores: ", f1_scores)

    # return a_mAP,f1_scores
