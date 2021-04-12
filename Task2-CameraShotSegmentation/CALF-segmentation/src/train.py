import logging
import os
from metrics_fast import AverageMeter ,average_mAP,calculate_f1_score
import time
# from metrics_fast import average_mAP_visibility
from metrics_visibility_fast import average_mAP_visibility

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
            max_epochs,
            evaluation_frequency,
	    advanced_test):

    logging.info("start training")

    best_loss_model_path = os.path.join(model_save_path, "model_best_loss.pth.tar")
    best_loss = 9e99
    best_loss_model_epoch = -1

    loss_weight_segmentation, loss_weight_detection = weights
    best_eval_model_path = os.path.join(model_save_path, "model_best_eval.pth.tar")
    best_eval = -1
    best_eval_model_epoch = -1

    for epoch in range(max_epochs):
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

        # remember best prec@1 and save checkpoint
        is_loss_better = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)

        # Save the best model based on loss
        if is_loss_better:
            best_loss_model_epoch = epoch + 1
            logging.info(f"New best loss model found at epoch {best_loss_model_epoch:3d}, with loss {best_loss:3.4f}")
            torch.save(state, best_loss_model_path)

        # Test the model on the validation and test set
        if (epoch + 1) % evaluation_frequency == 0 and epoch != 0:
            performance_validation = test(val_metric_loader, model, str(epoch + 1), model_save_path)
            log_performance(performance_validation, epoch + 1, "Validation performance", advanced_test)

            # default use boundary detection as evaluation metric for model selection. Unless loss_weight_detection is 0.
            a_map = performance_validation[0][0]
            eval_score = a_map
            if loss_weight_detection == 0:
                f1_scores = performance_validation[6]
                f1_macro = f1_scores[0]
                eval_score = f1_macro

            if eval_score > best_eval:
                best_eval = eval_score
                best_eval_model_epoch = epoch + 1
                torch.save(state, best_eval_model_path)
                logging.info(f"New best eval model found at epoch {best_eval_model_epoch:3d}, with eval {best_eval:3.4f}")
                performance_test = test(test_loader, model, str(epoch + 1), model_save_path)
                log_performance(performance_test, epoch + 1, "Test performance", advanced_test)

        if isinstance(scheduler, list): 
            current_learning_rate = optimizer.param_groups[0]['lr']
            new_learning_rate = current_learning_rate - (scheduler[0]-scheduler[1])/max_epochs  # Exponential decay * 0.993116. For same learning rate schedule hardcode max_epochs = 1000
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_learning_rate

        # print(new_learning_rate)
        # update the LR scheduler
        elif scheduler is not None:
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

    logging.info(f"Best loss model found at epoch: {best_loss_model_epoch}")
    logging.info(f"Best eval model found at epoch: {best_eval_model_epoch}")

    performance_test = test(test_loader, model, str(epoch + 1), model_save_path)
    log_performance(performance_test, epoch + 1, "Test performance at end training", advanced_test)
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
        
    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=160) as t:
        for i, (feats, labels, targets) in t: 
            # if np.random.randint(100, size=1)<90:
            #      continue
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
                desc = f'Train    {epoch:4d}: '
            else:
                desc = f'Evaluate {epoch:4d}: '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            desc += f'Loss {losses.avg:.4f} '
            desc += f'(Seg:{losses_segmentation.avg:.4f}/'
            desc += f'Spot:{losses_spotting.avg:.4f}) '
            t.set_description(desc)

    logging.info(desc)
    return losses.avg


def test(dataloader, model, epoch, model_save_path):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    spotting_grountruth = list()
    spotting_grountruth_visibility = list()

    spotting_predictions = list()
    segmentation_grountruth = list()
    segmentation_predictions = list()

    part_intersect = np.zeros(dataloader.dataset.num_classes_sgementation, dtype=np.float32)
    part_union = np.zeros(dataloader.dataset.num_classes_sgementation, dtype=np.float32)

    chunk_size = model.chunk_size
    receptive_field = model.receptive_field

    model.eval()

    count_visible = torch.FloatTensor([0.0]*dataloader.dataset.num_classes_camera_change)
    count_unshown = torch.FloatTensor([0.0]*dataloader.dataset.num_classes_camera_change)
    count_all = torch.FloatTensor([0.0]*dataloader.dataset.num_classes_camera_change)

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

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
        for i, (feat_half1, feat_half2, label_change_half1, label_change_half2,label_half1,label_half2) in t:
            # if np.random.randint(10, size=1)<8:
            #      continue
            data_time.update(time.time() - end)

            feat_half1 = feat_half1.cuda().squeeze(0)
            label_change_half1 = label_change_half1.float().squeeze(0)
            feat_half2 = feat_half2.cuda().squeeze(0)
            label_change_half2 = label_change_half2.float().squeeze(0)
            label_half1 = label_half1.float().squeeze(0)
            label_half2 = label_half2.float().squeeze(0)


            feat_half1=feat_half1.unsqueeze(1)
            feat_half2=feat_half2.unsqueeze(1)

            # Compute the output
            output_segmentation_half_1, output_spotting_half_1 = model(feat_half1)
            output_segmentation_half_2, output_spotting_half_2 = model(feat_half2)

            timestamp_long_half_1 = timestamps2long(output_spotting_half_1.cpu().detach(), label_change_half1.size()[0], chunk_size, receptive_field)
            timestamp_long_half_2 = timestamps2long(output_spotting_half_2.cpu().detach(), label_change_half2.size()[0], chunk_size, receptive_field)
            segmentation_long_half_1 = batch2long(output_segmentation_half_1.cpu().detach(), label_half1.size()[0], chunk_size, receptive_field)
            segmentation_long_half_2 = batch2long(output_segmentation_half_2.cpu().detach(), label_half2.size()[0], chunk_size, receptive_field)

            spotting_grountruth.append(torch.abs(label_change_half1))
            spotting_grountruth.append(torch.abs(label_change_half2))
            spotting_grountruth_visibility.append(label_change_half1)
            spotting_grountruth_visibility.append(label_change_half2)
            spotting_predictions.append(timestamp_long_half_1)
            spotting_predictions.append(timestamp_long_half_2)
            segmentation_predictions.append(segmentation_long_half_1)
            segmentation_predictions.append(segmentation_long_half_2)
            segmentation_grountruth.append(label_half1)
            segmentation_grountruth.append(label_half2)

            ## Metric detection abrupt/smooth

            count_all = count_all + torch.sum(torch.abs(label_change_half1), dim=0)
            count_visible = count_visible + torch.sum((torch.abs(label_change_half1)+label_change_half1)/2, dim=0)
            count_unshown = count_unshown + torch.sum((torch.abs(label_change_half1)-label_change_half1)/2, dim=0)
            count_all = count_all + torch.sum(torch.abs(label_change_half2), dim=0)
            count_visible = count_visible + torch.sum((torch.abs(label_change_half2)+label_change_half2)/2, dim=0)
            count_unshown = count_unshown + torch.sum((torch.abs(label_change_half2)-label_change_half2)/2, dim=0)


            ## Metric segmentation
            pred_np = segmentation_long_half_1.max(dim=1)[1].numpy() #pred.squeeze(0).cpu().numpy()
            target_np = label_half1.max(dim=1)[1].numpy() #targets.squeeze(0).cpu().numpy()    
            # print(pred_np,pred_np.shape)
            # print(target_np,target_np.shape)
            for cl in range(dataloader.dataset.num_classes_sgementation):
                cur_gt_mask = (target_np == cl)
                cur_pred_mask = (pred_np == cl)
                # print(cur_gt_mask)
                # print(cur_pred_mask)
                I = np.sum(np.logical_and(cur_gt_mask,cur_pred_mask), dtype=np.float32)
                U = np.sum(np.logical_or(cur_gt_mask,cur_pred_mask), dtype=np.float32)

                if U > 0:
                    part_intersect[cl] += I
                    part_union[cl] += U

            pred_np = segmentation_long_half_2.max(dim=1)[1].numpy() #pred.squeeze(0).cpu().numpy()
            target_np = label_half2.max(dim=1)[1].numpy() #targets.squeeze(0).cpu().numpy()    

            for cl in range(dataloader.dataset.num_classes_sgementation):
                cur_gt_mask = (target_np == cl)
                cur_pred_mask = (pred_np == cl)

                I = np.sum(np.logical_and(cur_gt_mask,cur_pred_mask), dtype=np.float32) + 1
                U = np.sum(np.logical_or(cur_gt_mask,cur_pred_mask), dtype=np.float32) + 1

                if U > 0:
                    part_intersect[cl] += I
                    part_union[cl] += U
    
    targets_numpy = list()
    targets_visibility_numpy = list()
    detections_numpy = list()
    for target, target_visibility, detection in zip(spotting_grountruth,spotting_grountruth_visibility,spotting_predictions):
        targets_numpy.append(target.numpy())
        targets_visibility_numpy.append(target_visibility.numpy())
        detections_numpy.append(detection.numpy())


    targets_numpy2 = list()
    detections_numpy2 = list()
    for target2, detection2 in zip(segmentation_grountruth,segmentation_predictions):
        targets_numpy2.append(target2.numpy())
        detections_numpy2.append(detection2.numpy())
    # np.save(os.path.join(model_save_path, f"spotting_groundtruth_{epoch}.npy"), targets_numpy)
    # np.save(os.path.join(model_save_path, f"spotting_predictions_{epoch}.npy"), detections_numpy)
    # np.save(os.path.join(model_save_path, f"segementation_groundtruth_{epoch}.npy"), targets_numpy2)
    # np.save(os.path.join(model_save_path, f"segementation_predictions_{epoch}.npy"), detections_numpy2)
    

    part_iou = np.divide(part_intersect, part_union)
    mean_part_iou = np.mean(part_iou)


    # a_mAP = average_mAP(targets_numpy, detections_numpy, model.framerate)
    f1_scores=calculate_f1_score(targets_numpy2,detections_numpy2)
    # print("Test average-mAP: ", a_mAP)
    # print("Test F1_scores: ", f1_scores)
    # print("Test mean_part_iou: ", mean_part_iou)

    if dataloader.dataset.version == 1:
        # a_mAP = average_mAP(spotting_grountruth, spotting_predictions, model.framerate)
        a_mAP = average_mAP(targets_numpy, detections_numpy, model.framerate)
        print("Average-mAP: ", a_mAP)
        return a_mAP,f1_scores,mean_part_iou
    else:

        # a_mAP = average_mAP(targets_numpy, detections_numpy, model.framerate)
        # print("Average-mAP: ", a_mAP)
        
        a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = \
        average_mAP_visibility(targets_visibility_numpy, detections_numpy, model.framerate)
        # average_mAP_visibility(spotting_grountruth_visibility, spotting_predictions, model.framerate)
        print("a_mAP visibility all: ", a_mAP)
        print("a_mAP visibility all per class: ", a_mAP_per_class)
        print("a_mAP visibility visible: ", a_mAP_visible)
        print("a_mAP visibility visible per class: ", a_mAP_per_class_visible)
        print("a_mAP visibility unshown: ", a_mAP_unshown)
        print("a_mAP visibility unshown per class: ", a_mAP_per_class_unshown)    
        print("Count all: ", torch.sum(count_all))
        print("Count all per class: ", count_all)
        print("Count visible: ", torch.sum(count_visible))
        print("Count visible per class: ", count_visible)
        print("Count unshown: ", torch.sum(count_unshown))
        print("Count unshown per class: ", count_unshown)
        return a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown, f1_scores, mean_part_iou

    # return a_mAP,f1_scores,mean_part_iou

def log_performance(performance, epoch, description, change_type):
    a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown, f1_scores, mean_part_iou = performance
    logging.info(f"{description}. epoch: {epoch:3d} a_mAP: {a_mAP[0]:.4f} a_mAP_{change_type}: {a_mAP_visible[0]:.4f} f1_macro: {f1_scores[0]:.4f} f1_micro: {f1_scores[1]:.4f} mIoU: {mean_part_iou:.4f}")
