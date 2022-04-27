# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Hanbin Dai (daihanbin.ac@gmail.com) and Feng Zhang (zhangfengwcy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from core.inference import get_max_preds
import random
from utils.utils import RunningMode

logger = logging.getLogger(__name__)


def train(config, train_loader, model, teacher, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, running_mode):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    if running_mode == RunningMode.GatePreTrain:
        for n, p in model.named_parameters():
            if 'pre_feature' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if running_mode == RunningMode.GatePreTrain:
            outputs_anchor = teacher(input)
            outputs_anchor= outputs_anchor.detach()
            optimizer.zero_grad()
            mask_base = torch.ones(outputs_anchor.shape[0], outputs_anchor.shape[1], int(outputs_anchor.shape[2] / 4),
                                   int(outputs_anchor.shape[3] / 4))
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            # outputs_anchor->accuracy
            patch_size = 16
            pred_anchor_, _ = get_max_preds(outputs_anchor.cpu().numpy())
            pred_anchor_=pred_anchor_.astype(int)
            # pred_anchor_=torch.from_numpy(pred_anchor_).cuda()
            mask_ratio = 0.6 #1 ALL MASK

            all_positive_mask = mask_base.clone().detach()
            for batch_num in range(pred_anchor_.shape[0]):
                for q in range(pred_anchor_.shape[1]):
                    mask_label = pred_anchor_[batch_num, q]
                    all_positive_mask[batch_num, :, int(mask_label[1] / 4), int(mask_label[0] / 4)] = 0

            m = torch.nn.Upsample(scale_factor=4 * patch_size, mode='nearest')
            mm = torch.nn.Upsample(scale_factor=4, mode='nearest')
            for k in range(3):
                indices = random.sample(range(pred_anchor_.shape[1]), int(pred_anchor_.shape[1] * mask_ratio))
                pred_anchor = np.take(pred_anchor_, indices,axis=1)
                mask_ = torch.ones(input.shape[0],input.shape[1],int(outputs_anchor.shape[2]/patch_size),int(outputs_anchor.shape[3]/patch_size))
                # mask_ = torch.ones(input.shape[0],input.shape[1],outputs_anchor.shape[2],outputs_anchor.shape[3])
                # mask_.scatter_(1, torch.LongTensor(pred_anchor), 0)
                for batch_num in range(pred_anchor.shape[0]):
                    for q in range(pred_anchor.shape[1]):
                        mask_label = pred_anchor[batch_num, q]
                        mask_[batch_num, :, int(mask_label[1]/patch_size), int(mask_label[0]/patch_size)] = 0
                mask = m(mask_)
                # mask = mask.T
                # for a in range(mask.shape[0]):
                #     for b in range(mask.shape[1]):
                #         torchvision.transforms.functional.rotate(mask[a,b],270)
                # mask = torch.rot90(mask, 1, [2, 3])
                input_replace = torch.mul(input,mask)
                outputs_replace = model(input_replace)

                outputs_pred = outputs_replace.detach()
                pred_positive_, _ = get_max_preds(outputs_pred.cpu().numpy())
                pred_positive_ = pred_positive_.astype(int)
                pred_positive_mask = mask_base.clone().detach()
                for batch_num in range(pred_positive_.shape[0]):
                    for q in range(pred_positive_.shape[1]):
                        mask_label = pred_positive_[batch_num, q]
                        pred_positive_mask[batch_num, :, int(mask_label[1] / 4), int(mask_label[0] / 4)] = 0

                weight_mask = torch.logical_xor(all_positive_mask, pred_positive_mask)
                weight_mask = weight_mask * 9
                weight_mask = weight_mask + 1
                weight_mask = weight_mask.type(torch.FloatTensor)

                weight_mask = mm(weight_mask)
                weight_mask = weight_mask.to('cuda')

                if isinstance(outputs_replace, list):
                    loss = criterion(outputs_replace[0], outputs_anchor[0], target_weight, weight_mask[0])
                    for output_replace,output_anchor, single_weight_mask in outputs_replace[1:],outputs_anchor[1:], weight_mask[1:]:
                        loss += criterion(output_replace, output_anchor, target_weight, single_weight_mask)
                else:
                    output_replace = outputs_replace
                    output_anchor = outputs_anchor
                    loss = criterion(output_replace, output_anchor, target_weight, weight_mask)

                # loss = criterion(output, target, target_weight)

                # compute gradient and do update step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # measure accuracy and record loss
                losses.update(loss.item(), input.size(0))

                _, avg_acc, cnt, pred = accuracy(output_replace.detach().cpu().numpy(),
                                                 target.detach().cpu().numpy())
                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % config.PRINT_FREQ == 0:
                    msg = 'Epoch: [{0}][{1}/{2}]\t' \
                          'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                          'Speed {speed:.1f} samples/s\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                          'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        speed=input.size(0) / batch_time.val,
                        data_time=data_time, loss=losses, acc=acc)
                    logger.info(msg)

                    writer = writer_dict['writer']
                    global_steps = writer_dict['train_global_steps']
                    writer.add_scalar('train_loss', losses.val, global_steps)
                    writer.add_scalar('train_acc', acc.val, global_steps)
                    writer_dict['train_global_steps'] = global_steps + 1

                    prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
                    save_debug_images(config, input_replace, meta, target, pred * 4, output_replace,
                                      prefix)
                # pred_anchor upsample


            # pred_anchor reshape
            # replace input
            # pred, _ = get_max_preds(outputs_anchor)
            #     pred_anchor =
                # pred from1 random
                #  input to zero
                # newinput

                # outputs_replace = model(replace)
                #      loss
        else:
            # compute output
            outputs = model(input)

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            if isinstance(outputs, list):
                loss = criterion(outputs[0], target, target_weight)
                for output in outputs[1:]:
                    loss += criterion(output, target, target_weight)
            else:
                output = outputs
                loss = criterion(output, target, target_weight)

                # loss = criterion(output, target, target_weight)

                # compute gradient and do update step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # measure accuracy and record loss
                losses.update(loss.item(), input.size(0))

                _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                                 target.detach().cpu().numpy())
                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % config.PRINT_FREQ == 0:
                    msg = 'Epoch: [{0}][{1}/{2}]\t' \
                          'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                          'Speed {speed:.1f} samples/s\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                          'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                              epoch, i, len(train_loader), batch_time=batch_time,
                              speed=input.size(0)/batch_time.val,
                              data_time=data_time, loss=losses, acc=acc)
                    logger.info(msg)

                    writer = writer_dict['writer']
                    global_steps = writer_dict['train_global_steps']
                    writer.add_scalar('train_loss', losses.val, global_steps)
                    writer.add_scalar('train_acc', acc.val, global_steps)
                    writer_dict['train_global_steps'] = global_steps + 1

                    prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
                    save_debug_images(config, input, meta, target, pred*4, output,
                                      prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


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
        self.avg = self.sum / self.count if self.count != 0 else 0
