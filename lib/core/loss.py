# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from PASS import RunningMode, MaskMode, MaskGate, m_cfg

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

class JointsTripleLoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsTripleLoss, self).__init__()
        self.criterion = nn.TripletMarginLoss(margin=200, swap=False, reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, dy,dx,y, target_weight):
        batch_size = dy.size(0)
        num_joints = dy.size(1)
        heatmaps_dy = dy.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_dx = dx.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_y = y.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0

        for idx in range(num_joints):
            heatmap_dy = heatmaps_dy[idx].squeeze()
            heatmap_dx = heatmaps_dx[idx].squeeze()
            heatmap_y = heatmaps_y[idx].squeeze()
            target_weight.cuda()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_dy.mul(target_weight[:, idx]),
                    heatmap_dx.mul(target_weight[:, idx]),
                    heatmap_y.mul(target_weight[:, idx])
                )
                pos_dist = 0.5 *torch.dist(heatmap_y.mul(target_weight[:, idx]), heatmap_dy.mul(target_weight[:, idx]))
                neg_dist = 0.5 *torch.dist(heatmap_y.mul(target_weight[:, idx]), heatmap_dx.mul(target_weight[:, idx]))
                pos_neg = pos_dist.detach() - neg_dist.detach()
            else:
                loss += 0.5 * self.criterion(heatmap_y, heatmap_dy, heatmap_dx)
                pos_dist = 0.5 *torch.dist(heatmap_y, heatmap_dy)
                neg_dist = 0.5 *torch.dist(heatmap_y, heatmap_dx)
                pos_neg = pos_dist.detach() - neg_dist.detach()

        return loss / num_joints, pos_dist / num_joints, neg_dist / num_joints, pos_neg / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)
