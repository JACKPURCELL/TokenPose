# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PASS import RunningMode, MaskMode, MaskGate, m_cfg

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss,JointsTripleLoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint,save_every_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary
from PASS import RunningMode

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    parser.add_argument('--mode', type=str, default='Pretrain',
                                 help='Mask Gate mode')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if args.mode == 'Pretrain':
        args.running_mode = RunningMode.GatePreTrain
    if args.mode == 'Finetuning':
        args.running_mode = RunningMode.FineTuning
    if args.mode == 'Test':
        args.running_mode = RunningMode.Test
        m_cfg.mask_mode = MaskMode.Anchor
    if args.mode == 'Origin':
        args.running_mode = RunningMode.BackboneTrain
        m_cfg.mask_mode = MaskMode.Anchor
    if args.mode == 'BackboneTest':
        args.running_mode = RunningMode.BackboneTest
        m_cfg.mask_mode = MaskMode.Anchor
    print(m_cfg.mask_mode,args.running_mode)
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg,args, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    # writer_dict['writer'].add_graph(model, (dump_input, ))

    logger.info(get_model_summary(model, dump_input))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    if args.running_mode == RunningMode.GatePreTrain:
        criterion = JointsTripleLoss(
            use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
        ).cuda()
    else:
        criterion = JointsMSELoss(
            use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
        ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        state_dict = checkpoint['state_dict']
        # len_origin = len(checkpoint['optimizer']['param_groups'][0]['params'])
        # len_new = len(optimizer.param_groups[0]['params'])
        # while len_origin < len_new:
        #     len_origin += 1
        #     checkpoint['optimizer']['param_groups'][0]['params'].append(len_origin)
        # # checkpoint['optimizer']['param_groups'][0]['params'] = optimizer.param_groups[0]['params']
        # optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.param_groups[0]['initial_lr'] = checkpoint['optimizer']['param_groups'][0]['initial_lr']
        optimizer.param_groups[0]['lr'] = checkpoint['optimizer']['param_groups'][0]['lr']
        optimizer.param_groups[0]['betas'] = checkpoint['optimizer']['param_groups'][0]['betas']
        optimizer.param_groups[0]['eps'] = checkpoint['optimizer']['param_groups'][0]['eps']
        optimizer.param_groups[0]['weight_decay'] = checkpoint['optimizer']['param_groups'][0]['weight_decay']
        optimizer.param_groups[0]['amsgrad'] = checkpoint['optimizer']['param_groups'][0]['amsgrad']


        # state_optimizer_ = checkpoint['optimizer']


        # state_dict = {}
        model_state_dict={}
        # state_optimizer = {}

        # convert data_parallal to model
        # for k in state_dict_:
        #     if k.startswith('module') and not k.startswith('module_list'):
        #         state_dict[k[7:]] = state_dict_[k]
        #     else:
        #         state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()
        # for k in model_state_dict_:
        #     if k.startswith('module') and not k.startswith('module_list'):
        #         model_state_dict[k[7:]] = model_state_dict_[k]
        #     else:
        #         model_state_dict[k] = model_state_dict_[k]
        # for k in state_optimizer_:
        #     if k.startswith('module') and not k.startswith('module_list'):
        #         state_optimizer[k[7:]] = state_optimizer_[k]
        #     else:
        #         state_optimizer[k] = state_optimizer_[k]
        # model_state_optimizer = optimizer.state_dict()
        # # check loaded parameters and created model parameters

        msg = 'If you see this, your model does not fully load the ' + \
              'pre-trained weight. Please make sure ' + \
              'you have correctly specified --arch xxx ' + \
              'or set the correct --num_classes for your own dataset.'
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, ' \
                          'loaded shape{}. {}'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape, msg))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k) + msg)
        for k in model_state_dict:
            print(k)
            if not (k in state_dict):
                print('No param {}.'.format(k) + msg)
                state_dict[k] = model_state_dict[k]

        # for k in state_optimizer:
        #     if k in model_state_optimizer:
        #         if state_optimizer[k].shape != model_state_optimizer[k].shape:
        #             print('Skip loading parameter {}, required shape{}, ' \
        #                   'loaded shape{}. {}'.format(
        #                 k, model_state_optimizer[k].shape, state_dict[k].shape, msg))
        #             state_optimizer[k] = model_state_optimizer[k]
        #     else:
        #         print('Drop parameter {}.'.format(k) + msg)
        # for k in model_state_optimizer:
        #     if not (k in state_optimizer):
        #         print('No param {}.'.format(k) + msg)
        #         state_optimizer[k] = model_state_dict[k]

        model.load_state_dict(state_dict, strict=False)

        # model.load_state_dict(checkpoint['state_dict'])


        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    ) if cfg.TRAIN.LR_SCHEDULER is 'MultiStepLR' else torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.TRAIN.END_EPOCH, eta_min=cfg.TRAIN.LR_END, last_epoch=last_epoch)


    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        logger.info("=> current learning rate is {:.6f}".format(lr_scheduler.get_last_lr()[0]))
        # lr_scheduler.step()

        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict,args.running_mode)

        if args.running_mode == RunningMode.GatePreTrain:
            lr_scheduler.step()
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_every_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },  final_output_dir, filename=str(epoch + 1)+'checkpoint.pth')
            lr_scheduler.step()

        else:
            # evaluate on validation set
            perf_indicator = validate(
                cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, writer_dict
            )
            lr_scheduler.step()


            if perf_indicator >= best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False

            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
