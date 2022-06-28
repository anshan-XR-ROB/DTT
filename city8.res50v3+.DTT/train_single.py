from __future__ import division
import os.path as osp
import os
import sys
import time
import argparse
import math
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from config import config
from dataloader import get_train_loader
from network import Network
from dataloader import CityScape
from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
# from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
from tensorboardX import SummaryWriter

try:
    from apex.apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

try:
    from azureml.core import Run
    azure = True
    run = Run.get_context()
except:
    azure = False

parser = argparse.ArgumentParser()

os.environ['MASTER_PORT'] = '169711'

if os.getenv('debug') is not None:
    is_debug = os.environ['debug']
else:
    is_debug = False

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True

    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader + unsupervised data loader
    train_loader, train_sampler = get_train_loader(engine, CityScape, train_source=config.train_source, \
                                                   unsupervised=False)
    unsupervised_train_loader, unsupervised_train_sampler = get_train_loader(engine, CityScape, \
                train_source=config.unsup_source, unsupervised=True)

    if engine.distributed and (engine.local_rank == 0):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        logger = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
    criterion_csst = nn.MSELoss(reduction='mean')

    if engine.distributed:
        BatchNorm2d = SyncBatchNorm

    # define and init the model
    model = Network(config.num_classes, criterion=criterion,
                    pretrained_model=config.pretrained_model,
                    norm_layer=BatchNorm2d)
    # init_weight(model.branch1.business_layer, nn.init.kaiming_normal_,
    #             BatchNorm2d, config.bn_eps, config.bn_momentum,
    #             mode='fan_in', nonlinearity='relu')
    # init_weight(model.branch2.business_layer, nn.init.kaiming_normal_,
    #             BatchNorm2d, config.bn_eps, config.bn_momentum,
    #             mode='fan_in', nonlinearity='relu')
    # init_weight(model.branch3.business_layer, nn.init.kaiming_normal_,
    #             BatchNorm2d, config.bn_eps, config.bn_momentum,
    #             mode='fan_in', nonlinearity='relu')
    # model.load_state_dict(torch.load(path))
    pretrained_dict = torch.load('/data/home/scv3198/model/1-4-R101/epoch-last.pth')
    pretrained_dict = pretrained_dict['model']
    model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict)

    # define the learning rate
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr * engine.world_size

    # define the two optimizers
    params_list_l = []
    params_list_l = group_weight(params_list_l, model.branch3.backbone,
                               BatchNorm2d, base_lr)
    for module in model.branch3.business_layer:
        params_list_l = group_weight(params_list_l, module, BatchNorm2d,
                                   base_lr)        # head lr * 10

    optimizer_l = torch.optim.SGD(params_list_l,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    # params_list_r = []
    # params_list_r = group_weight(params_list_r, model.branch2.backbone,
    #                            BatchNorm2d, base_lr)
    # for module in model.branch2.business_layer:
    #     params_list_r = group_weight(params_list_r, module, BatchNorm2d,
    #                                base_lr)        # head lr * 10

    # optimizer_r = torch.optim.SGD(params_list_r,
    #                             lr=base_lr,
    #                             momentum=config.momentum,
    #                             weight_decay=config.weight_decay)

    # params_list_3 = []
    # params_list_3 = group_weight(params_list_3, model.branch3.backbone,
    #                            BatchNorm2d, base_lr)
    # for module in model.branch3.business_layer:
    #     params_list_3 = group_weight(params_list_3, module, BatchNorm2d,
    #                                base_lr)        # head lr * 10

    # optimizer_3 = torch.optim.SGD(params_list_3,
    #                             lr=base_lr,
    #                             momentum=config.momentum,
    #                             weight_decay=config.weight_decay)

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        print('distributed !!')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DataParallelModel(model, device_ids=engine.devices)
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,optimizer_l=optimizer_l)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    model.train()
    print('begin train')
    saved_prototyope_epoch = None
    print(engine.state.epoch)
    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

        if is_debug:
            pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
        else:
            pbar = tqdm(range(int(config.niters_per_epoch)), file=sys.stdout, bar_format=bar_format)


        dataloader = iter(train_loader)
        unsupervised_dataloader = iter(unsupervised_train_loader)

        sum_loss_sup = 0
        sum_loss_unsup = 0

        ''' supervised part '''
        for idx in pbar:

            optimizer_l.zero_grad()
            engine.update_iteration(epoch, idx)
            start_time = time.time()

            minibatch = dataloader.next()
            imgs_1 = minibatch['data'].cuda(non_blocking=True)
            gts_1 = minibatch['label'].cuda(non_blocking=True)

            unsup_minibatch = unsupervised_dataloader.next()
            unsup_imgs = unsup_minibatch['data'].cuda(non_blocking=True)

            b, c, h, w = imgs_1.shape
            # _, pred_sup_1 = model(imgs_1, step=1)
            # _, pred_sup_2 = model(imgs_1, step=2)
            _, pred_sup_3 = model(imgs_1, step=3)

            _, pred_unsup_1 = model(unsup_imgs, step=1)
            _, pred_unsup_2 = model(unsup_imgs, step=2)
            _, pred_unsup_3 = model(unsup_imgs, step=3)

            tao = 0.5

            loss_unsup = 0

            pred_unsup_1_2 = pred_unsup_1 + pred_unsup_2

            # pred_unsup_2_3_soft = pred_unsup_2_3.softmax(1)
            # pred_soft, _ = torch.max(pred_unsup_2_3_soft, dim=1)
            # # print(pred_unsup_2_3_soft)
            # pred_soft_mean = pred_soft.mean()
            # print(pred_soft_mean)

            # if pred_soft_mean < 0.9: continue

            # pred_sup_1_2 = pred_sup_1 + pred_sup_2
            # pred_1_2 = torch.cat([pred_sup_1_2, pred_unsup_1_2], dim=0)
            _, max_1_2 = torch.max(pred_unsup_1_2, dim=1)
            max_1_2 = max_1_2.long()
            # pred_3 = torch.cat([pred_sup_3, pred_unsup_3], dim=0)


            loss_unsup = criterion(pred_unsup_3, max_1_2)
            dist.all_reduce(loss_unsup, dist.ReduceOp.SUM)
            loss_unsup = loss_unsup / engine.world_size

            # gts_1 [batch, 512, 512]
            # pred_sup_1 [batch, 21, 512, 512]
        
            ### standard cross entropy loss ###
            # print(gts_1)
            loss_sup = criterion(pred_sup_3, gts_1)
            dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
            loss_sup = loss_sup / engine.world_size

            # reset the learning rate

            # print(criterion(pred_sup_1, gts_1))
            # loss_sup = loss_unsup

            loss = loss_unsup*1.5 + loss_sup

            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            # reset the learning rate
            optimizer_l.param_groups[0]['lr'] = lr
            optimizer_l.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_l.param_groups)):
                optimizer_l.param_groups[i]['lr'] = lr
            # optimizer_r.param_groups[0]['lr'] = lr
            # optimizer_r.param_groups[1]['lr'] = lr
            # for i in range(2, len(optimizer_r.param_groups)):
            #     optimizer_r.param_groups[i]['lr'] = lr
            # optimizer_3.param_groups[0]['lr'] = lr
            # optimizer_3.param_groups[1]['lr'] = lr
            # for i in range(2, len(optimizer_3.param_groups)):
            #     optimizer_3.param_groups[i]['lr'] = lr
            
            loss.backward()
            optimizer_l.step()
            # optimizer_r.step()
            # optimizer_3.step()

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss_sup=%.2f' % loss_sup.item() \
                        + ' loss_unsup=%.2f' % loss_unsup.item()

            sum_loss_sup += loss_sup.item()
            sum_loss_unsup += loss_unsup.item()

            pbar.set_description(print_str, refresh=False)

            # engine.save_and_link_checkpoint(config.snapshot_dir,
            #                                     config.log_dir,
            #                                     config.log_dir_link)
            end_time = time.time()


        if engine.distributed and (engine.local_rank == 0):
            logger.add_scalar('train_loss_sup', sum_loss_sup / len(pbar), epoch)
            logger.add_scalar('train_loss_unsup', sum_loss_unsup / len(pbar), epoch)

        if azure and engine.local_rank == 0:
            run.log(name='Supervised Training Loss sup', value=sum_loss_sup / len(pbar))
            run.log(name='Supervised Training Loss unsup', value=sum_loss_unsup / len(pbar))


        if  (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
 