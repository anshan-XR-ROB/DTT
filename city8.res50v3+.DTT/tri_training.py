from network import SingleNetwork
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
import numpy as np

from config import config
from dataloader import get_train_loader, get_U_loader
from network import Network
from dataloader import CityScape
from dataloader import ValPre
from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR
from engine.evaluator import Evaluator
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from seg_opr.metric import hist_info, compute_score
from utils.visualize import print_iou, show_img
# from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
from tensorboardX import SummaryWriter
from apex.apex.parallel import DistributedDataParallel, SyncBatchNorm


class TriEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        name = data['fn']
        pred = self.sliding_eval(img, config.eval_crop_size, config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp,
                        'correct': correct_tmp}
        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct,
                                                       labeled)

        return [iu, mean_IU, mean_pixel_acc]


class TriTraining(nn.Module):
    def __init__(self, num_classes, criterion, norm_layer, pretrained_model=None):
        super(TriTraining, self).__init__()
        self.branch1 = SingleNetwork(num_classes, criterion, norm_layer, pretrained_model)
        self.branch2 = SingleNetwork(num_classes, criterion, norm_layer, pretrained_model)
        self.branch3 = SingleNetwork(num_classes, criterion, norm_layer, pretrained_model)
        self.init_weights()

    def init_weights(self,):

        self.classifiers = [self.branch1, self.branch2, self.branch3]

        base_lr = 0.0001

        params_list_1 = []
        params_list_1 = group_weight(params_list_1, self.branch1.backbone,
                               BatchNorm2d, base_lr)
        for module in self.branch1.business_layer:
            params_list_1 = group_weight(params_list_1, module, BatchNorm2d,
                                    base_lr)        # head lr * 10

        self.optimizer_1 = torch.optim.SGD(params_list_1,
                                    lr=base_lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)

        params_list_2 = []
        params_list_2 = group_weight(params_list_2, self.branch2.backbone,
                                BatchNorm2d, base_lr)
        for module in self.branch2.business_layer:
            params_list_2 = group_weight(params_list_2, module, BatchNorm2d,
                                    base_lr)        # head lr * 10

        self.optimizer_2 = torch.optim.SGD(params_list_2,
                                    lr=base_lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)

        params_list_3 = []
        params_list_3 = group_weight(params_list_3, self.branch3.backbone,
                                BatchNorm2d, base_lr)
        for module in self.branch3.business_layer:
            params_list_3 = group_weight(params_list_3, module, BatchNorm2d,
                                    base_lr)        # head lr * 10

        self.optimizer_3 = torch.optim.SGD(params_list_3,
                                    lr=base_lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)
        self.optimizers = [self.optimizer_1, self.optimizer_2, self.optimizer_3]
    
    
    def save(self, PATH, it):
        torch.save(self.state_dict(), PATH + str(it) + '.pth')
    
    def save_max(self, PATH, it):
        torch.save(self.state_dict(), PATH + '/max.pth')

    def classifier_voting(self, U, X, j, k, o, device_id):
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
        X_batch = X.next()
        X_imgs = X_batch['data']
        X_gts = X_batch['label']
        X_tensor = X_imgs.cuda(non_blocking=True, device=device_id)
        gt_tensor = X_gts.cuda(non_blocking=True, device=device_id)
        
        # for i in (range(itrations)):
        while True:
            U_batch = U.next()
            U_imgs = U_batch['data']
            batch_U_tensor = U_imgs.cuda(non_blocking=True, device=device_id)

            _, pred_j_o = self.classifiers[j](batch_U_tensor)
            _, pred_k_o = self.classifiers[k](batch_U_tensor)
            _, pred_j = torch.max(pred_j_o, dim=1)
            _, pred_k = torch.max(pred_k_o, dim=1)

            ones = torch.ones_like(pred_k)
            vote_index = pred_j == pred_k
            r = vote_index.sum().float() / ones.sum().float()
            # print(r)

            if r > 0.9:
                _, pred = torch.max(pred_j_o+pred_k_o, dim=1)
                pred = pred.long()
                self.optimizers[o].zero_grad()
                _, pred_i = self.classifiers[o](batch_U_tensor)
                loss_unsup = criterion(pred_i, pred)
                _, pred_x = self.classifiers[o](X_tensor)
                loss_sup = criterion(pred_x, gt_tensor)
                loss = loss_sup*0.5 + loss_unsup#*0.1
                print("train {} classifier: {}".format(o, loss))
                loss.backward()
                self.optimizers[o].step()
                break

    def forward(self, data, step=1):
        pred1 = self.branch1(data)
        pred2 = self.branch2(data)
        pred3 = self.branch3(data)
        return pred1+pred2+pred3


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
U, U_sampler = get_U_loader(CityScape, \
                train_source=config.unsup_source, unsupervised=True)
X, X_sampler = get_U_loader(CityScape, \
                train_source=config.train_source, unsupervised=False)
# U_loader = iter(unsupervised_train_loader)
parser = argparse.ArgumentParser()
# with Engine(custom_parser=parser) as engine:
args = parser.parse_args()
# torch.distributed.init_process_group('nccl',world_size=1,rank=0)
cudnn.benchmark = True

seed = config.seed
# if engine.distributed:
#     seed = engine.local_rank
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


# BatchNorm2d = SyncBatchNorm
BatchNorm2d = nn.BatchNorm2d
criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
model = TriTraining(config.num_classes, criterion=criterion,
                pretrained_model=config.pretrained_model,
                norm_layer=BatchNorm2d)

pretrained_dict = torch.load('/DTT/CPS-101-1-4-single/snapshot/epoch-last.pth')
pretrained_dict = pretrained_dict['model']
model.load_state_dict(pretrained_dict)

PATH = '/DTT/CPS-101-1-4-single/'
if not os.path.exists(PATH):
    os.makedirs(PATH)
# device_ids=[0,1,2,3]
device_id = 0
# model = torch.nn.DataParallel(model, device_ids=device_ids)
model = model.cuda(device=device_id)

data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}
val_pre = ValPre()
val_dataset = CityScape(data_setting, 'val', val_pre, training=False)

segmentor = TriEvaluator(val_dataset, config.num_classes, config.image_mean,
                            config.image_std, model,
                            config.eval_scale_array, config.eval_flip,
                            [device_id], False, PATH,
                            False)

print('data compelete')
# error = model.measure_error(X_tensor, Y_tensor, 0, 1)
# error = model.classifier_voting(X_tensor, 0, 1)
# print(error)

# torch.save(model.state_dict(),PATH+'/tri.pth')

update = [False]*3
improve = True
iteration = 0
max_IoU = 0.

while improve:
    U_loader = iter(U)
    X_loader = iter(X)
    iteration += 1#count iterations 

    # if iteration%100==0:
    #     model.save(PATH, iteration)
    
    if iteration%100 == 0:
        model.eval()
        with torch.no_grad():
            results = segmentor.tri_run(model)
            iou = results[1]
            with open(PATH+'mIoU2.txt', 'a') as f:
                f.write(str(iou)+'\n')
            if iou > max_IoU:
                max_IoU = iou
                model.save_max(PATH, iteration)
        model.train()
    
    for i in range(3):    
        j, k = np.delete(np.array([0,1,2]),i)
        update[i] = False
        
        print("[{}] classifier_voting : {}, current max iou: {}".format(iteration, i, max_IoU))
        model.classifier_voting(U_loader, X_loader, j, k, i, device_id)
        
    if iteration>500000:
        improve = False#if no classifier was updated, no improvement

    


