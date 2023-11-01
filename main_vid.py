#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 00:45:11 2022

@author: 22905553
"""

from __future__ import print_function
import datetime
import os
import time
import yaml
import pprint
import random
import pickle
import shutil
import inspect
import argparse
from collections import OrderedDict, defaultdict

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
#import apex
#from scheduler import GradualWarmupScheduler
from utils import count_params, import_class
from sklearn.metrics import confusion_matrix

from scipy.stats import spearmanr
import csv
import thop
from thop import clever_format

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def weights_init(m):
    #for m in modules:
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            torch.nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
                
def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='MS-G3D')

    parser.add_argument(
        '--work-dir',
        type=str,
        required=True,
        help='the work folder for storing results')
    parser.add_argument('--model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--assume-yes',
        action='store_true',
        help='Say yes to every prompt')

    parser.add_argument(
        '--phase',
        default='train',
        help='must be train or test')
    parser.add_argument(
        '--step-size',
        default=10,
        help='step-size')
    parser.add_argument(
        '--warm-up-epoch',
        default=50,
        help='warmup_epoch')
    parser.add_argument(
        '--warm-up',
        type=str2bool,
        default=False,
        help='if ture, warm up epochs starts')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')
    parser.add_argument(
        '--pretrain-JFE',
        type=str2bool,
        default=False,
        help='if ture, the JFE will run in evaluation mode')
    parser.add_argument(
        '--cosine-schedule',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')
    parser.add_argument(
        '--seed',
        type=int,
        default=random.randrange(200),
        help='random seed')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--class-loss',
        type=str2bool,
        default=False,
        help='if ture, the classification loss will be used')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--eval-start',
        type=int,
        default=1,
        help='The epoch number to start evaluating models')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')
    
    parser.add_argument(
        '--stream',
        default='SKEL',
        nargs='+',
        help='Input Streams: RGB, SKEL')

    parser.add_argument(
        '--feeder',
        default='feeder.feeder',
        help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=os.cpu_count(),
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    parser.add_argument(
        '--model1',
        default=None,
        help='the model will be used')
    parser.add_argument(
        '--model3',
        default=None,
        help='the model will be used')
    parser.add_argument(
        '--model2',
        default=None,
        help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--seq-model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--vid-model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights1',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--weights2',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--weights3',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights2',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')
    parser.add_argument(
        '--ignore-weights3',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')
    parser.add_argument(
        '--ignore-weights1',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')
    parser.add_argument(
        '--half',
        action='store_true',
        help='Use half-precision (FP16) training')
    parser.add_argument(
        '--amp-opt-level',
        type=int,
        default=1,
        help='NVIDIA Apex AMP optimization level')

    parser.add_argument(
        '--base-lr',
        type=float,
        default=0.01,
        help='initial learning rate')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.01,
        help='lr_decay_rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--optimizer',
        default='SGD',
        help='type of optimizer')
    parser.add_argument(
        '--nesterov',
        type=str2bool,
        default=False,
        help='use nesterov or not')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='training batch size')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=256,
        help='test batch size')
    parser.add_argument(
        '--forward-batch-size',
        type=int,
        default=16,
        help='Batch size during forward pass, must be factor of --batch-size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--optimizer-states',
        type=str,
        help='path of previously saved optimizer states')
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='path of previously saved training checkpoint')
    parser.add_argument(
        '--debug',
        type=str2bool,
        default=False,
        help='Debug mode; default false')
    parser.add_argument(
        '--pretrain',
        type=str2bool,
        default=False,
        help='Pretrain mode; default false')
    return parser


class Processor():
    """Processor for Skeleton-based Action Recgnition"""

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            # Added control through the command line
            arg.train_feeder_args['debug'] = arg.train_feeder_args['debug'] or self.arg.debug
            logdir = os.path.join(arg.work_dir, 'trainlogs')
            if not arg.train_feeder_args['debug']:
                # logdir = arg.model_saved_name
                if os.path.isdir(logdir):
                    print(f'log_dir {logdir} already exists')
                    if arg.assume_yes:
                        answer = 'y'
                    else:
                        answer = input('delete it? [y]/n:')
                    if answer.lower() in ('y', ''):
                        shutil.rmtree(logdir)
                        print('Dir removed:', logdir)
                    else:
                        print('Dir not removed:', logdir)

                self.train_writer = SummaryWriter(os.path.join(logdir, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(logdir, 'val'), 'val')
            else:
                self.train_writer = SummaryWriter(os.path.join(logdir, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(logdir, 'val'), 'val')
        out_folder_path = os.path.join(self.arg.work_dir, 'score')     
        os.makedirs(out_folder_path, exist_ok=True)

        self.load_model()
        self.load_param_groups()
        self.load_optimizer()
        if not self.arg.warm_up:
            self.load_lr_scheduler()
        self.load_data()

        self.global_step = 0
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.best_loss = 10000
        self.best_loss_epoch = 0
        '''
        if self.arg.half:
            self.print_log('*************************************')
            self.print_log('*** Using Half Precision Training ***')
            self.print_log('*************************************')
            print(isinstance(self.optimizer, torch.optim.Optimizer) )
            print(isinstance(self.optimizer, list))
            
            if 'SKEL' in self.arg.stream:
                self.model, self.seq_model, self.optimizer = apex.amp.initialize(
                        self.model,
                        self.seq_model,
                        self.optimizer,
                        opt_level=f'O{self.arg.amp_opt_level}'
                    )
            if 'RGB' in self.arg.stream:
                self.vid_model = apex.amp.initialize(
                        self.vid_model,
                        opt_level=f'O{self.arg.amp_opt_level}'
                    )
            if self.arg.amp_opt_level != 1:
                self.print_log('[WARN] nn.DataParallel is not yet supported by amp_opt_level != "O1"')
        '''
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.print_log(f'{len(self.arg.device)} GPUs available, using DataParallel')
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device
                )

    def load_model(self):
        output_device = self.arg.device[0] if type(
            self.arg.device) is list else self.arg.device
        self.output_device = output_device
        if 'SKEL' in self.arg.stream:
            Model = import_class(self.arg.model1)
            Seq_Model = import_class(self.arg.model2)
            self.model = Model(**self.arg.model_args).cuda(output_device)
            self.seq_model = Seq_Model(**self.arg.seq_model_args).cuda(output_device)
        if 'RGB' in self.arg.stream:
            Vid_model = import_class(self.arg.model3)
            self.vid_model = Vid_model(**self.arg.vid_model_args).cuda(output_device)
            print('Pretrain = ',self.arg.pretrain)
            if self.arg.pretrain:
                #print()
                for name, params in self.vid_model.named_parameters():
                    if 'features.0.8' in name or 'gru.' in name or 'fc.' in name:
                        print(name)
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
        if self.arg.stream == '2s':
            Seq_Model = import_class(self.arg.model2)
            self.seq_model = Seq_Model(**self.arg.seq_model_args).cuda(output_device)
            Vid_model = import_class(self.arg.model3)
            self.vid_model = Vid_model(**self.arg.vid_model_args).cuda(output_device)
        # Copy model file and main
        #shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        shutil.copy2(os.path.join('.', __file__), self.arg.work_dir)
        shutil.copytree('./model', self.arg.work_dir, dirs_exist_ok=True)
        
        if self.arg.weights1:
            try:
                self.global_step = int(self.arg.weights1[:-3].split('-')[-1])
            except:
                print('Cannot parse global_step from model weights filename')
                self.global_step = 0

            self.print_log(f'Loading weights from {self.arg.weights1}')
            if '.pkl' in self.arg.weights1:
                with open(self.arg.weights1, 'r') as f:
                    weights1 = pickle.load(f)
            else:
                weights1 = torch.load(self.arg.weights1)

            weights1 = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights1.items()])

            for w in self.arg.ignore_weights1:
                if weights1.pop(w, None) is not None:
                    self.print_log(f'Sucessfully Remove Weights: {w}')
                else:
                    self.print_log(f'Can Not Remove Weights: {w}')

            try:
                self.model.load_state_dict(weights1)
            except:
                state1 = self.model.state_dict()
                diff1 = list(set(state1.keys()).difference(set(weights1.keys())))
                self.print_log('Can not find these weights:')
                for d in diff1:
                    self.print_log('  ' + d)
                state1.update(weights1)
                self.model.load_state_dict(state1)
        
        if self.arg.weights2:
            try:
                self.global_step = int(self.arg.weights2[:-3].split('-')[-1])
            except:
                print('Cannot parse global_step from model weights filename')
                self.global_step = 0

            self.print_log(f'Loading weights from {self.arg.weights2}')
            if '.pkl' in self.arg.weights2:
                with open(self.arg.weights2, 'r') as f:
                    weights2 = pickle.load(f)
            else:
                weights2 = torch.load(self.arg.weights2)

            weights2 = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights2.items()])

            for w in self.arg.ignore_weights2:
                if weights2.pop(w, None) is not None:
                    self.print_log(f'Sucessfully Remove Weights: {w}')
                else:
                    self.print_log(f'Can Not Remove Weights: {w}')

            try:
                self.seq_model.load_state_dict(weights2)
            except:
                state2 = self.seq_model.state_dict()
                diff2 = list(set(state2.keys()).difference(set(weights2.keys())))
                self.print_log('Can not find these weights:')
                for d in diff2:
                    self.print_log('  ' + d)
                state2.update(weights2)
                self.seq_model.load_state_dict(state2)
        
        if self.arg.weights3:
            try:
                self.global_step = int(self.arg.weights3[:-3].split('-')[-1])
            except:
                print('Cannot parse global_step from model weights filename')
                self.global_step = 0

            self.print_log(f'Loading weights from {self.arg.weights3}')
            if '.pkl' in self.arg.weights3:
                with open(self.arg.weights3, 'r') as f:
                    weights3 = pickle.load(f)
            else:
                weights3 = torch.load(self.arg.weights3)

            weights3 = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights3.items()])

            for w in self.arg.ignore_weights3:
                if weights3.pop(w, None) is not None:
                    self.print_log(f'Sucessfully Remove Weights: {w}')
                else:
                    self.print_log(f'Can Not Remove Weights: {w}')

            try:
                self.vid_model.load_state_dict(weights3)
            except:
                state3 = self.vid_model.state_dict()
                diff3 = list(set(state3.keys()).difference(set(weights3.keys())))
                self.print_log('Can not find these weights:')
                for d in diff3:
                    self.print_log('  ' + d)
                state3.update(weights3)
                self.vid_model.load_state_dict(state3)
                
        self.L1loss = nn.L1Loss().cuda(output_device)
        self.L2loss = nn.MSELoss().cuda(output_device)
        if self.arg.class_loss:
            self.class_loss = nn.CrossEntropyLoss().cuda(output_device)
        if 'SKEL' in self.arg.stream:    
            self.print_log(f'GCN Model total number of params: {count_params(self.model)}')
            self.print_log(f'MLP Model total number of params: {count_params(self.seq_model)}')
        if 'RGB' in self.arg.stream:
            self.print_log(f'Video Model total number of params: {count_params(self.vid_model)}')

        
            
    def load_param_groups(self):
        """
        Template function for setting different learning behaviour
        (e.g. LR, weight decay) of different groups of parameters
        """
        if self.arg.stream == 'SKEL':
            self.param_groups = defaultdict(list)

            for name, params in self.model.named_parameters():
                if params.requires_grad:
                    self.param_groups['other'].append(params)

            self.optim_param_groups = {
                'other': {'params': self.param_groups['other']}
                }
        
            self.param_groups1 = defaultdict(list)

            for name, params in self.seq_model.named_parameters():
                if params.requires_grad:
                    self.param_groups1['other'].append(params)

            self.optim_param_groups1 = {
                'other': {'params': self.param_groups1['other']}
                }
    

    def load_optimizer(self):
        #params = list(self.optim_param_groups.values())
        if self.arg.stream == 'SKEL':
            if self.arg.pretrain_JFE:
                params = list(self.seq_model.parameters())
            else:
                params = list(self.model.parameters()) + list(self.seq_model.parameters())
            if self.arg.optimizer == 'SGD':
                self.optimizer = optim.SGD(
                    params,
                    lr=self.arg.base_lr,
                    momentum=0.9,
                    nesterov=self.arg.nesterov,
                    weight_decay=self.arg.weight_decay)
            elif self.arg.optimizer == 'Adam':
                self.optimizer = optim.Adam(
                    params,
                    lr=self.arg.base_lr,
                    weight_decay=self.arg.weight_decay)
        elif self.arg.stream == '2s':
            params = list(self.seq_model.parameters()) + list(self.vid_model.parameters())             
            if self.arg.optimizer == 'SGD':
                self.optimizer = optim.SGD(
                    params,
                    lr=self.arg.base_lr,
                    momentum=0.9,
                    nesterov=self.arg.nesterov,
                    weight_decay=self.arg.weight_decay)
            elif self.arg.optimizer == 'Adam':
                self.optimizer = optim.Adam(
                    params,
                    lr=self.arg.base_lr,
                    weight_decay=self.arg.weight_decay)
        elif self.arg.stream == 'RGB':
            params = list(self.vid_model.parameters()) 
            if self.arg.optimizer == 'SGD':
                self.optimizer = optim.SGD(
                    params,
                    lr=self.arg.base_lr,
                    momentum=0.9,
                    nesterov=self.arg.nesterov,
                    weight_decay=self.arg.weight_decay)
            elif self.arg.optimizer == 'Adam':
                self.optimizer = optim.Adam(
                    params,
                    lr=self.arg.base_lr,
                    weight_decay=self.arg.weight_decay)
        else:
            raise ValueError('Unsupported optimizer: {}'.format(self.arg.optimizer))

        # Load optimizer states if any
        if self.arg.checkpoint is not None:
            self.print_log(f'Loading optimizer states from: {self.arg.checkpoint}')
            self.optimizer.load_state_dict(torch.load(self.arg.checkpoint)['optimizer_states'])
            current_lr = self.optimizer.param_groups[0]['lr']
            self.print_log(f'Starting LR: {current_lr}')
            self.print_log(f'Starting WD1: {self.optimizer.param_groups[0]["weight_decay"]}')
            if len(self.optimizer.param_groups) >= 2:
                self.print_log(f'Starting WD2: {self.optimizer.param_groups[1]["weight_decay"]}')
    
    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def load_lr_scheduler(self):
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=self.arg.step, gamma=0.1)
        if self.arg.cosine_schedule:
            self.lr_scheduler = StepLR(self.optimizer, step_size=self.arg.step_size, gamma=0.1)        
            self.scheduler_warmup = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=self.arg.warmup_epoch, after_scheduler=self.lr_scheduler)
        if self.arg.checkpoint is not None:
            scheduler_states = torch.load(self.arg.checkpoint)['lr_scheduler_states']
            self.print_log(f'Loading LR scheduler states from: {self.arg.checkpoint}')
            self.lr_scheduler.load_state_dict(scheduler_states)
            self.print_log(f'Starting last epoch: {scheduler_states["last_epoch"]}')
            self.print_log(f'Loaded milestones: {scheduler_states["last_epoch"]}')
    
    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()

        def worker_seed_fn(worker_id):
            # give workers different seeds
            return init_seed(self.arg.seed + worker_id + 1)

        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=worker_seed_fn)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=worker_seed_fn)

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open(os.path.join(self.arg.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log(f'Local current time: {localtime}')

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(s, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def save_states(self, epoch, states, out_folder, out_name):
        out_folder_path = os.path.join(self.arg.work_dir, out_folder)
        out_path = os.path.join(out_folder_path, out_name)
        os.makedirs(out_folder_path, exist_ok=True)
        torch.save(states, out_path)

    def save_checkpoint(self, epoch, out_folder='checkpoints'):
        state_dict = {
            'epoch': epoch,
            'optimizer_states': self.optimizer.state_dict(),
            'lr_scheduler_states': self.lr_scheduler.state_dict(),
        }

        checkpoint_name = f'checkpoint-{epoch}-fwbz{self.arg.forward_batch_size}-{int(self.global_step)}.pt'
        self.save_states(epoch, state_dict, out_folder, checkpoint_name)

    def save_weights(self, epoch, out_folder='weights'):
        if 'SKEL' in self.arg.stream:
            state_dict1 = self.model.state_dict()
            weights1 = OrderedDict([
                [k.split('module.')[-1], v.cpu()]
                for k, v in state_dict1.items()
            ])
            weights_name1 = f'weights-{epoch}-{int(self.global_step)}.pt'
            self.save_states(epoch, weights1, out_folder, weights_name1)
            
            state_dict2 = self.seq_model.state_dict()
            weights2 = OrderedDict([
                [k.split('module.')[-1], v.cpu()]
                for k, v in state_dict2.items()
                ])

            weights_name2 = f'seq-weights-{epoch}-{int(self.global_step)}.pt'
            self.save_states(epoch, weights2, out_folder, weights_name2)
            
        if 'RGB' in self.arg.stream:
            state_dict3 = self.vid_model.state_dict()
            weights3 = OrderedDict([
                [k.split('module.')[-1], v.cpu()]
                for k, v in state_dict3.items()
                ])

            weights_name3 = f'vid-weights-{epoch}-{int(self.global_step)}.pt'
            self.save_states(epoch, weights3, out_folder, weights_name3)
        if '2s' in self.arg.stream:
            state_dict2 = self.seq_model.state_dict()
            weights2 = OrderedDict([
                [k.split('module.')[-1], v.cpu()]
                for k, v in state_dict2.items()
                ])

            weights_name2 = f'seq-weights-{epoch}-{int(self.global_step)}.pt'
            self.save_states(epoch, weights2, out_folder, weights_name2)
            
            state_dict3 = self.vid_model.state_dict()
            weights3 = OrderedDict([
                [k.split('module.')[-1], v.cpu()]
                for k, v in state_dict3.items()
                ])

            weights_name3 = f'vid-weights-{epoch}-{int(self.global_step)}.pt'
            self.save_states(epoch, weights3, out_folder, weights_name3)

        
    def train(self, epoch, save_model=False):
        if 'SKEL' in self.arg.stream:
            if self.arg.pretrain_JFE:
                self.model.eval()
            else:
                self.model.train()
            self.seq_model.train()
        if 'RGB' in self.arg.stream:
            self.vid_model.train()
        if self.arg.stream == '2s':
            self.seq_model.train()
            self.vid_model.train()
            
        loader = self.data_loader['train']
        
        if self.arg.warm_up:
            self.adjust_learning_rate(epoch)
        
        loss_values = []
        self.train_writer.add_scalar('epoch', epoch + 1, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        current_lr = self.optimizer.param_groups[0]['lr']
        self.print_log(f'Training epoch: {epoch + 1}, LR: {current_lr:.4f}')
        #torch.autograd.set_detect_anomaly(True)
        process = tqdm(loader, dynamic_ncols=True)
        if self.arg.cosine_schedule:
            self.scheduler_warmup.step(epoch+1)
        #print("start")
        for batch_idx, (data, label, clss, index) in enumerate(process):
            self.global_step += 1
            # get data
            with torch.no_grad():                     
                if self.arg.stream == '2s':
                    data1 = data[0].float().cuda(self.output_device)  
                    data2 = data[1].float().cuda(self.output_device)  
                else:
                    data1 = data.float().cuda(self.output_device)  
                if self.arg.class_loss:
                    clss = clss.long().cuda(self.output_device)
                label = label.float().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            # backward
            self.optimizer.zero_grad()
            '''
            if self.arg.cosine_schedule and epoch==0:
                #print("Warmup step\n")
                self.optimizer.step()
            '''
            ############## Gradient Accumulation for Smaller Batches ##############
            real_batch_size = self.arg.forward_batch_size
            splits = len(data1) // real_batch_size
            assert len(data1) % real_batch_size == 0, \
                'Real batch size should be a factor of arg.batch_size!'
                
            for i in range(splits):
                left = i * real_batch_size
                right = left + real_batch_size
                if self.arg.stream == '2s':
                    batch_data, batch_label = data1[left:right], label[left:right]
                    print(batch_data.shape)
                    batch_data2 = data2[left:right]
                    vid_feat, clss_pred_vid = self.vid_model(batch_data2)
                    out_score = self.seq_model(batch_data, vid_feat)
                    
                    
                if self.arg.stream == 'RGB':
                    batch_data, batch_label = data1[left:right], label[left:right]
                    if self.arg.class_loss:
                        batch_clss = clss[left:right]
                    # forward
                    out_score, clss_pred_vid = self.vid_model(batch_data)
                    
                elif 'SKEL' in self.arg.stream:
                    batch_data, batch_label = data1[left:right], label[left:right]
                    if self.arg.class_loss:
                        batch_clss = clss[left:right]
                    # clip based learning
                    seq_len = 0
                    for sq in range(7):
                        clip_data = batch_data[:,:,seq_len:seq_len+500,:]
                        if self.arg.class_loss:
                            output = self.model(clip_data)
                            #print(torch.is_tensor(clss_pred))
                        else:
                            output, _ = self.model(clip_data)
                        seq_len = seq_len + 500
                        if sq == 0:
                            features = output
                           
                        else:
                            features=torch.cat((features,output),dim=2)
                             
                    #print(torch.is_tensor(class_prediction)  )
                    if 'RGB' in self.arg.stream:
                        batch_data2 = data2[left:right]
                        vid_feat, clss_pred_vid = self.vid_model(batch_data2)
                        out_score = self.seq_model(features, vid_feat)
                    else:
                        out_score, class_prediction = self.seq_model(features)
                #print(class_prediction.shape)
                #print(class_prediction)
                if self.arg.class_loss:
                    if self.arg.stream == 'SKEL': 
                        clss_loss = self.class_loss(class_prediction, batch_clss)  
                    elif self.arg.stream == 'RGB':
                        clss_loss_vid = self.class_loss(clss_pred_vid, batch_clss)    
                        if 'SKEL' in self.arg.stream:
                            clss_loss = self.class_loss(class_prediction, batch_clss)  
                            clss_loss = clss_loss+clss_loss_vid
                        else:
                            clss_loss = clss_loss_vid
                L1loss = self.L1loss(out_score, batch_label) / splits
                L2loss = self.L2loss(out_score, batch_label) / splits
                
                ## standard deviation loss in difference
                #dif = torch.abs(out_score - batch_label)
                #mean = torch.mean(dif)
                #stdloss = torch.sqrt(torch.sum((dif-mean)**2)/len(out_score)+0.00001)
                
                
                
                if self.arg.class_loss:
                    loss = L2loss + 0.001*L1loss + clss_loss
                else:
                    loss = L2loss + 0.001*L1loss# + stdloss
                
                out = out_score.detach().cpu().numpy()
                lab = batch_label.detach().cpu().numpy()
                coef, _ = spearmanr(out, lab) 
                
                if self.arg.half:
                    with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                #loss.backward()

                loss_values.append(loss.item())
                timer['model'] += self.split_time()

                # Display loss
                process.set_description(f'(BS {real_batch_size}) loss: {loss.item():.4f}')

                
                #self.train_writer.add_scalar('Correlation', coef, self.global_step)
                self.train_writer.add_scalar('L2loss', L2loss.item() * splits, self.global_step)
                self.train_writer.add_scalar('L1loss', L1loss.item() * splits, self.global_step)
                self.train_writer.add_scalar('loss', loss.item() * splits, self.global_step)
                if self.arg.class_loss:
                    self.train_writer.add_scalar('Class loss', clss_loss.item() * splits, self.global_step)
                                
                
            #####################################

            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
            self.optimizer.step()

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

            # Delete output/loss after each batch since it may introduce extra mem during scoping
            # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770/3
            
        # statistics of time consumption and loss
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }

        mean_loss = np.mean(loss_values)
        num_splits = self.arg.batch_size // self.arg.forward_batch_size
        self.print_log(f'\tMean training loss: {mean_loss:.4f} (BS {self.arg.batch_size}: {mean_loss * num_splits:.4f}).')
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))
        
        if mean_loss < self.best_loss:
                self.best_loss = mean_loss
                self.best_loss_epoch = epoch + 1
                
        # PyTorch > 1.2.0: update LR scheduler here with `.step()`
        # and make sure to save the `lr_scheduler.state_dict()` as part of checkpoint
        if not self.arg.warm_up:
            self.lr_scheduler.step()

        if save_model:
            # save training checkpoint & weights
            self.save_weights(epoch + 1)
            if not self.arg.warm_up:
                self.save_checkpoint(epoch + 1)

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        # Skip evaluation if too early
        if epoch + 1 < self.arg.eval_start:
            return

        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        with torch.no_grad():
            if 'SKEL' in self.arg.stream:
                self.model = self.model.cuda(self.output_device)
                self.seq_model = self.seq_model.cuda(self.output_device)
                self.model.eval()
                self.seq_model.eval()
            if 'RGB' in self.arg.stream:
                self.vid_model = self.vid_model.cuda(self.output_device)
                self.vid_model.eval()
            if self.arg.stream == '2s':
                self.seq_model = self.seq_model.cuda(self.output_device)
                self.seq_model.eval()
                self.vid_model = self.vid_model.cuda(self.output_device)
                self.vid_model.eval()
                
            self.print_log(f'Eval epoch: {epoch + 1}')
            for ln in loader_name:
                loss_values = []
                score_batches = []
                step = 0
                label_list = []
                pred_list = []
                process = tqdm(self.data_loader[ln], dynamic_ncols=True)
                for batch_idx, (data, label, _, index) in enumerate(process):
                    label_list.append(label)
                    if self.arg.stream == '2s':
                        data1 = data[0].float().cuda(self.output_device)
                        data2 = data[1].float().cuda(self.output_device)
                    else:
                        data1 = data.float().cuda(self.output_device)
                    label = label.float().cuda(self.output_device)
                    #print(data1.shape)
                    start = datetime.datetime.now()
                    if self.arg.stream == '2s':
                        vid_feat, _ = self.vid_model(data2)
                        out_score = self.seq_model(data1, vid_feat)
                        
                        
                    if self.arg.stream == 'RGB':
                        out_score, _ = self.vid_model(data1)
                    elif 'SKEL' in self.arg.stream:
                        seq_len = 0
                        for sq in range(7):
                            clip_data = data1[:,:,seq_len:seq_len+500,:]
                            output = self.model(clip_data)
                            seq_len = seq_len + 500
                            if sq == 0:
                                features = output
                            else:
                                features=torch.cat((features,output),dim=2)   
                       # out_path = "/home/uniwa/students3/students/22905553/linux/phd_codes/Olympics/Data/MSG3D_extracted_joint_features/"
                        #with open(f'{out_path}{index[0]}_joint_features.pkl', 'wb') as f:
                         #   pickle.dump((index[0], features.detach().cpu()), f)
                        if 'RGB' in self.arg.stream:
                            vid_feat, _ = self.vid_model(data2)
                            out_score, _ = self.seq_model(features, vid_feat)
                        else:
                            out_score, _ = self.seq_model(features)
                    end = datetime.datetime.now()
                    delta = end - start
                    print(int(delta.total_seconds()* 1000))
                    #    out_score = self.seq_model(features, 0)
                        
                    
                    
    
                    loss = self.L2loss(out_score, label)
                    score_batches.append(out_score.data.cpu().numpy())
                    loss_values.append(loss.item())

                    step += 1

                    if wrong_file is not None or result_file is not None:
                        predict = list(out_score.cpu().numpy())
                        true = list(label.data.cpu().numpy())
                        for i, x in enumerate(predict):
                            if result_file is not None:
                                f_r.write(str(x) + ',' + str(true[i]) + '\n')
                            if x != true[i] and wrong_file is not None:
                                f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(score_batches)
            label_list = np.concatenate(label_list)
            loss = np.mean(loss_values)
            coef, p = spearmanr(score, label_list)
            if coef > self.best_acc:
                    self.best_acc = coef
                    self.best_acc_epoch = epoch + 1
            print('Spearman’s Rank Correlation: ', coef, ' model: ', self.arg.work_dir)
            self.print_log(f'Test: Spearman’s Rank Correlation: {coef}')
            alpha = 0.05
            if p > alpha:
                self.print_log(f'Samples are uncorrelated (fail to reject H0) p={p}')            	
            else:
                self.print_log(f'Samples are correlated (reject H0) p={p}')
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('L2loss', loss, self.global_step)
                #self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                self.val_writer.add_scalar('Correlation', coef, self.global_step)

            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log(f'\tMean {ln} loss of {len(self.data_loader[ln])} batches: {np.mean(loss_values)}.')
            #for k in self.arg.show_topk:
            #    self.print_log(f'\tTop {k}: {100 * self.data_loader[ln].dataset.top_k(score, k):.2f}%')

            if save_score:
                with open('{}/score/epoch{}_{}_score.pkl'.format(self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)
            
            

        # Empty cache after evaluation
        torch.cuda.empty_cache()

    def start(self):
        if self.arg.phase == 'train':
            self.print_log(f'Parameters:\n{pprint.pformat(vars(self.arg))}\n')
            if 'SKEL' in self.arg.stream:
                self.print_log(f'Model total number of params: {count_params(self.model)}')
                num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            elif 'RGB' in self.arg.stream:
                    self.print_log(f'Model total number of params: {count_params(self.vid_model)}')
                    num_params = sum(p.numel() for p in self.vid_model.parameters() if p.requires_grad)
            elif '2s' in self.arg.stream:
                    self.print_log(f'vid_model total number of params: {count_params(self.vid_model)}')
                    self.print_log(f'Seq_model total number of params: {count_params(self.seq_model)}')
                    num_params = sum(p.numel() for p in self.vid_model.parameters() if p.requires_grad) + sum(p.numel() for p in self.seq_model.parameters() if p.requires_grad)
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch)
                self.train(epoch, save_model=save_model)
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Lowest loss: {self.best_loss}')
            self.print_log(f'Lowest loss epoch number: {self.best_loss_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Forward Batch Size: {self.arg.forward_batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = os.path.join(self.arg.work_dir, 'wrong-samples.txt')
                rf = os.path.join(self.arg.work_dir, 'right-samples.txt')
            else:
                wf = rf = None
            if 'SKEL' in self.arg.stream:
                if self.arg.weights1 is None and self.arg.weights2 is None:
                    raise ValueError('Please appoint --weights.')
                self.print_log(f'Model:   {self.arg.model1}')
                self.print_log(f'Weights: {self.arg.weights1}')
                self.print_log(f'Model:   {self.arg.model2}')
                self.print_log(f'Weights: {self.arg.weights2}')
            if 'RGB' in self.arg.stream:
                if self.arg.weights3 is None:
                    raise ValueError('Please appoint --weights.')
                self.print_log(f'Model:   {self.arg.model3}')
                self.print_log(f'Weights: {self.arg.weights3}')

            self.eval(
                epoch=0,
                save_score=self.arg.save_score,
                loader_name=['test'],
                wrong_file=wf,
                result_file=rf
            )

            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()


if __name__ == '__main__':
    main()