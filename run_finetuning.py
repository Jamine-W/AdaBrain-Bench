# --------------------------------------------------------
# Based on LaBraM, EEGPT, CBraMod, BIOT, EEG_Image_decode, BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/935963004/LaBraM
# https://github.com/BINE022/EEGPT/tree/main/downstream
# https://github.com/wjq-learning/CBraMod
# https://github.com/ycq091044/BIOT
# https://github.com/ncclab-sustech/EEG_Image_decode
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import argparse
import datetime
from pyexpat import model
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import json
import os

from pathlib import Path
from collections import OrderedDict
from timm.models import create_model
from util.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from util.utils import NativeScalerWithGradNormCount as NativeScaler
import util.utils as utils
from util.eegdatasets import EEGDataset
from engine_for_finetuning import train_one_epoch, evaluate, main_train_loop
import csv
from functools import partial
from models.cbramod import CBraMod
from models.EEGPT_mcae import EEGTransformer, Conv1dWithConstraint, LinearWithConstraint
from models.biot import BIOTClassifier
from models.EEGNet import EEGNet
from models.LMDA import LMDA
from models.EEGConformer import Conformer
from models.EEGTransformer import STTransformer
from models.loss import ClipLoss

from torch.utils.data import random_split, ConcatDataset

# -------------------------------The pre-trained weights of the foundation model---------------------------------------
finetune_list = {
    'LaBraM': './checkpoints/labram-base.pth',
    'CBraMod': './checkpoints/pretrained_weights.pth',
    'EEGPT': './checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt',
    'BIOT': "./checkpoints/EEG-six-datasets-18-channels.ckpt"
}
# ---------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------Parameters------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()
    # Fine-tuning parameters
    parser.add_argument('--dataset', default='SEED', type=str, 
                        choices=['SEED', 'SEED-IV', 'BCI-IV-2A', 'SHU', 'SEED-VIG', 'EEGMAT',
                                 'Sleep-EDF', 'HMC', 'SHHS', 'TUAB', 'TUEV', 'Things-EEG'])
    parser.add_argument('--model_name', default='LaBraM', type=str,
                        choices=['LaBraM', 'CBraMod', 'EEGPT', 'BIOT', 'EEGNet', 'LMDA', 'EEGConformer', 'ST-Transformer'])
    parser.add_argument('--task_mod', default="Classification", type=str, choices=['Classification', 'Regression', 'Retrieval'],
                        help='type of task')
    parser.add_argument('--subject_mod', default="multi", type=str, choices=['multi', 'cross', 'fewshot', 'single'],
                        help='evaluation settings including cross-subject, multi-subject and few-shot settings (single-subject setting for retrieval task)')
    parser.add_argument('--finetune_mod', default='full', type=str, choices=['full', 'linear', 'all'], help='model finetune mod')
    parser.add_argument('--nb_classes', default=0, type=int, help='number of classes in the datasets (classification task)')
    parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
    parser.add_argument('--epochs', default=50, type=int, help='epoches for training')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--norm_method', default='z_score', type=str, choices=['z_score', '0.1mv', '95'],
                        help='normalization methods including z-score, 95-percentile, and unit rescale (0.1mv)')
    
    parser.add_argument('--max_subject', default=8, type=int, help='number of subjects used for spliting validation set')
    parser.add_argument('--sampling_rate', default=200, type=int, choices=[200, 256], help='BIOT, LaBraM and CBraMod is 200Hz; EEGPT is 256Hz')
    parser.add_argument('--k_shot', default=10, type=float, help='number of shots in the few_shot setting')
    

    parser.add_argument('--logger', type=bool, default=False, help='enable WandB logging for retrieval')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--save_ckpt', action='store_true', default=True)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pin_mem', action='store_true', default=True,
                        help='pin CPU memory in dataLoader for more efficient (sometimes) transfer to GPU')
    parser.add_argument('--mv_norm_value', default=0.01, type=float,
                        help='scale_value when using 0.1mv norm_method, default is 0.01.')
    parser.add_argument('--subject_id', type=int, default=1, help='subject id for single subject retrieval task')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--update_freq', default=1, type=int)


    # Distributed training parameters
    parser.add_argument('--distributed', default=False)
    parser.add_argument('--dist_eval', action='store_true', default=False, help='enabling distributed evaluation')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--auto_resume', action='store_true', default=True)

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init
# -------------------------------------------------------------------------------------------------------------

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, flatten=0, dropout=0, patch_mean=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        self.flatten = flatten
        self.patch_mean = patch_mean
        self.drop_out = nn.Dropout(p=dropout) if dropout else None
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )

    def forward(self, x):
        if self.flatten:
            x = x.flatten(self.flatten)
        elif self.patch_mean:
            x = x.reshape(x.shape[0], -1, x.shape[-1]).mean(1)
        if self.drop_out is not None:
            x = self.drop_out(x)
        # if self.doWeightNorm:
        #     self.weight.data = torch.renorm(
        #         self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        #     )
        return super().forward(x)

class RegressionLayers(nn.Sequential):
    def __init__(self, input_dim, hidden_dim, output_dim, flatten=0, patch_mean=False, remove_cls=False):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        self.flatten = flatten
        self.patch_mean = patch_mean
        self.remove_cls = remove_cls
    def forward(self, x):
        if self.remove_cls:
            x = x[..., 1:, :]
        if self.flatten:
            x = x.flatten(self.flatten)
        elif self.patch_mean:
            x = x.reshape(x.shape[0], -1, x.shape[-1]).mean(1)
        out = self.clshead(x)
        return out

# -------------------------------------------------------------------------------------------------------------

# -----------------------------------------Custom Classes for models---------------------------------------------


class Ada_LaBraM(nn.Module):
    def __init__(self, args, ch_names, num_t, from_pretrain=False):
        super().__init__()
        # Load LaBraM model
        model = create_model(
            'labram_base_patch200_200',
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=0.0,
            drop_path_rate=0.1,
            attn_drop_rate=0.0,
            drop_block_rate=None,
            use_mean_pooling=True,
            init_scale=0.001,
            use_rel_pos_bias=True,
            use_abs_pos_emb=True,
            init_values=0.1,
            qkv_bias=True,
            num_ch=len(ch_names),
            num_t=num_t
        )
        

        # load the pre-trained weights.
        if from_pretrain:
            if finetune_list[args.model_name].startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    finetune_list[args.model_name], map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(finetune_list[args.model_name], map_location='cpu')

            print("Load ckpt from %s" % finetune_list[args.model_name])
            checkpoint_model = None
            args.model_key = 'model|module'
            for model_key in args.model_key.split('|'):
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print("Load state_dict by model_key = %s" % model_key)
                    break
            if checkpoint_model is None:
                checkpoint_model = checkpoint
            if (checkpoint_model is not None):
                all_keys = list(checkpoint_model.keys())
                new_dict = OrderedDict()
                for key in all_keys:
                    if key.startswith('student.'):
                        new_dict[key[8:]] = checkpoint_model[key]
                    else:
                        pass
                checkpoint_model = new_dict

            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            all_keys = list(checkpoint_model.keys())
            for key in all_keys:
                if "relative_position_index" in key:
                    checkpoint_model.pop(key)

            utils.load_state_dict(model, checkpoint_model)
        
        
        
        model.head = nn.Identity()
        self.main_model = model
        self.ch_names = ch_names

        self.task_head=nn.Identity()
        
    
    def forward(self, x):
        b, n, t = x.shape
        x = x.reshape(b, n, -1, 200)
        input_chans = utils.get_input_chans(self.ch_names)
        output = self.main_model(x, input_chans, return_all_tokens=True)
        output = self.task_head(output)
        return output

class Ada_CBraMod(nn.Module):
    def __init__(self, args, from_pretrain=False):
        super().__init__()
        model = CBraMod()
        
        if from_pretrain:
            print("Load ckpt from %s" % finetune_list[args.model_name])
            model.load_state_dict(torch.load(finetune_list[args.model_name], map_location=torch.device('cpu')))
        
        model.proj_out = nn.Identity()
        self.main_model = model

        self.task_head=nn.Identity()
        
    
    def forward(self, x):
        b, n, t = x.shape
        x = x.reshape(b, n, -1, 200)
        output = self.main_model(x)
        output = self.task_head(output)
        return output

class Ada_EEGPT(nn.Module):
    def __init__(self, args, ch_names, num_t, from_pretrain=False):
        super().__init__()

        with open("./util/eegpt_use_channels_names.json", "r") as f:
            model_channels = json.load(f)
        use_channels_names = model_channels.get(args.dataset, None)
        use_channels_names = use_channels_names.split(", ") if use_channels_names is not None else ch_names
        chans_num = len(use_channels_names)

        # init model
        model = EEGTransformer(
            img_size=[chans_num, 256 * num_t],
            patch_size=32 * 2,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.chans_id = model.prepare_chan_ids(use_channels_names)

        if from_pretrain:
            print(f"Load ckpt from {finetune_list[args.model_name]}")
            checkpoint_path = finetune_list[args.model_name]
            pretrain_ckpt = torch.load(checkpoint_path)
            target_encoder_stat = {}
            for k, v in pretrain_ckpt['state_dict'].items():
                if k.startswith("target_encoder."):
                    target_encoder_stat[k[15:]] = v
            model.load_state_dict(target_encoder_stat)
        
        self.main_model = model
        self.chan_conv = Conv1dWithConstraint(len(ch_names), chans_num, 1, max_norm=1)
        self.task_head = nn.Identity()

    def forward(self, x):
        x = self.chan_conv(x)
        output = self.main_model(x, self.chans_id.to(x))
        output = self.task_head(output)
        return output

class Ada_BIOT(nn.Module):
    def __init__(self, args, ch_names, from_pretrain=False):
        super().__init__()
        in_channels = 18

        model = BIOTClassifier(n_classes=args.nb_classes, n_channels=in_channels, n_fft=200, hop_length=100)
        if from_pretrain:
            model.biot.load_state_dict(torch.load(finetune_list[args.model_name]))
            print(f"load pretrain model from {finetune_list[args.model_name]}")
        
        model.classifier = nn.Identity()
        self.main_model = model
        self.chan_conv = Conv1dWithConstraint(len(ch_names), in_channels, 1, max_norm=1)

        self.task_head=nn.Identity()

    def forward(self, x):
        x = self.chan_conv(x)
        output = self.main_model(x)
        output = self.task_head(output)
        return output

class Ada_EEGNet(nn.Module):
    def __init__(self, args, ch_names, num_t):
        super().__init__()
        model = EEGNet(chans=len(ch_names), classes=args.nb_classes, time_points=num_t * 200)
        model.fc = nn.Identity()
        self.main_model = model

        self.task_head=nn.Identity()

    def forward(self, x):
        output = self.main_model(x)
        output = self.task_head(output)
        return output

class Ada_LMDA(nn.Module):
    def __init__(self, args, ch_names, num_t):
        super().__init__()
        model = LMDA(num_classes=args.nb_classes, chans=len(ch_names), samples=num_t * 200, channel_depth1=24, channel_depth2=7)
        model.classifier = nn.Identity()
        self.main_model = model

        self.task_head=nn.Identity()

    def forward(self, x):
        output = self.main_model(x)
        output = self.task_head(output)
        return output

class Ada_EEGConformer(nn.Module):
    def __init__(self, args, ch_names, num_t):
        super().__init__()
        model = Conformer(C=len(ch_names), time_points=num_t * 200, n_classes=args.nb_classes)
        model.classification_head = nn.Identity()
        self.main_model = model

        self.task_head=nn.Identity()

    def forward(self, x):
        output = self.main_model(x)
        output = self.task_head(output)
        return output

class Ada_STTransformer(nn.Module):
    def __init__(self, args, ch_names, num_t):
        super().__init__()
        model = STTransformer(n_classes=args.nb_classes, channel_legnth=num_t * 200, n_channels=len(ch_names))
        self.main_model = model
        self.task_head=nn.Identity()
        
    def forward(self, x):
        output = self.main_model(x)
        output = self.task_head(output)
        return output

# --------------------------------------------------------------------------------------------------

# -----------------------------Load the models based on args.model_name------------------------------
def get_models(args, ch_names, num_t):
    from_pretrain = False
    if args.model_name in ['LaBraM', 'CBraMod', 'EEGPT', 'BIOT']:
        if args.finetune_mod in ['full', 'linear']:
            from_pretrain=True
 
    # init models
    if args.model_name == 'LaBraM':
        model = Ada_LaBraM(args, ch_names, num_t, from_pretrain)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint((len(ch_names) * num_t + 1) * 200, args.nb_classes, max_norm=1, flatten=1)
        elif args.task_mod == 'Regression':
            model.task_head = RegressionLayers(input_dim=200, hidden_dim=200, output_dim=1, patch_mean=True, remove_cls=True)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint((len(ch_names) * num_t + 1) * 200, 1024, max_norm=1, flatten=1)
    elif args.model_name == 'CBraMod':
        model = Ada_CBraMod(args, from_pretrain)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint(len(ch_names) * num_t * 200, args.nb_classes, max_norm=1, flatten=1)
        elif args.task_mod == 'Regression':
            model.task_head = RegressionLayers(input_dim=(len(ch_names) * num_t) * 200, hidden_dim=200, output_dim=1, flatten=1)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(len(ch_names) * num_t * 200, 1024, max_norm=1, flatten=1)
    elif args.model_name == 'EEGPT':
        model = Ada_EEGPT(args, ch_names, num_t, from_pretrain)
        if args.task_mod == 'Classification':
            model.task_head = nn.Sequential(
                LinearWithConstraint(2048, 16, max_norm=1, flatten=2, dropout=0.5),
                LinearWithConstraint(4 * num_t * 16, args.nb_classes, max_norm=0.25, flatten=1)
            )
        elif args.task_mod == 'Regression':
            model.task_head = RegressionLayers(input_dim=512, hidden_dim=256, output_dim=1, patch_mean=True)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(4 * 2048, 1024, max_norm=1, flatten=1)
    elif args.model_name == 'BIOT':
        model = Ada_BIOT(args, ch_names, from_pretrain=from_pretrain)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint(256, args.nb_classes, max_norm=1)
        elif args.task_mod == 'Regression':
            model.task_head = RegressionLayers(input_dim=256, hidden_dim=256, output_dim=1)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(256, 1024, max_norm=1)
    elif args.model_name == 'EEGNet':
        model = Ada_EEGNet(args, ch_names, num_t)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint(model.linear_size, args.nb_classes, max_norm=1)
        elif args.task_mod == 'Regression':
            model.task_head = RegressionLayers(input_dim=model.linear_size, hidden_dim=200, output_dim=1)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(model.linear_size, 1024, max_norm=1)
    elif args.model_name == 'LMDA':
        model = Ada_LMDA(args, ch_names, num_t)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint(model.linear_size, args.nb_classes, max_norm=1)
        elif args.task_mod == 'Regression':
            model.task_head = RegressionLayers(input_dim=model.linear_size, hidden_dim=200, output_dim=1)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(model.linear_size, 1024, max_norm=1)
    elif args.model_name == 'EEGConformer':
        model = Ada_EEGConformer(args, ch_names, num_t)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint(model.time_points * 40, args.nb_classes, max_norm=1)
        elif args.task_mod == 'Regression':
            model.task_head = RegressionLayers(input_dim=model.time_points * 40, hidden_dim=40, output_dim=1)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(model.time_points * 40, 1024, max_norm=1)
    elif args.model_name == 'ST-Transformer':
        model = Ada_STTransformer(args, ch_names, num_t)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint(256, args.nb_classes, max_norm=1)
        elif args.task_mod == 'Regression':
            model.task_head = RegressionLayers(input_dim=256, hidden_dim=256, output_dim=1)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(256, 1024, max_norm=1)
    else:
        print("Unknown model name!")
        exit(0)
    
    # check task head
    if model.task_head is None:
        print("Task head is None, please check your args or code.")
        exit(0)
    
    if args.finetune_mod == 'linear':
        for p in model.main_model.parameters():
            p.requires_grad = False
    
    # add modules for retrieval
    if args.task_mod == 'Retrieval':
        model.loss_scale = nn.Parameter(torch.tensor(1.0))
        model.loss_func = ClipLoss()

    return model
# ----------------------------------------------------------------------------------------------------------------

# ------------------------------------------Load the dataset-------------------------------------------------------
def get_datasets(args, dataset_info):
    root = dataset_info['root'][args.subject_mod]
    if args.subject_mod == 'fewshot':
        dataset_train = utils.FewShotDataLoader(root + '/train.json', args.sampling_rate, args.norm_method, k_shot=args.k_shot)
        dataset_val = utils.CustomDataLoader(root + '/val.json', args.sampling_rate, args.norm_method)
    else:
        if os.path.exists(root + '/val.json'):
            dataset_train = utils.CustomDataLoader(root + '/train.json', args.sampling_rate, args.norm_method)
            dataset_val = utils.CustomDataLoader(root + '/val.json', args.sampling_rate, args.norm_method)
        else:
            dataset_train = None
            dataset_val = None
            for i in range(args.max_subject):
                subject_dataset = utils.CustomDataLoader(root + '/train.json', args.sampling_rate, args.norm_method, cross=True, subject_id=i)
                train_size = int(0.8 * len(subject_dataset))
                valid_size = len(subject_dataset) - train_size
                train_dataset, valid_dataset = random_split(subject_dataset, [train_size, valid_size])
                if dataset_train is None:
                    dataset_train = train_dataset
                    dataset_val = valid_dataset
                else:
                    dataset_train = ConcatDataset([dataset_train, train_dataset])
                    dataset_val = ConcatDataset([dataset_val, valid_dataset])
    
    dataset_test = utils.CustomDataLoader(root + '/test.json', args.sampling_rate, args.norm_method)
    ch_names = dataset_test.get_ch_names()
    ch_names = [ch.upper() for ch in ch_names]
    args.nb_classes = dataset_info['num_classes']
    if args.nb_classes == 2:
        args.nb_classes = 1
    return dataset_train, dataset_test, dataset_val, ch_names
# -------------------------------------------------------------------------------------------------------------

# -------------------------------Main function for fine-tuning-------------------------------------------------
def main(args, ds_init):

    if ds_init is not None:
        utils.create_ds_config(args)

    args.save_ckpt_freq = args.epochs

    args.output_dir = f"finetuning_results/{args.task_mod}/{args.model_name}_results/finetune_{args.finetune_mod}/{args.dataset}_{args.finetune_mod}_epoch{args.epochs}_bs{args.batch_size}_lr{args.lr}_{args.norm_method}_{args.seed}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # get dataset
    with open(f'./dataset_config/{args.task_mod}.json', 'r') as file:
        data = json.load(file)
    dataset_info = data.get(args.dataset)
    if args.task_mod == 'Retrieval':
        os.environ["WANDB_API_KEY"] = ""
        os.environ["WANDB_MODE"] = 'offline'
        dataset_train = EEGDataset(args.dataset, train=True, subject_mod=args.subject_mod, subject_id=args.subject_id, sampling_rate=args.sampling_rate, norm_method=args.norm_method)
        dataset_test = EEGDataset(args.dataset, train=False, subject_mod=args.subject_mod, subject_id=args.subject_id, sampling_rate=args.sampling_rate, norm_method=args.norm_method)
        dataset_val = None
        ch_names = dataset_train.get_ch_names()
    else:
        dataset_train, dataset_test, dataset_val, ch_names = get_datasets(args, dataset_info)

    # ----------------------------Get dataloaders.--------------------------------
    if args.disable_eval_during_finetuning:
        dataset_val = None
        dataset_test = None

    # if True:  # args.distributed:
    global_rank = 0
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            if type(dataset_test) == list:
                sampler_test = [torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True) for dataset in dataset_test]
            else:
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val is not None else None
            sampler_test = torch.utils.data.SequentialSampler(dataset_test) if dataset_test is not None else None
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)



    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None
    
    if dataset_test is not None:
        if type(dataset_test) == list:
            data_loader_test = [torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                batch_size=int(args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            ) for dataset, sampler in zip(dataset_test, sampler_test)]
        else:
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=int(args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
    else:
        data_loader_test = None
    # ------------------------------------------------------------------------------------------

    # load the model
    model = get_models(args, ch_names, dataset_info['num_t'])
    model.to(device)
    # model_ema = None
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    # layer_decay for labram and cbramod
    # if args.layer_decay < 1.0:
    if args.model_name in ['LaBraM', 'CBraMod']:
        if args.model_name == 'LaBraM':
            num_layers = model_without_ddp.main_model.get_num_layers()
        elif args.model_name == 'CBraMod':
            num_layers = len(model_without_ddp.main_model.encoder.layers)
        else:
            print("Layer_decay is not supported by the model. ")
            exit(0)
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = []

    # get optimizer, lr_scheduler...
    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()


    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    # load checkpoint for resume
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler)
    
    # start finetuning
    print(f"Start training for {args.epochs} epochs")

    if args.task_mod == 'Retrieval':
        current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
        img_features_train_all = dataset_train.img_features
        img_features_test_all = dataset_test.img_features
        results = main_train_loop(
            args, current_time, model, data_loader_train, data_loader_test, optimizer, device, 
            img_features_train_all, img_features_test_all, config=args, loss_scaler=loss_scaler, 
            logger=args.logger, lr_schedule_values=lr_schedule_values, ch_names=ch_names,
            wd_schedule_values=wd_schedule_values, num_training_steps_per_epoch=num_training_steps_per_epoch)
        
        # Save results to a CSV file
        results_dir = os.path.join(args.output_dir, current_time)
        os.makedirs(results_dir, exist_ok=True)

        results_file = f"{results_dir}/results.csv"
        with open(results_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            print(f'Results saved to {results_file}')
    else:
        start_time = time.time()
        max_accuracy = 0.0
        max_accuracy_test = 0.0
        max_r2 = 0.0
        max_r2_test = 0.0

        # metrics: list of strings, the metrics you want to use. We utilize PyHealth to implement it.
        if args.task_mod == 'Regression':
            metrics = ["Pearson_Correlation", 'R2_Score', 'RMSE']
        elif args.nb_classes > 1:
            metrics = ["accuracy", 'balanced_accuracy', 'f1_weighted', 'cohen_kappa']
        else:
            metrics = ["accuracy", 'balanced_accuracy', 'pr_auc', 'roc_auc']

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            
            # see engine_for_finetuning.py
            train_stats = train_one_epoch(
                args, model, data_loader_train, optimizer,
                device, epoch, loss_scaler, 
                start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
                num_training_steps_per_epoch=num_training_steps_per_epoch, ch_names=ch_names
            )
            
            # save checkpoint
            if args.output_dir and args.save_ckpt:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, save_ckpt_freq=args.save_ckpt_freq)
                
            # val and test
            if data_loader_val is not None:
                val_stats = evaluate(args, data_loader_val, model, device, header='Val:', ch_names=ch_names, metrics=metrics)
                test_stats = evaluate(args, data_loader_test, model, device, header='Test:', ch_names=ch_names, metrics=metrics)

                if args.task_mod == 'Classification':
                    print(f"Accuracy on the val set: {val_stats['accuracy']*100:.2f}%")
                    print(f"Accuracy on the test set: {test_stats['accuracy']*100:.2f}%")
                else:
                    print(f"R2_Score on the val set: {val_stats['R2_Score']:.2f}")
                    print(f"R2_Score on the test set: {test_stats['R2_Score']:.2f}")
                
                # save best checkpoint
                if args.task_mod == 'Classification':
                    if max_accuracy < val_stats["accuracy"]:
                        max_accuracy = val_stats["accuracy"]
                        if args.output_dir and args.save_ckpt:
                            utils.save_model(
                                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch="best")
                        max_accuracy_test = test_stats["accuracy"]
                    print(f'Max accuracy val: {max_accuracy*100:.2f} %, max accuracy test: {max_accuracy_test*100:.2f} %')
                else:
                    if max_r2 < val_stats["R2_Score"]:
                        max_r2 = val_stats["R2_Score"]
                        if args.output_dir and args.save_ckpt:
                            utils.save_model(
                                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch="best")
                        max_r2_test = test_stats["R2_Score"]
                    print(f'Max R2_Score val: {max_r2:.2f}, max R2_Score test: {max_r2_test:.2f}')
                        
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'val_{k}': v for k, v in val_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    opts, ds_init = get_args()
    main(opts, ds_init)
