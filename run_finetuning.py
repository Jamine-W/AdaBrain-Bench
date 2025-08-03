# --------------------------------------------------------
# Ada-Brainbench
# Based on LaBraM, EEG_Image_decode, BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/935963004/LaBraM
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
from timm.utils import ModelEma

from util.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from util.utils import NativeScalerWithGradNormCount as NativeScaler
import util.utils as utils
from util.eegdatasets import EEGDataset

from engine_for_finetuning import train_one_epoch, evaluate, main_train_loop
from scipy import interpolate
import csv

from functools import partial
import models.modeling_finetune
from models.cbramod import CBraMod
from models.EEGPT_mcae import EEGTransformer, Conv1dWithConstraint, LinearWithConstraint
from models.biot import BIOTClassifier
from models.EEGNet import EEGNet
from models.LMDA import LMDA
from models.EEGConformer import Conformer
from models.EEGTransformer import STTransformer
from models.loss import ClipLoss

from torch.utils.data import random_split, ConcatDataset
from einops.layers.torch import Rearrange

# -------------------------------The pre-trained weights of the foundation model---------------------------------------
fintune_list = {
    'LaBraM': './checkpoints/labram-base.pth',
    'CBraMod': './checkpoints/pretrained_weights.pth',
    'EEGPT': './checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt',
    'BIOT': [
        "./checkpoints/EEG-PREST-16-channels.ckpt",
        "./checkpoints/EEG-SHHS+PREST-18-channels.ckpt",
        "./checkpoints/EEG-six-datasets-18-channels.ckpt"
    ]
}
# ---------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------Parameters------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SEED', type=str, help='dataset', 
                        choices=["SEED", "SEED-IV", "BCI-IV-2A", "SHU", "SEED-VIG", "EEGMAT", 
                                 "Sleep-EDF", "HMC", "SHHS", "TUAB", "TUEV", "Things-EEG"])
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=0, type=int)

    # robust evaluation
    parser.add_argument('--robust_test', default=None, type=str,
                        help='robust evaluation dataset')
    
    # Model parameters
    parser.add_argument('--model', default='labram_base_patch200_200', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=True)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=200, type=int,
                        help='EEG input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=1.0)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Finetuning params
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    # Dataset parameters
    parser.add_argument('--nb_classes', default=0, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--finetune_mod', default='full', type=str,
                        choices=['full', 'linear', 'all'],
                        help='model finetune mod')
    parser.add_argument('--norm_method', default=" ", type=str,
                        choices=['z_score', '95', 'min_max', 'ems', '0.1mv', 'mv'],
                        help='normalization method')
    parser.add_argument('--mv_norm_value', default=0.01, type=float,
                        help='scale_value when using mv norm_method, default is 0.01.')
    parser.add_argument('--subject_mod', default="multi", type=str,
                        choices=['multi', 'cross', 'fewshot', 'single', 'loso'],
                        help='multi_subject, cross_subject, few_shot')
    parser.add_argument('--max_subject', default=8, type=int,
                        help='max_subject_id')
    parser.add_argument('--model_name', default="LaBraM", type=str,
                        help='Name of models. You can choose from: LaBraM, CBraMod, EEGPT, BIOT, EEGNet, LMDA, EEGConformer, ST-Transformer')
    parser.add_argument("--use_channels_names", default=None, type=str,
                        help='follow the EEGPT method')
    parser.add_argument('--sampling_rate', default=200, type=int,
                        help='sampling rate of the pre-trained models')
    parser.add_argument('--pretrain_model_choice', default=2, type=int, choices=[0, 1, 2],
                        help='pretrain_model_choice for BIOT')
    parser.add_argument('--k_shot', default=10, type=float,
                        help='k_shot for few_shot task')
    parser.add_argument('--task_mod', default="Classification", type=str,
                        choices=['Classification', 'Regression', 'Retrieval'],
                        help='Task name')
    parser.add_argument('--subject_id', type=int, default=1, help='subject id for single subject retrieval task')
    parser.add_argument('--logger', type=bool, default=True, help='Enable WandB logging for retrieval')
    parser.add_argument('--project', type=str, default="Test", help='WandB project name for retrieval')
    parser.add_argument('--name', type=str, default="img_pos_pro_eeg", help='Experiment name for retrieval')

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

# ----------------------------Predefined heads for fine-tuning tasks.------------------------------
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, flatten=0, dropout=0, patch_mean=False, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        self.flatten = flatten
        self.patch_mean = patch_mean
        self.drop_out = nn.Dropout(p=dropout) if dropout else None
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.flatten:
            x = x.flatten(self.flatten)
        elif self.patch_mean:
            x = x.reshape(x.shape[0], -1, x.shape[-1]).mean(1)
        if self.drop_out is not None:
            x = self.drop_out(x)
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

class Linear3Layers(nn.Sequential):
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

class Custom_LaBraM(nn.Module):
    def __init__(self, args, ch_names, num_t, from_pretrain=False):
        super().__init__()
        # Load LaBraM model
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            use_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            qkv_bias=args.qkv_bias,
            num_ch=len(ch_names),
            num_t=num_t
        )
        # If required, load the pre-trained weights.
        if from_pretrain:
            if fintune_list[args.model_name].startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    fintune_list[args.model_name], map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(fintune_list[args.model_name], map_location='cpu')

            print("Load ckpt from %s" % fintune_list[args.model_name])
            checkpoint_model = None
            for model_key in args.model_key.split('|'):
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print("Load state_dict by model_key = %s" % model_key)
                    break
            if checkpoint_model is None:
                checkpoint_model = checkpoint
            if (checkpoint_model is not None) and (args.model_filter_name != ''):
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

            utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
        
        model.head = nn.Identity()
        self.task_head = None
        self.main_model = model
        self.ch_names = ch_names
    def forward(self, x):
        b, n, t = x.shape
        x = x.reshape(b, n, -1, 200)
        input_chans = utils.get_input_chans(self.ch_names)
        y = self.main_model(x, input_chans, return_all_tokens=True)
        return self.task_head(y)

class Custom_CBraMod(nn.Module):
    def __init__(self, args, from_pretrain=False):
        super().__init__()
        model = CBraMod()
        if from_pretrain:
            print("Load ckpt from %s" % fintune_list[args.model_name])
            model.load_state_dict(torch.load(fintune_list[args.model_name], map_location=torch.device('cpu')))
        model.proj_out = nn.Identity()
        self.task_head = None
        self.main_model = model
    def forward(self, x):
        b, n, t = x.shape
        x = x.reshape(b, n, -1, 200)
        y = self.main_model(x)
        return self.task_head(y)

class Custom_EEGPT(nn.Module):
    def __init__(self, args, ch_names, num_t, from_pretrain=False):
        super().__init__()
        use_channels_names = args.use_channels_names.split(", ") if args.use_channels_names is not None else ch_names
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
            drop_rate=args.drop,
            attn_drop_rate=args.attn_drop_rate,
            drop_path_rate=args.drop_path,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.chans_id = model.prepare_chan_ids(use_channels_names)

        if from_pretrain:
            print(f"Load ckpt from {fintune_list[args.model_name]}")
            checkpoint_path = fintune_list[args.model_name]
            pretrain_ckpt = torch.load(checkpoint_path)
            target_encoder_stat = {}
            for k, v in pretrain_ckpt['state_dict'].items():
                if k.startswith("target_encoder."):
                    target_encoder_stat[k[15:]] = v
            model.load_state_dict(target_encoder_stat)
        self.task_head = None
        self.main_model = model
        self.chan_conv = Conv1dWithConstraint(len(ch_names), chans_num, 1, max_norm=1)

    def forward(self, x):
        x = self.chan_conv(x)
        y = self.main_model(x, self.chans_id.to(x))
        return self.task_head(y)

class Custom_BIOT(nn.Module):
    def __init__(self, args, ch_names, from_pretrain=False, pretrain_model_choice=2):
        super().__init__()
        if pretrain_model_choice == 0:
            in_channels = 16
        elif pretrain_model_choice == 1:
            in_channels = 18
        elif pretrain_model_choice == 2:
            in_channels = 18
        else:
            raise ValueError("pretrain_model_choice should be 0, 1, or 2")
        model = BIOTClassifier(n_classes=args.nb_classes, n_channels=in_channels, n_fft=200, hop_length=100)
        if from_pretrain:
            model.biot.load_state_dict(torch.load(fintune_list[args.model_name][args.pretrain_model_choice]))
            print(f"load pretrain model from {fintune_list[args.model_name][args.pretrain_model_choice]}")
        model.classifier = nn.Identity()
        self.task_head = None
        self.main_model = model
        self.chan_conv = Conv1dWithConstraint(len(ch_names), in_channels, 1, max_norm=1)

    def forward(self, x):
        x = self.chan_conv(x)
        y = self.main_model(x)
        return self.task_head(y)

class Custom_EEGNet(nn.Module):
    def __init__(self, args, ch_names, num_t):
        super().__init__()
        model = EEGNet(chans=len(ch_names), classes=args.nb_classes, time_points=num_t * 200)
        model.fc = nn.Identity()
        self.task_head = None
        self.main_model = model
    def forward(self, x):
        y = self.main_model(x)
        return self.task_head(y)

class Custom_LMDA(nn.Module):
    def __init__(self, args, ch_names, num_t):
        super().__init__()
        model = LMDA(num_classes=args.nb_classes, chans=len(ch_names), samples=num_t * 200, channel_depth1=24, channel_depth2=7)
        model.classifier = nn.Identity()
        self.task_head = None
        self.main_model = model
    def forward(self, x):
        y = self.main_model(x)
        return self.task_head(y)

class Custom_EEGConformer(nn.Module):
    def __init__(self, args, ch_names, num_t):
        super().__init__()
        model = Conformer(C=len(ch_names), time_points=num_t * 200, n_classes=args.nb_classes)
        model.classification_head = nn.Identity()
        self.task_head = None
        self.main_model = model
    def forward(self, x):
        y = self.main_model(x)
        return self.task_head(y)

class Custom_STTransformer(nn.Module):
    def __init__(self, args, ch_names, num_t):
        super().__init__()
        model = STTransformer(n_classes=args.nb_classes, channel_legnth=num_t * 200, n_channels=len(ch_names))
        model.classification = nn.Identity()
        self.task_head = None
        self.main_model = model
    def forward(self, x):
        y = self.main_model(x)
        return self.task_head(y)

# --------------------------------------------------------------------------------------------------

# -----------------------------Load the models based on args.model_name------------------------------
def get_models(args, ch_names, num_t):

    # set args for pretrained weights
    if args.finetune_mod in ['full', 'linear']:
        if args.model_name in ['EEGNet', 'LMDA', 'EEGConformer', 'ST-Transformer']:
            print("No pretrained weights, start training from scratch")
            from_pretrain = False
        else:
            from_pretrain = True
    else:
        from_pretrain = False
    
    # init models
    if args.model_name == 'LaBraM':
        model = Custom_LaBraM(args, ch_names, num_t, from_pretrain)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint((len(ch_names) * num_t + 1) * 200, args.nb_classes, max_norm=1, flatten=1)
        elif args.task_mod == 'Regression':
            model.task_head = Linear3Layers(input_dim=200, hidden_dim=200, output_dim=1, patch_mean=True, remove_cls=True)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint((len(ch_names) * num_t + 1) * 200, 1024, max_norm=1, flatten=1)
    elif args.model_name == 'CBraMod':
        model = Custom_CBraMod(args, from_pretrain)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint(len(ch_names) * num_t * 200, args.nb_classes, max_norm=1, flatten=1)
        elif args.task_mod == 'Regression':
            model.task_head = Linear3Layers(input_dim=(len(ch_names) * num_t) * 200, hidden_dim=200, output_dim=1, flatten=1)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(len(ch_names) * num_t * 200, 1024, max_norm=1, flatten=1)
    elif args.model_name == 'EEGPT':
        model = Custom_EEGPT(args, ch_names, num_t, from_pretrain)
        if args.task_mod == 'Classification':
            model.task_head = nn.Sequential(
                LinearWithConstraint(2048, 16, max_norm=1, flatten=2, dropout=0.5),
                LinearWithConstraint(4 * num_t * 16, args.nb_classes, max_norm=0.25, flatten=1)
            )
        elif args.task_mod == 'Regression':
            model.task_head = Linear3Layers(input_dim=512, hidden_dim=256, output_dim=1, patch_mean=True)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(4 * 2048, 1024, max_norm=1, flatten=1)
    elif args.model_name == 'BIOT':
        model = Custom_BIOT(args, ch_names, from_pretrain=from_pretrain, pretrain_model_choice=2)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint(256, args.nb_classes, max_norm=1)
        elif args.task_mod == 'Regression':
            model.task_head = Linear3Layers(input_dim=256, hidden_dim=256, output_dim=1)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(256, 1024, max_norm=1)
    elif args.model_name == 'EEGNet':
        model = Custom_EEGNet(args, ch_names, num_t)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint(model.linear_size, args.nb_classes, max_norm=1)
        elif args.task_mod == 'Regression':
            model.task_head = Linear3Layers(input_dim=model.linear_size, hidden_dim=200, output_dim=1)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(model.linear_size, 1024, max_norm=1)
    elif args.model_name == 'LMDA':
        model = Custom_LMDA(args, ch_names, num_t)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint(model.linear_size, args.nb_classes, max_norm=1)
        elif args.task_mod == 'Regression':
            model.task_head = Linear3Layers(input_dim=model.linear_size, hidden_dim=200, output_dim=1)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(model.linear_size, 1024, max_norm=1)
    elif args.model_name == 'EEGConformer':
        model = Custom_EEGConformer(args, ch_names, num_t)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint(model.time_points * 40, args.nb_classes, max_norm=1)
        elif args.task_mod == 'Regression':
            model.task_head = Linear3Layers(input_dim=model.time_points * 40, hidden_dim=40, output_dim=1)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(model.time_points * 40, 1024, max_norm=1)
    elif args.model_name == 'ST-Transformer':
        model = Custom_STTransformer(args, ch_names, num_t)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint(256, args.nb_classes, max_norm=1)
        elif args.task_mod == 'Regression':
            model.task_head = Linear3Layers(input_dim=256, hidden_dim=256, output_dim=1)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(256, 1024, max_norm=1)
    else:
        print("Unknown model name!")
        exit(0)
    
    # check if task_head is correct
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

    print(args.output_dir)
    
    # utils.init_distributed_mode(args)
    args.distributed = False
    if ds_init is not None:
        utils.create_ds_config(args)

    if args.save_ckpt_freq == 0:
        args.save_ckpt_freq = args.epochs
    
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    # dataset_train, dataset_test, dataset_val: follows the standard format of torch.utils.data.Dataset.
    # ch_names: list of strings, channel names of the dataset. It should be in capital letters.

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

    if True:  # args.distributed:
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

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

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
    model_ema = None
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
    if args.layer_decay < 1.0:
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

    # get optimizer, lr_schedule...(code from labram)
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

    print("Use step level LR scheduler!")
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
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
    
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
            if log_writer is not None:
                log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
            
            # see engine_for_finetuning.py
            train_stats = train_one_epoch(
                args, model, data_loader_train, optimizer,
                device, epoch, loss_scaler, model_ema,
                log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
                num_training_steps_per_epoch=num_training_steps_per_epoch, ch_names=ch_names
            )
            
            # save checkpoint
            if args.output_dir and args.save_ckpt:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, save_ckpt_freq=args.save_ckpt_freq)
                
            # val and test
            if data_loader_val is not None:
                val_stats = evaluate(args, data_loader_val, model, device, header='Val:', ch_names=ch_names, metrics=metrics)
                test_stats = evaluate(args, data_loader_test, model, device, header='Test:', ch_names=ch_names, metrics=metrics)

                if args.task_mod == 'Classification':
                    print(f"Accuracy of the network on the {len(dataset_val)} val EEG: {val_stats['accuracy']:.2f}%")
                    print(f"Accuracy of the network on the {len(dataset_test)} test EEG: {test_stats['accuracy']:.2f}%")
                else:
                    print(f"R2_Score of the network on the {len(dataset_val)} val EEG: {val_stats['R2_Score']:.2f}")
                    print(f"R2_Score of the network on the {len(dataset_test)} test EEG: {test_stats['R2_Score']:.2f}")
                
                # save best checkpoint
                if args.task_mod == 'Classification':
                    if max_accuracy < val_stats["accuracy"]:
                        max_accuracy = val_stats["accuracy"]
                        if args.output_dir and args.save_ckpt:
                            utils.save_model(
                                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                        max_accuracy_test = test_stats["accuracy"]
                    print(f'Max accuracy val: {max_accuracy:.2f}%, max accuracy test: {max_accuracy_test:.2f}%')
                else:
                    if max_r2 < val_stats["R2_Score"]:
                        max_r2 = val_stats["R2_Score"]
                        if args.output_dir and args.save_ckpt:
                            utils.save_model(
                                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                        max_r2_test = test_stats["R2_Score"]
                    print(f'Max R2_Score val: {max_r2:.2f}, max R2_Score test: {max_r2_test:.2f}')

                # log metrics
                if log_writer is not None:
                    for key, value in val_stats.items():
                        if key == 'accuracy':
                            log_writer.update(accuracy=value, head="val", step=epoch)
                        elif key == 'balanced_accuracy':
                            log_writer.update(balanced_accuracy=value, head="val", step=epoch)
                        elif key == 'f1_weighted':
                            log_writer.update(f1_weighted=value, head="val", step=epoch)
                        elif key == 'pr_auc':
                            log_writer.update(pr_auc=value, head="val", step=epoch)
                        elif key == 'roc_auc':
                            log_writer.update(roc_auc=value, head="val", step=epoch)
                        elif key == 'cohen_kappa':
                            log_writer.update(cohen_kappa=value, head="val", step=epoch)
                        elif key == 'Pearson_Correlation':
                            log_writer.update(Pearson_Correlation=value, head="val", step=epoch)
                        elif key == 'R2_Score':
                            log_writer.update(R2_Score=value, head="val", step=epoch)
                        elif key == 'RMSE':
                            log_writer.update(RMSE=value, head="val", step=epoch)
                        elif key == 'loss':
                            log_writer.update(loss=value, head="val", step=epoch)
                    for key, value in test_stats.items():
                        if key == 'accuracy':
                            log_writer.update(accuracy=value, head="test", step=epoch)
                        elif key == 'balanced_accuracy':
                            log_writer.update(balanced_accuracy=value, head="test", step=epoch)
                        elif key == 'f1_weighted':
                            log_writer.update(f1_weighted=value, head="test", step=epoch)
                        elif key == 'pr_auc':
                            log_writer.update(pr_auc=value, head="test", step=epoch)
                        elif key == 'roc_auc':
                            log_writer.update(roc_auc=value, head="test", step=epoch)
                        elif key == 'cohen_kappa':
                            log_writer.update(cohen_kappa=value, head="test", step=epoch)
                        elif key == 'Pearson_Correlation':
                            log_writer.update(Pearson_Correlation=value, head="test", step=epoch)
                        elif key == 'R2_Score':
                            log_writer.update(R2_Score=value, head="test", step=epoch)
                        elif key == 'RMSE':
                            log_writer.update(RMSE=value, head="test", step=epoch)
                        elif key == 'loss':
                            log_writer.update(loss=value, head="test", step=epoch)
                        
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
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        opts.output_dir = f"finetuning_results/{opts.task_mod}/{opts.model_name}_results/{opts.output_dir}/finetune_{opts.finetune_mod}/{opts.dataset}_{opts.finetune_mod}_epoch{opts.epochs}_bs{opts.batch_size}_lr={opts.lr}_{opts.norm_method}_{opts.seed}"
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
