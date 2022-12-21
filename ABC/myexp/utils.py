import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    omic = torch.cat([item[1] for item in batch], dim = 0).type(torch.FloatTensor)
    label = torch.LongTensor([item[2] for item in batch])
    return [img, omic, label]


def collate_MIL_sig(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    omic1 = torch.cat([item[1] for item in batch], dim = 0).type(torch.FloatTensor)
    omic2 = torch.cat([item[2] for item in batch], dim = 0).type(torch.FloatTensor)
    omic3 = torch.cat([item[3] for item in batch], dim = 0).type(torch.FloatTensor)
    omic4 = torch.cat([item[4] for item in batch], dim = 0).type(torch.FloatTensor)
    omic5 = torch.cat([item[5] for item in batch], dim = 0).type(torch.FloatTensor)
    omic6 = torch.cat([item[6] for item in batch], dim = 0).type(torch.FloatTensor)

    label = torch.LongTensor([item[7] for item in batch])
    return [img, omic1, omic2, omic3, omic4, omic5, omic6, label]


def get_split_loader(split_dataset1, split_dataset2, mode='coattn', batch_size=1):
    """
        return training loader: loader1 and validation loader: loader2 
    """
    if mode == 'coattn':
        collate = collate_MIL_sig
    else:
        collate = collate_MIL

    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    loader1 = DataLoader(split_dataset1, batch_size=batch_size,
                            sampler = SequentialSampler(split_dataset1), collate_fn = collate, **kwargs)

    loader2 = DataLoader(split_dataset2, batch_size=batch_size,
                        sampler = SequentialSampler(split_dataset2), collate_fn = collate, **kwargs)
    

    return loader1, loader2


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


def get_custom_exp_code(args):
    exp_code = '_'.join(args.split_dir.split('_')[:2])
    dataset_path = 'dataset_csv'
    param_code = ''

    ### Model Type
    if args.model_type == 'abc':
      param_code += 'ABC'
    elif args.model_type == 'amil':
      param_code += 'AMIL'
    else:
      raise NotImplementedError

    ### Learning Rate
    if args.lr != 2e-4:
      param_code += '_lr%s' % format(args.lr, '.0e')

    param_code += '_%s' % args.which_splits.split("_")[0]

    ### Batch Size
    if args.batch_size != 1:
      param_code += '_b%s' % str(args.batch_size)

    ### Applying Which Features
    if args.apply_sigfeats:
      param_code += '_sig'
      dataset_path += '_sig'

    ### Fusion Operation
    if args.fusion != "None":
      param_code += '_' + args.fusion

    args.exp_code = exp_code + "_" + param_code
    args.param_code = param_code
    args.dataset_path = dataset_path

    return args