from __future__ import print_function

import argparse
import pdb
import os
import math
import sys

import numpy as np
import pandas as pd

### Internal Imports 
from myexp.loaddata import Generic_MIL_Dataset
from myexp.train import train
from myexp.utils import get_custom_exp_code

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler


def main(args):
    #### Create Results Directory
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    folds = np.arange(0, args.k)

    ### Start 5-Fold CV Evaluation.
    for i in folds:
        # start = timer()
        seed_torch(args.seed)

        ### Gets the Train + Val Dataset Loader.
        #TODO: 改路径！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        train_dataset, val_dataset = dataset.return_splits(from_id=False, 
                csv_path='splits/5foldcv/tcga_gbmlgg/splits_{}.csv'.format(i))    # Use this
        # train_dataset, val_dataset = dataset.return_splits(from_id=False, 
        #         csv_path='splits/5foldcv/tcga_gbmlgg/splits_test.csv')
        
        print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
        datasets = (train_dataset, val_dataset)
        
        ### Specify the input dimension size if using genomic features.
        if 'coattn' in args.mode:
            args.omic_input_dim = train_dataset.omic_sizes
            print('Genomic Dimensions', args.omic_input_dim)
        else:
            args.omic_input_dim = train_dataset.omic_input_size 

        train(datasets, i, args)

        # break

### Training settings
parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
### Checkpoint + Misc. Pathing Parameters
parser.add_argument('--data_root_dir',   type=str, default='path/to/data_root_dir')
parser.add_argument('--seed',              type=int, default=1)
parser.add_argument('--k',                  type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--results_dir',     type=str, default='./results')
parser.add_argument('--which_splits',    type=str, default='5foldcv')
parser.add_argument('--split_dir',       type=str, default='tcga_gbmlgg')
parser.add_argument('--results_root_dir', type=str, default='./results1') 

### Model Parameters.
parser.add_argument('--model_type',      type=str, choices=['abc', 'amil'], default='abc')
parser.add_argument('--mode',            type=str, choices=['path', 'pathomic', 'coattn'], default='coattn')
parser.add_argument('--fusion',          type=str, choices=['None', 'concat', 'bilinear'], default='concat')
parser.add_argument('--apply_sig',         action='store_true', default=False, help='Use genomic features as signature embeddings.')
parser.add_argument('--apply_sigfeats',  action='store_true', default=False, help='Use genomic features as tabular features.')
parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi',  type=str, default='small')
parser.add_argument('--model_size_omic', type=str, default='small')
### Optimizer Parameters + Survival Loss Function
parser.add_argument('--batch_size',      type=int, default=1)
parser.add_argument('--max_epochs',      type=int, default=30)
parser.add_argument('--lr',                 type=float, default=2e-4)
parser.add_argument('--reg',              type=float, default=1e-5, help='weight decay')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Creates Experiment Code from argparse + Folder Name to Save Results
args = get_custom_exp_code(args)
args.task = '_'.join(args.split_dir.split('_')[:2]) + '_classifier'
print("Experiment Name:", args.exp_code)

### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size_wsi': args.model_size_wsi,
            'model_size_omic': args.model_size_omic,
            "use_drop_out": args.drop_out}
print('\nLoad Dataset')

args.n_classes = 6

dataset = Generic_MIL_Dataset(mode = args.mode,
                                        apply_sig = args.apply_sig,
                                        data_dir= './WSIvectors',  # !!!
                                        shuffle = False, 
                                        seed = args.seed, 
                                        print_info = True,
                                        patient_strat= False,
                                        label_col = 'oncotree_code',
                                        ignore=[])


### Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)
args.results_root_dir = args.results_dir

### Appends to the results_dir path: 1) which splits were used for training (e.g. - 5foldcv), and then 2) the parameter code and 3) experiment code
args.results_dir = os.path.join(args.results_dir, args.which_splits,
                                args.param_code, str(args.exp_code) + 'slide{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

### Sets the absolute path of split_dir
args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)
print("split_dir", args.split_dir)
assert os.path.isdir(args.split_dir)
settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")
