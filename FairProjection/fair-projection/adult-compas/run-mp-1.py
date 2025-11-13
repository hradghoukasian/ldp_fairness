## standard packages
import sys
import os.path
import numpy as np
import pandas as pd
import random
import pickle
from tqdm import tqdm
from time import localtime, strftime
import time
import argparse


## scikit learn
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

## aif360
from aif360.datasets import StandardDataset

## custom packages
from utils import load_enem, MP_tol, MP_tol_ldp
from DataLoader import load_data

parser = argparse.ArgumentParser(description = "Configuration.")
parser.add_argument('--dataset', type=str, choices=['adult', 'compas'], default='adult')
args = parser.parse_args()

start_time = time.localtime()
start_time_str = strftime("%Y-%m-%d-%H.%M.%S", start_time)
filename = args.dataset + '-mp'
f = open(filename+'-log.txt','w')

if args.dataset == 'adult':
    df = load_data(name='adult')
    protected_attrs = ['race_1']
    label_name = 'income_1'
elif args.dataset == 'compas':
    df = load_data(name='compas')
    protected_attrs = ['race']
    label_name = 'is_recid'
else:
    f.write('Undefined Dataset')

repetition = 30
use_protected = True
use_sample_weight = True
tune_threshold = False
tolerance = [0.5]

f.write('Setup Summary\n')
f.write(' Sampled Dataset Shape: ' + str(df.shape) + '\n')
f.write(' repetition: '+str(repetition) + '\n')
f.write(' use_protected: '+str(use_protected) + '\n')
f.write(' use_sample_weight: '+str(use_sample_weight) + '\n')
f.write(' tune_threshold: '+str(tune_threshold) + '\n')
f.write(' tolerance: '+str(tolerance) + '\n')
f.flush()


# ### KL
# ## GBM
f.write('GMB - KL - meo\n')
gbm_kl_meo = MP_tol(df, protected_attrs=protected_attrs, label_name=label_name, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='gbm', div='kl', num_iter=repetition, rand_seed=42, constraint='meo')
# ##
f.write('GMB - KL - sp\n')
gbm_kl_sp = MP_tol(df, protected_attrs=protected_attrs, label_name=label_name, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='gbm', div='kl', num_iter=repetition, rand_seed=42, constraint='sp')

f.write('GMB - KL - sp - ldp\n')
gbm_kl_sp_ldp = MP_tol_ldp(df, protected_attrs=protected_attrs, label_name=label_name, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=1, log = f, model='gbm', div='kl', num_iter=repetition, rand_seed=42, constraint='sp', epsilon = 1)

f.write('GMB - KL - meo - ldp\n')
gbm_kl_sp_ldp = MP_tol_ldp(df, protected_attrs=protected_attrs, label_name=label_name, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=1, log = f, model='gbm', div='kl', num_iter=repetition, rand_seed=42, constraint='meo', epsilon = 1)


save = {
    'gbm_kl_meo': gbm_kl_meo,
    'gbm_kl_sp': gbm_kl_sp,
    'gbm_kl_sp_ldp': gbm_kl_sp_ldp,
    'tolerance': tolerance
}

savename = args.dataset + '-mp' +'.pkl'
with open(savename, 'wb+') as pickle_f:
    pickle.dump(save, pickle_f, 2)

f.write('Total Run Time: {:4.3f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(start_time))/60))
f.write('Finished!!!\n')
f.flush()
f.close()