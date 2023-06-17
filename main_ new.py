import time
import sklearn
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from tqdm import tqdm
import numpy as np
import pandas as pd
from model_new import *
from optim_new import ScheduledOptim
from dataset_new import *
from config import *
from embeddings import get_embeddings
from loss import cal_loss

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# from roc_new import plot_roc
from imblearn.over_sampling import SMOTE
import time
import os
from scipy.stats import wasserstein_distance
from imblearn.over_sampling import RandomOverSampler
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
import argparse


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--model', type=str, help="Please choose a model from the following list: ['transformer', 'biLSTM', 'MLP', 'resnet', 'fusion', 'CCA_fusion', 'CCA_ds', 'WD_fusion', 'WD_ds']")
    parser.add_argument('--modality', type = str, default = None, help="Please choose a modality from the following list: ['eeg', 'text', fusion]")
    parser.add_argument('--dataset', type=str, help="Please choose a dataset from the following list: ['KEmoCon', 'ZuCo']")
    parser.add_argument('--task', default ='SA', type=str, help="If dataset == Zuco, please choose a task from the following list: ['SA', 'RD']")
    parser.add_argument('--level', type=str, default = 'sentence', help="If ZuCo, please choose the level of EEG feature you want to work with from this list: ['word', 'concatword', 'sentence']")
    return parser.parse_args()