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

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from roc_new import plot_roc
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

def cal_loss(label, device, args, pred = None, pred2 = None, out = None ):

    if (args.model == 'transformer') or (args.model == 'biLSTM') or (args.model == 'MLP') or (args.model == 'resnet'):
        loss = F.cross_entropy(pred, label, reduction='sum')
        pred = pred.max(1)[1]
        n_correct = pred.eq(label).sum().item()

        return loss, n_correct

    if args.model == 'fusion':
        loss = F.cross_entropy(out, label, reduction='sum')
        out = out.max(1)[1]
        n_correct = out.eq(label).sum().item()

        return loss, n_correct

    if args.model == 'CCA_fusion':
        loss1 = model.loss
        loss1 = loss1(pred, pred2)

        loss2 = F.cross_entropy(out, label, reduction = 'sum')
        loss = loss2 + loss1
        out = out.max(1)[1]
    
        n_correct = out.eq(label).sum().item()
        return loss, n_correct
    
    if args.model == 'WD_fusion':
        loss1 = wasserstein_distance(pred.cpu().detach().numpy().flatten(), 
        pred2.cpu().detach().numpy().flatten())

        loss1 = torch.tensor(loss1, requires_grad=True)

        loss2 = F.cross_entropy(out, label, reduction = 'sum')
        loss = loss2 + loss1
        out = out.max(1)[1]
    
        n_correct = out.eq(label).sum().item()
        return loss, n_correct
    
    if (args.model == 'CCA_ds') and (args.modality == 'text'):
        loss1 = F.cross_entropy(pred, label, reduction='sum')

        loss2 = model1.loss
        loss2 = loss2(pred, pred2)
        loss = loss2+loss1
        pred = pred.max(1)[1]

        n_correct = pred.eq(label).sum().item()

        return loss, n_correct

    if (args.model == 'CCA_ds') and (args.modality == 'eeg'):
        loss1 = F.cross_entropy(pred2, label, reduction='sum')

        loss2 = model1.loss
        loss2 = loss2(pred, pred2)
        loss = loss2+loss1
        pred = pred2.max(1)[1]

        n_correct = pred2.eq(label).sum().item()

        return loss, n_correct

    if (args.model == 'WD_ds') and (args.modality == 'text'):
        loss1 = F.cross_entropy(pred, label, reduction='sum')

        loss2 = wasserstein_distance(pred.cpu().detach().numpy().flatten(), 
        pred2.cpu().detach().numpy().flatten())
        loss2 = torch.tensor(loss2, requires_grad=True)

        loss = loss2+loss1
        pred = pred.max(1)[1]

        n_correct = pred.eq(label).sum().item()

        return loss, n_correct

    if (args.model == 'WD_ds') and (args.modality == 'eeg'):
        loss1 = F.cross_entropy(pred2, label, reduction='sum')

        loss2 = wasserstein_distance(pred.cpu().detach().numpy().flatten(), 
        pred2.cpu().detach().numpy().flatten())
        loss2 = torch.tensor(loss2, requires_grad=True)

        loss = loss2+loss1
        pred = pred2.max(1)[1]

        n_correct = pred2.eq(label).sum().item()

        return loss, n_correct


def cal_statistic(cm):
    total_pred = cm.sum(0)
    total_true = cm.sum(1)

    acc_SP = sum([cm[i, i] for i in range(1, class_num)]) / total_pred[1: class_num].sum()
    pre_i = [cm[i, i] / total_pred[i] for i in range(class_num)]
    print(pre_i)
    rec_i = [cm[i, i] / total_true[i] for i in range(class_num)]
    print(rec_i)
    F1_i = [2 * pre_i[i] * rec_i[i] / (pre_i[i] + rec_i[i]) for i in range(class_num)]

    pre_i = np.array(pre_i)
    rec_i = np.array(rec_i)
    F1_i = np.array(F1_i)
    pre_i[np.isnan(pre_i)] = 0
    rec_i[np.isnan(rec_i)] = 0
    F1_i[np.isnan(F1_i)] = 0

    return acc_SP, list(pre_i), list(rec_i), list(F1_i)


def train_raw(train_loader, device, model, optimizer, total_num, args):
    all_labels = []
    all_res = []
    all_pred =[]
    model.train()
    total_loss = 0
    total_correct = 0    
    
    for batch in tqdm(train_loader, mininterval=100, desc='- (Training)  ', leave=False): 

        sig2, sig, label, = map(lambda x: x.to(device), batch)
        optimizer.zero_grad()
        if args.modality == 'text':
            pred = model(sig)
            all_labels.extend(label.cpu().numpy())
            all_res.extend(pred.max(1)[1].cpu().numpy())
            loss, n_correct = cal_loss(label, device, args, pred = pred)
            all_pred.extend(pred.cpu().detach().numpy())
            loss.backward()
            optimizer.step_and_update_lr()

            total_loss += loss.item()
            total_correct += n_correct
            cm = confusion_matrix(all_labels, all_res)

        if args.modality == 'eeg':
            pred = model(sig2)
            all_labels.extend(label.cpu().numpy())
            all_res.extend(pred.max(1)[1].cpu().numpy())
            loss, n_correct = cal_loss(label, device, args, pred = pred)
            all_pred.extend(pred.cpu().detach().numpy())
            loss.backward()
            optimizer.step_and_update_lr()

            total_loss += loss.item()
            total_correct += n_correct
            cm = confusion_matrix(all_labels, all_res)

    train_loss = total_loss / total_num
    train_acc = total_correct / total_num
    return train_loss, train_acc, cm, all_pred, all_labels


def train_fusion(train_loader, device, model, optimizer, total_num, args):
    model.train()
    all_labels = []
    all_res = []
    all_pred = []
    total_loss = 0
    total_correct = 0

    for batch in tqdm(train_loader, mininterval=100, desc='- (Training)  ', leave=False): 
        
        sig2, sig1, label1, = map(lambda x: x.to(device), batch)
      
      
        optimizer.zero_grad()
      
        if args.model == 'fusion':

            out, _, _ = model(sig1, sig2)
            all_labels.extend(label1.cpu().numpy())
            all_res.extend(out.max(1)[1].cpu().numpy())
            all_pred.extend(out.cpu().detach().numpy())
            loss, n_correct1 = cal_loss(label1, device, args, out = out)
            
            
            loss.backward()
            optimizer.step_and_update_lr()
            total_loss += loss.item()
            total_correct += (n_correct1)
            
            cm = confusion_matrix(all_labels, all_res)

        if (args.model == 'CCA_fusion') or (args.model == 'WD_fusion'):

            out, pred, pred2 = model(sig1, sig2)
            all_labels.extend(label1.cpu().numpy())
            all_res.extend(out.max(1)[1].cpu().numpy())
            all_pred.extend(out.cpu().detach().numpy())
            loss, n_correct1 = cal_loss(label1, device, args, pred = pred, pred2 = pred2, out = out)
            
            
            loss.backward()
            optimizer.step_and_update_lr()
            total_loss += loss.item()
            total_correct += (n_correct1)
            
            cm = confusion_matrix(all_labels, all_res)
    

    train_loss = total_loss / total_num
    train_acc = total_correct / total_num

    return train_loss, train_acc, cm, all_pred, all_labels


def train_alignment_ds(train_loader, device, model, optimizer, total_num, args):
    all_labels = []
    all_res = []
    all_pred_train = []
    model.train()
    total_loss = 0
    total_correct = 0    
    
    for batch in tqdm(train_loader, mininterval=100, desc='- (Training)  ', leave=False): 

        sig2, sig1, label, = map(lambda x: x.to(device), batch)
        optimizer.zero_grad()
        pred, pred2 = model(sig1, sig2)  
        all_labels.extend(label.cpu().numpy())

        if args.modality == 'text':
            all_res.extend(pred.max(1)[1].cpu().numpy())
            all_pred_train.extend(pred.detach().cpu().numpy())
            loss, n_correct = cal_loss(label, device, args, pred = pred, pred2 = pred2)
            loss.backward()
            optimizer.step_and_update_lr()

            total_loss += loss.item()
            total_correct += n_correct
            cm = confusion_matrix(all_labels, all_res)

        if args.modality == 'eeg':
            all_res.extend(pred2.max(1)[1].cpu().numpy())
            all_pred_train.extend(pred2.detach().cpu().numpy())
            loss, n_correct = cal_loss(label, device, args, pred = pred, pred2 = pred2)
            loss.backward()
            optimizer.step_and_update_lr()

            total_loss += loss.item()
            total_correct += n_correct
            cm = confusion_matrix(all_labels, all_res)

    train_loss = total_loss / total_num
    train_acc = total_correct / total_num
    return train_loss, train_acc, cm, all_pred_train, all_labels


def eval_raw(valid_loader, device, model, total_num, args):
    all_labels = []
    all_res = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, mininterval=100, desc='- (Validation)  ', leave=False):
            sig2, sig, label, = map(lambda x: x.to(device), batch)

            if args.modality == 'text':
                pred = model(sig)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred.max(1)[1].cpu().numpy())
                all_pred.extend(pred.cpu().detach().numpy())
                loss, n_correct = cal_loss(label, device, args, pred = pred)

                total_loss += loss.item()
                total_correct += n_correct

            if args.modality == 'eeg':
                pred = model(sig2)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred.max(1)[1].cpu().numpy())
                all_pred.extend(pred.cpu().detach().numpy())
                loss, n_correct = cal_loss(label, device, args, pred = pred)

                total_loss += loss.item()
                total_correct += n_correct

    cm = confusion_matrix(all_labels, all_res)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    valid_loss = total_loss / total_num
    valid_acc = total_correct / total_num
    return valid_loss, valid_acc, cm, sum(rec_i[1:]) * 0.6 + sum(pre_i[1:]) * 0.4, all_pred, all_labels


def eval_fusion(valid_loader1, device, model, total_num, args):
    model.eval()

    all_labels = []
    all_res = []
    all_pred = []
    total_loss = 0
    total_correct = 0

    with torch.no_grad():
      
     
      for batch in tqdm(valid_loader1, mininterval=100, desc='- (Training)  ', leave=False): 

        sig2, sig1, label1, = map(lambda x: x.to(device), batch)

        if args.model == 'fusion':

            out, _, _ = model(sig1, sig2)
            all_labels.extend(label1.cpu().numpy())
            all_res.extend(out.max(1)[1].cpu().numpy())
            all_pred.extend(out.cpu().detach().numpy())
            loss, n_correct1 = cal_loss(label1, device, args, out = out )
            total_loss += loss.item()
            total_correct += (n_correct1)

        if (args.model == 'CCA fusion') or (args.model == 'WD fusion'):

            out, pred, pred2 = model(sig1, sig2)
            all_labels.extend(label1.cpu().numpy())
            all_res.extend(out.max(1)[1].cpu().numpy())
            all_pred.extend(out.cpu().detach().numpy())
            loss, n_correct1 = cal_loss(label1, device, args, pred=pred, pred2 = pred2, out = out )
            total_loss += loss.item()
            total_correct += (n_correct1)


    cm = confusion_matrix(all_labels, all_res)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    valid_loss = total_loss / total_num 
    valid_acc = total_correct /total_num 
    return valid_loss, valid_acc, cm, sum(rec_i[1:]) * 0.6 + sum(pre_i[1:]) * 0.4, all_pred, all_labels

def eval_alignment_ds(valid_loader, device, model, total_num, args):
    all_labels = []
    all_res = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, mininterval=100, desc='- (Validation)  ', leave=False):
            sig2, sig1, label, = map(lambda x: x.to(device), batch)

            if args.modality == 'text':
                pred, pred2 = model(sig1, sig2)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred.max(1)[1].cpu().numpy())
                all_pred.extend(pred.detach().cpu().numpy())
                loss, n_correct = cal_loss(label, device, args, pred=pred, pred2=pred2)

                total_loss += loss.item()
                total_correct += n_correct

            if args.modality == 'eeg':
                pred, pred2 = model(sig1, sig2)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred2.max(1)[1].cpu().numpy())
                all_pred.extend(pred2.detach().cpu().numpy())
                loss, n_correct = cal_loss(label, device, args, pred=pred, pred2=pred2)

                total_loss += loss.item()
                total_correct += n_correct

    cm = confusion_matrix(all_labels, all_res)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    valid_loss = total_loss / total_num
    valid_acc = total_correct / total_num
    return valid_loss, valid_acc, cm, sum(rec_i[1:]) * 0.6 + sum(pre_i[1:]) * 0.4, all_pred, all_labels


def test_raw(test_loader, device, model, total_num):
    all_labels = []
    all_res = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, mininterval=0.5, desc='- (Validation)  ', leave=False):

            sig2, sig, label, = map(lambda x: x.to(device), batch)

            if args.modality == 'text': 
                pred = model(sig)  
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred.max(1)[1].cpu().numpy())
                all_pred.extend(pred.cpu().numpy())
                loss, n_correct = cal_loss(label, device, args, pred = pred)

                total_loss += loss.item()
                total_correct += n_correct

            if args.modality == 'eeg': 
                pred = model(sig2)  
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred.max(1)[1].cpu().numpy())
                all_pred.extend(pred.cpu().numpy())
                loss, n_correct = cal_loss(label, device, args, pred = pred)

                total_loss += loss.item()
                total_correct += n_correct


    np.savetxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred.txt',all_pred)
    np.savetxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label.txt', all_labels)
    all_pred = np.array(all_pred)
    cm = confusion_matrix(all_labels, all_res)
    print("test_cm:", cm)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    test_acc = total_correct / total_num
    print('test_acc is : {test_acc}'.format(test_acc=test_acc))

def test_fusion(test_loader, device, model, total_num, args):
    all_labels = []
    all_res = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
     
      for batch in tqdm(test_loader, mininterval=100, desc='- (Training)  ', leave=False): 

        sig2, sig1, label1, = map(lambda x: x.to(device), batch)

        if args.model == 'fusion':
            out, _, _ = model(sig1, sig2)  
            all_labels.extend(label1.cpu().numpy())
            all_res.extend(out.max(1)[1].cpu().numpy())
            all_pred.extend(out.cpu().numpy())
            loss, n_correct1 = cal_loss(label1,device, args, out = out)
            total_loss += loss.item()
            total_correct += (n_correct1)
    
        if (args.model == 'CCA fusion') or (args.model == 'WD fusion'):
            out, pred, pred2 = model(sig1, sig2)  
            all_labels.extend(label1.cpu().numpy())
            all_res.extend(out.max(1)[1].cpu().numpy())
            all_pred.extend(out.cpu().numpy())
            loss, n_correct1 = cal_loss(label1,device, args, pred=pred, pred2=pred2, out = out)
            total_loss += loss.item()
            total_correct += (n_correct1)

    np.savetxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred.txt',all_pred)
    np.savetxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label.txt', all_labels)
    all_pred = np.array(all_pred)
    cm = confusion_matrix(all_labels, all_res)
    print("test_cm:", cm)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    test_acc = total_correct / total_num 
    print('test_acc is : {test_acc}'.format(test_acc=test_acc))


def test_alignment_ds(test_loader, device, model, total_num, args):
    all_labels = []
    all_res = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, mininterval=0.5, desc='- (Validation)  ', leave=False):

            sig2, sig1, label, = map(lambda x: x.to(device), batch)

            if args.modality == 'text':

                pred, pred2 = model(sig1, sig2)  
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred.max(1)[1].cpu().numpy())
                all_pred.extend(pred.cpu().numpy())
                loss, n_correct = cal_loss(label, device, args, pred=pred, pred2=pred2)

                total_loss += loss.item()
                total_correct += n_correct
            
            if args.modality == 'eeg':
                pred, pred2 = model(sig1, sig2)  
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred2.max(1)[1].cpu().numpy())
                all_pred.extend(pred2.cpu().numpy())
                loss, n_correct = cal_loss(label, device, args, pred=pred, pred2=pred2)

                total_loss += loss.item()
                total_correct += n_correct


    np.savetxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred.txt',all_pred)
    np.savetxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label.txt', all_labels)
    all_pred = np.array(all_pred)
    cm = confusion_matrix(all_labels, all_res)
    print("test_cm:", cm)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    test_acc = total_correct / total_num
    print('test_acc is : {test_acc}'.format(test_acc=test_acc))


if __name__ == "__main__":
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.dataset == 'KEmoCon':
        
        # --- Preprocess
        df = pd.read_csv('preprocessed_kemo/KEmoCon/df.csv')

        X = df.drop([emotion], axis = 1)
        y= df[[emotion]]

        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 2, test_size = 0.2, shuffle = True, stratify = y)
        ros = RandomOverSampler(random_state=2)
        X_resampled_text, y_resampled_text = ros.fit_resample(X_train, y_train)
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, random_state= 2, test_size = 0.5, shuffle = True, stratify = y_val)
        df_test = pd.concat([X_test, y_test], axis = 1)
        df_train = pd.concat([X_resampled_text, y_resampled_text], axis = 1)
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        df_val = pd.concat([X_val, y_val], axis = 1)

        df_train_text = df_train[[emotion, 'new_words']]
        df_train_eeg = df_train[eeg]

        df_val_text = df_val[[emotion, 'new_words']]
        df_val_eeg = df_val[eeg]

        df_test_text = df_test[[emotion, 'new_words']]
        df_test_eeg = df_test[eeg]

        # --- Save CSV
        df_train_text.to_csv('df_train_text.csv', header = None, index = False, index_label = False)
        df_train_eeg.to_csv('df_train_eeg.csv', header = None, index = False, index_label = False)

        df_val_text.to_csv('df_val_text.csv', header = None, index = False, index_label = False)
        df_val_eeg.to_csv('df_val_eeg.csv', header = None, index = False, index_label=False)


        df_test_text.to_csv('df_test_text.csv', header = None, index = False, index_label = False)
        df_test_eeg.to_csv('df_test_eeg.csv', header = None, index = False, index_label=False)

        # --- Load CSV
        df_train_text = pd.read_csv('df_train_text.csv', header = None).values
        df_train_eeg = pd.read_csv('df_train_eeg.csv', header = None).values

        df_val_text= pd.read_csv('df_val_text.csv', header = None).values
        df_val_eeg = pd.read_csv('df_val_eeg.csv', header = None).values

        df_test_text= pd.read_csv('df_test_text.csv', header = None).values
        df_test_eeg = pd.read_csv('df_test_eeg.csv', header = None).values


        time_start_i = time.time()

        embeddings_train = get_embeddings(df_train_text[:,1], device)
        embeddings_val = get_embeddings(df_val_text[:,1], device)
        embeddings_test = get_embeddings(df_test_text[:,1], device)

                # --- Text and EEG
        train_text_eeg = Text_EEGDataset(
            texts = embeddings_train,
            labels = df_train_text[:,0],
            signals = df_train_eeg[:, 1:]
        )
        val_text_eeg = Text_EEGDataset(
            texts = embeddings_val,
            labels = df_val_text[:, 0],
            signals = df_val_eeg[:, 1:]
        )

        test_text_eeg = Text_EEGDataset(
        texts = embeddings_test,
        labels = df_test_text[:, 0],
        signals = df_test_eeg[:, 1:]

        )
        
        # --- Sampler
        target = df_train_text[:, 0].astype('int')
        class_sample_count = np.unique(target, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = weight[target]
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        # --- Loader
        train_loader_text_eeg = DataLoader(dataset=train_text_eeg,
                                batch_size=batch_size,
                                num_workers=2,
                                sampler = sampler)

        valid_loader_text_eeg = DataLoader(dataset=val_text_eeg,
                                batch_size=batch_size,
                                num_workers=2,
                                shuffle=True)
        test_loader_text_eeg = DataLoader(dataset=test_text_eeg,
                                batch_size=batch_size,
                                num_workers=2,
                                shuffle=True)

        if args.model == 'transformer':

            if args.modality == 'text':

                model = Transformer(device=device, d_feature=SIG_LEN, d_model=d_model, d_inner=d_inner,
                        n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)

        
            if args.modality == 'eeg':

                model = Transformer2(device=device, d_feature=SIG_LEN2, d_model=d_model, d_inner=d_inner,
                        n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
            
        if (args.model == 'fusion') or (args.model == 'WD_fusion') or (args.model == 'CCA_fusion'):

            model1 = Transformer(device=device, d_feature=SIG_LEN, d_model=d_model, d_inner=d_inner,
                    n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
            model2 = Transformer2(device=device, d_feature=SIG_LEN2, d_model=d_model, d_inner=d_inner,
                        n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
            model1 = nn.DataParallel(model1)
            model2 = nn.DataParallel(model2)
            model2 = model2.to(device)
            model1 = model1.to(device)
            model = Fusion(device=device, model1 = model1, model2 = model2, outdim_size=outdim_size, 
            use_all_singular_values=use_all_singular_values, class_num=class_num).to(device) 

        if (args.model == 'WD_ds') or (args.model == 'CCA_ds'):

            model1 = Transformer(device=device, d_feature=SIG_LEN, d_model=d_model, d_inner=d_inner,
                    n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
            model2 = Transformer2(device=device, d_feature=SIG_LEN2, d_model=d_model, d_inner=d_inner,
                        n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
            model1 = nn.DataParallel(model1)
            model2 = nn.DataParallel(model2)
            model2 = model2.to(device)
            model1 = model1.to(device)
            model = CAM(model1, model2)

        if args.model == 'biLSTM':
            
            if args.modality == 'text':
                model = BiLSTM(vocab_size = SIG_LEN, device = device)
            if args.modality == 'eeg':
                model = BiLSTM(vocab_size = SIG_LEN2, device = device)

        if args.model == 'MLP':
            
            if args.modality == 'text':
                model = MLP(vocab_size = SIG_LEN, device = device)
            if args.modality == 'eeg':
                model = MLP(vocab_size = SIG_LEN2, device = device)

        if args.model == 'resnet':
            
            if args.modality == 'text':
                model = ResNet1D(in_channels=1, base_filters=SIG_LEN, kernel_size=1, stride=2,
                groups = 2, n_block = 3, n_classes=class_num)
            if args.modality == 'eeg':
                model = ResNet1D(in_channels=1, base_filters=SIG_LEN2, kernel_size=1, stride=2,
                groups = 2, n_block = 3, n_classes=class_num)
            
        model = nn.DataParallel(model)
        model = model.to(device)



    if (args.dataset == 'ZuCo') and (args.task == 'SA'):
        if args.level == 'sentence':
            df = pd.read_csv(f'preprocessed_eeg/ZuCo/SA/{patient}_sentence.csv')
        if args.level == 'word':
            df = pd.read_csv(f'preprocessed_eeg/ZuCo/SA/{patient}_word.csv')
        if args.level == 'concatword':
            df = pd.read_csv(f'preprocessed_eeg/ZuCo/SA/{patient}_concatword.csv')

        X = df.drop([emotion], axis = 1)
        y= df[[emotion]]

        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 2, test_size = 0.2, shuffle = True)
        ros = RandomOverSampler(random_state=2)
        X_resampled_text, y_resampled_text = ros.fit_resample(X_train, y_train)

        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, random_state= 2, test_size = 0.5, shuffle = True)
        df_test = pd.concat([X_test, y_test], axis = 1)
        df_train = pd.concat([X_resampled_text, y_resampled_text], axis = 1)
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        df_val = pd.concat([X_val, y_val], axis = 1)

        df_train_text = df_train[[emotion, 'new_words']]
        df_train_eeg_label = df_train[[emotion]]
        df_train_eeg = df_train.iloc[:, 2:]
        df_train_eeg = pd.concat([df_train_eeg_label, df_train_eeg], axis=1)

        df_val_text = df_val[[emotion, 'new_words']]
        df_val_eeg_label = df_val[[emotion]]
        df_val_eeg = df_val.iloc[:, 2:]

        df_val_eeg = pd.concat([df_val_eeg_label, df_val_eeg], axis=1)

        df_test_text = df_test[[emotion, 'new_words']]
        df_test_eeg_label = df_test[[emotion]]
        df_test_eeg = df_test.iloc[:, 2:]
        df_test_eeg = pd.concat([df_test_eeg_label, df_test_eeg], axis=1)

        # --- Save CSV
        df_train_text.to_csv('df_train_text.csv', header = None, index = False, index_label = False)
        df_train_eeg.to_csv('df_train_eeg.csv', header = None, index = False, index_label = False)

        df_val_text.to_csv('df_val_text.csv', header = None, index = False, index_label = False)
        df_val_eeg.to_csv('df_val_eeg.csv', header = None, index = False, index_label=False)


        df_test_text.to_csv('df_test_text.csv', header = None, index = False, index_label = False)
        df_test_eeg.to_csv('df_test_eeg.csv', header = None, index = False, index_label=False)

        # --- Load CSV
        df_train_text = pd.read_csv('df_train_text.csv', header = None).values
        df_train_eeg = pd.read_csv('df_train_eeg.csv', header = None).values

        df_val_text= pd.read_csv('df_val_text.csv', header = None).values
        df_val_eeg = pd.read_csv('df_val_eeg.csv', header = None).values

        df_test_text= pd.read_csv('df_test_text.csv', header = None).values
        df_test_eeg = pd.read_csv('df_test_eeg.csv', header = None).values

        time_start_i = time.time()

        embeddings_train = get_embeddings(df_train_text[:,1], device)
        embeddings_val = get_embeddings(df_val_text[:,1], device)
        embeddings_test = get_embeddings(df_test_text[:,1], device)

        # --- Text and EEG
        train_text_eeg = Text_EEGDataset(
            texts = embeddings_train,
            labels = df_train_text[:,0],
            signals = df_train_eeg[:, 1:]
        )
        val_text_eeg = Text_EEGDataset(
            texts = embeddings_val,
            labels = df_val_text[:, 0],
            signals = df_val_eeg[:, 1:]
        )

        test_text_eeg = Text_EEGDataset(
        texts = embeddings_test,
        labels = df_test_text[:, 0],
        signals = df_test_eeg[:, 1:]

        )
        
        # --- Sampler
        target = df_train_text[:, 0].astype('int')
        class_sample_count = np.unique(target, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = weight[target]
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        # --- Loader
        train_loader_text_eeg = DataLoader(dataset=train_text_eeg,
                                batch_size=batch_size,
                                num_workers=2,
                                sampler = sampler)

        valid_loader_text_eeg = DataLoader(dataset=val_text_eeg,
                                batch_size=batch_size,
                                num_workers=2,
                                shuffle=True)
        test_loader_text_eeg = DataLoader(dataset=test_text_eeg,
                                batch_size=batch_size,
                                num_workers=2,
                                shuffle=True)

        if args.model == 'transformer':

            if args.modality == 'text':

                model = Transformer(device=device, d_feature=SIG_LEN, d_model=d_model, d_inner=d_inner,
                        n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)

            if args.modality == 'eeg':

                model = Transformer2(device=device, d_feature=SIG_LEN2, d_model=d_model, d_inner=d_inner,
                        n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
            
        if (args.model == 'fusion') or (args.model == 'WD_fusion') or (args.model == 'CCA_fusion'):

            model1 = Transformer(device=device, d_feature=SIG_LEN, d_model=d_model, d_inner=d_inner,
                    n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
            model2 = Transformer2(device=device, d_feature=SIG_LEN2, d_model=d_model, d_inner=d_inner,
                        n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
            model1 = nn.DataParallel(model1)
            model2 = nn.DataParallel(model2)
            model2 = model2.to(device)
            model1 = model1.to(device)
            model = Fusion(device=device, model1 = model1, model2 = model2, outdim_size=outdim_size, 
            use_all_singular_values=use_all_singular_values, class_num=class_num).to(device) 

        if (args.model == 'WD_ds') or (args.model == 'CCA_ds'):

            model1 = Transformer(device=device, d_feature=SIG_LEN, d_model=d_model, d_inner=d_inner,
                    n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
            model2 = Transformer2(device=device, d_feature=SIG_LEN2, d_model=d_model, d_inner=d_inner,
                        n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
            model1 = nn.DataParallel(model1)
            model2 = nn.DataParallel(model2)
            model2 = model2.to(device)
            model1 = model1.to(device)
            model = CAM(model1, model2)

        if args.model == 'biLSTM':
            
            if args.modality == 'text':
                model = BiLSTM(vocab_size = SIG_LEN, device = device)
            if args.modality == 'eeg':
                model = BiLSTM(vocab_size = SIG_LEN2, device = device)

        if args.model == 'MLP':
            
            if args.modality == 'text':
                model = MLP(vocab_size = SIG_LEN, device = device)
            if args.modality == 'eeg':
                model = MLP(vocab_size = SIG_LEN2, device = device)

        if args.model == 'resnet':
            
            if args.modality == 'text':
                model = ResNet1D(in_channels=1, base_filters=SIG_LEN, kernel_size=1, stride=2,
                groups = 2, n_block = 3, n_classes=class_num)
            if args.modality == 'eeg':
                model = ResNet1D(in_channels=1, base_filters=SIG_LEN2, kernel_size=1, stride=2,
                groups = 2, n_block = 3, n_classes=class_num)
            
        model = nn.DataParallel(model)
        model = model.to(device)


    if (args.dataset == 'ZuCo') and (args.task == 'RD'):
        if args.level == 'sentence':
            df = pd.read_csv(f'preprocessed_eeg/ZuCo/RD/{patient}_sentence.csv')
        if args.level == 'word':
            df = pd.read_csv(f'preprocessed_eeg/ZuCo/RD/{patient}_word.csv')
        if args.level == 'concatword':
            df = pd.read_csv(f'preprocessed_eeg/ZuCo/RD/{patient}_concatword.csv')

        X = df.drop([emotion], axis = 1)
        y= df[[emotion]]

        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 2, test_size = 0.2, shuffle = True)
        ros = RandomOverSampler(random_state=2)
        X_resampled_text, y_resampled_text = ros.fit_resample(X_train, y_train)

        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, random_state= 2, test_size = 0.5, shuffle = True)
        df_test = pd.concat([X_test, y_test], axis = 1)
        df_train = pd.concat([X_resampled_text, y_resampled_text], axis = 1)
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        df_val = pd.concat([X_val, y_val], axis = 1)

        df_train_text = df_train[['label', 'new_words']]
        df_train_eeg_label = df_train[['label']]
        df_train_eeg = df_train.iloc[:, 2:]
        df_train_eeg = pd.concat([df_train_eeg_label, df_train_eeg], axis=1)

        df_val_text = df_val[['label', 'new_words']]
        df_val_eeg_label = df_val[['label']]
        df_val_eeg = df_val.iloc[:, 2:]

        df_val_eeg = pd.concat([df_val_eeg_label, df_val_eeg], axis=1)

        df_test_text = df_test[['label', 'new_words']]
        df_test_eeg_label = df_test[['label']]
        df_test_eeg = df_test.iloc[:, 2:]
        df_test_eeg = pd.concat([df_test_eeg_label, df_test_eeg], axis=1)

        # --- Save CSV
        df_train_text.to_csv('df_train_text.csv', header = None, index = False, index_label = False)
        df_train_eeg.to_csv('df_train_eeg.csv', header = None, index = False, index_label = False)

        df_val_text.to_csv('df_val_text.csv', header = None, index = False, index_label = False)
        df_val_eeg.to_csv('df_val_eeg.csv', header = None, index = False, index_label=False)


        df_test_text.to_csv('df_test_text.csv', header = None, index = False, index_label = False)
        df_test_eeg.to_csv('df_test_eeg.csv', header = None, index = False, index_label=False)

        # --- Load CSV
        df_train_text = pd.read_csv('df_train_text.csv', header = None).values
        df_train_eeg = pd.read_csv('df_train_eeg.csv', header = None).values

        df_val_text= pd.read_csv('df_val_text.csv', header = None).values
        df_val_eeg = pd.read_csv('df_val_eeg.csv', header = None).values

        df_test_text= pd.read_csv('df_test_text.csv', header = None).values
        df_test_eeg = pd.read_csv('df_test_eeg.csv', header = None).values

        time_start_i = time.time()

        embeddings_train = get_embeddings(df_train_text[:,1], device)
        embeddings_val = get_embeddings(df_val_text[:,1], device)
        embeddings_test = get_embeddings(df_test_text[:,1], device)

        # --- Text and EEG
        train_text_eeg = Text_EEGDataset(
            texts = embeddings_train,
            labels = df_train_text[:,0],
            signals = df_train_eeg[:, 1:]
        )
        val_text_eeg = Text_EEGDataset(
            texts = embeddings_val,
            labels = df_val_text[:, 0],
            signals = df_val_eeg[:, 1:]
        )

        test_text_eeg = Text_EEGDataset(
        texts = embeddings_test,
        labels = df_test_text[:, 0],
        signals = df_test_eeg[:, 1:]

        )
        
        # --- Sampler
        target = df_train_text[:, 0].astype('int')
        class_sample_count = np.unique(target, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = weight[target]
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        # --- Loader
        train_loader_text_eeg = DataLoader(dataset=train_text_eeg,
                                batch_size=batch_size,
                                num_workers=2,
                                sampler = sampler)

        valid_loader_text_eeg = DataLoader(dataset=val_text_eeg,
                                batch_size=batch_size,
                                num_workers=2,
                                shuffle=True)
        test_loader_text_eeg = DataLoader(dataset=test_text_eeg,
                                batch_size=batch_size,
                                num_workers=2,
                                shuffle=True)

        if args.model == 'transformer':

            if args.modality == 'text':

                model = Transformer(device=device, d_feature=SIG_LEN, d_model=d_model, d_inner=d_inner,
                        n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)

        
            if args.modality == 'eeg':

                model = Transformer2(device=device, d_feature=SIG_LEN2, d_model=d_model, d_inner=d_inner,
                        n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
            
        if (args.model == 'fusion') or (args.model == 'WD_fusion') or (args.model == 'CCA_fusion'):

            model1 = Transformer(device=device, d_feature=SIG_LEN, d_model=d_model, d_inner=d_inner,
                    n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
            model2 = Transformer2(device=device, d_feature=SIG_LEN2, d_model=d_model, d_inner=d_inner,
                        n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
            model1 = nn.DataParallel(model1)
            model2 = nn.DataParallel(model2)
            model2 = model2.to(device)
            model1 = model1.to(device)
            model = Fusion(device=device, model1 = model1, model2 = model2, outdim_size=outdim_size, 
            use_all_singular_values=use_all_singular_values, class_num=class_num).to(device) 

        if (args.model == 'WD_ds') or (args.model == 'CCA_ds'):

            model1 = Transformer(device=device, d_feature=SIG_LEN, d_model=d_model, d_inner=d_inner,
                    n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
            model2 = Transformer2(device=device, d_feature=SIG_LEN2, d_model=d_model, d_inner=d_inner,
                        n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
            model1 = nn.DataParallel(model1)
            model2 = nn.DataParallel(model2)
            model2 = model2.to(device)
            model1 = model1.to(device)
            model = CAM(model1, model2)

        if args.model == 'biLSTM':
            
            if args.modality == 'text':
                model = BiLSTM(vocab_size = SIG_LEN, device = device)
            if args.modality == 'eeg':
                model = BiLSTM(vocab_size = SIG_LEN2, device = device)

        if args.model == 'MLP':
            
            if args.modality == 'text':
                model = MLP(vocab_size = SIG_LEN, device = device)
            if args.modality == 'eeg':
                model = MLP(vocab_size = SIG_LEN2, device = device)

        if args.model == 'resnet':
            
            if args.modality == 'text':
                model = ResNet1D(in_channels=1, base_filters=SIG_LEN, kernel_size=1, stride=2,
                groups = 2, n_block = 3, n_classes=class_num)
            if args.modality == 'eeg':
                model = ResNet1D(in_channels=1, base_filters=SIG_LEN2, kernel_size=1, stride=2,
                groups = 2, n_block = 3, n_classes=class_num)
            
        model = nn.DataParallel(model)
        model = model.to(device)
    

    train_accs = []
    valid_accs = []
    eva_indis = []
    train_losses = []
    valid_losses = []
    all_pred_train1 = []
    all_label_train1=[]
    all_pred_val1 = []
    epochs = []
    all_label_val1=[]
    writer = SummaryWriter()

    optimizer = ScheduledOptim(
    Adam(filter(lambda x: x.requires_grad, model.parameters()),
        betas=(0.9, 0.98), eps=1e-4, lr = 1e-5, weight_decay=1e-2), d_model, warm_steps)

    for epoch_i in range(epoch):
        print('[ Epoch', epoch_i, ']')
        start = time.time()

        if (args.model == 'transformer') or (args.model == 'biLSTM') or (args.model == 'MLP') or (args.model == 'resnet'):
            train_loss, train_acc, train_cm, all_pred_train, all_label_train = train_raw(train_loader_text_eeg, device, model, optimizer, train_text_eeg.__len__(), args)

        if (args.model == 'fusion') or (args.model == 'CCA_fusion') or (args.model == 'WD_fusion'):
            train_loss, train_acc, train_cm, all_pred_train, all_label_train = train_fusion(train_loader_text_eeg, device, model, optimizer, train_text_eeg.__len__(), args)

        if (args.model == 'WD_ds') or (args.model == 'CCA_cs'):
            train_loss, train_acc, train_cm, all_pred_train, all_label_train = train_alignment_ds(train_loader_text_eeg, device, model, optimizer,train_text_eeg.__len__(), args)

        all_pred_train1.extend(all_pred_train)
        all_label_train1.extend(all_label_train)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        start = time.time()

        if (args.model == 'transformer') or (args.model == 'biLSTM') or (args.model == 'MLP') or (args.model == 'resnet'):
            valid_loss, valid_acc, valid_cm, eva_indi, all_pred_val, all_label_val = eval_raw(valid_loader_text_eeg, device, model, train_text_eeg.__len__(), args)

        if (args.model == 'fusion') or (args.model == 'CCA_fusion') or (args.model == 'WD_fusion'):
            valid_loss, valid_acc, valid_cm, eva_indi, all_pred_val, all_label_val  = eval_fusion(valid_loader_text_eeg, device, model, train_text_eeg.__len__(), args)

        if (args.model == 'WD_ds') or (args.model == 'CCA_cs'):
            valid_loss, valid_acc, valid_cm, eva_indi, all_pred_val, all_label_val = eval_alignment_ds(valid_loader_text_eeg, device, model, train_text_eeg.__len__(), args)

        all_pred_val1.extend(all_pred_val)
        all_label_val1.extend(all_label_val)
        valid_accs.append(valid_acc)
        eva_indis.append(eva_indi)
        valid_losses.append(valid_loss)

        model_state_dict = model.state_dict()

        checkpoint = {
            'model': model_state_dict,
            'config_file': 'config',
            'epoch': epoch_i}

        if eva_indi >= max(eva_indis):
            torch.save(checkpoint, f'baselines/{args.model}_{args.modality}/{args.level}.chkpt')
            print('    - [Info] The checkpoint file has been updated.')

    
        print('  - (Training)  loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '
                  'elapse: {elapse:3.3f} min'.format(loss=train_loss, accu=100 * train_acc,
                                                      elapse=(time.time() - start) / 60))
        print("train_cm:", train_cm)
        print('  - (Validation)  loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '
                  'elapse: {elapse:3.3f} min'.format(loss=valid_loss, accu=100 * valid_acc,
                                                      elapse=(time.time() - start) / 60))
        print("valid_cm:", valid_cm)
        writer.add_scalar('Accuracy', train_acc, epoch_i)
        writer.add_scalar('Loss', train_loss, epoch_i)
        epochs.append(epoch_i)

    dic = {}

    dic['train_acc'] = train_accs
    dic['train_loss'] = train_losses
    dic['valid_acc'] = valid_accs
    dic['valid_loss'] = valid_losses
    dic['epoch'] = epochs
    new_df = pd.DataFrame(dic)
    new_df.to_csv(f'baselines/{args.model}_{args.modality}/{args.level}_acc_loss.csv')
    

    np.savetxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred_val.txt',all_pred_val)
    np.savetxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label_val.txt',all_label_val)

    print('ALL DONE')               
    time_consume = (time.time() - time_start_i)
    print('total ' + str(time_consume) + 'seconds')
    fig1 = plt.figure('Figure 1')
    plt.plot(train_losses, label = 'train')
    plt.plot(valid_losses, label= 'valid')
    plt.xlabel('epoch')
    plt.ylim([0.0, 2])
    plt.ylabel('loss')
    plt.legend(loc ="upper right")
    plt.title('loss change curve')

    plt.savefig(f'baselines/{args.model}_{args.modality}/{args.level}_results_loss.png')

    fig2 = plt.figure('Figure 2')
    plt.plot(train_accs, label = 'train')
    plt.plot(valid_accs, label = 'valid')
    plt.xlabel('epoch')
    plt.ylim([0.0, 1])
    plt.ylabel('accuracy')
    plt.legend(loc ="upper right")
    plt.title('accuracy change curve')

    plt.savefig(f'baselines/{args.model}_{args.modality}/{args.level}_results_acc.png')
    

    test_model_name = f'baselines/{args.model}_{args.modality}/{args.level}.chkpt'
  
    chkpoint = torch.load(test_model_name, map_location='cuda')
    model.load_state_dict(chkpoint['model'])
    model = model.to(device)
    if (args.model == 'transformer') or (args.model == 'biLSTM') or (args.model == 'MLP') or (args.model == 'resnet'):
        test_raw(test_loader_text_eeg, device, model, test_text_eeg.__len__(), args)
    if (args.model == 'fusion') or (args.model == 'CCA_fusion') or (args.model == 'WD_fusion'):
        test_fusion(test_loader_text_eeg, device, model, test_text_eeg.__len__(), args)
    if (args.model == 'WD_ds') or (args.model == 'CCA_cs'):
        test_alignment_ds(test_loader_text_eeg, device, model, test_text_eeg.__len__(), args)

writer.close()



        

            

