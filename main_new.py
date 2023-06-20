import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import json
from torch.utils.data import DataLoader
import time

from config import device, EEG_LEN, TEXT_LEN, d_model, d_inner, \
    num_layers, num_heads, d_k, d_v, class_num, dropout
from optim_new import ScheduledOptim
from trainer import train
from model_new import Transformer
from utils import open_file
from dataset_new import prepare_sr_eeg_data, EEGDataset, clean_dic, shuffle_split_data


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--model', type=str, help="Please choose a model from the following list: ['transformer', 'biLSTM', 'MLP', 'resnet', 'fusion', 'CCA_fusion', 'CCA_ds', 'WD_fusion', 'WD_ds']")
    parser.add_argument('--modality', type = str, default = None, help="Please choose a modality from the following list: ['eeg', 'text', fusion]")
    parser.add_argument('--dataset', type=str, help="Please choose a dataset from the following list: ['KEmoCon', 'ZuCo']")
    parser.add_argument('--task', default ='SA', type=str, help="If dataset == Zuco, please choose a task from the following list: ['SA', 'RD']")
    parser.add_argument('--level', type=str, default = 'sentence', help="If ZuCo, please choose the level of EEG feature you want to work with from this list: ['word', 'concatword', 'sentence']")
    parser.add_argument('--batch_size', type=int, default = 64)
    parser.add_argument('--text_feature_len', type = int, default = 768)
    parser.add_argument('--eeg_feature_len', type = int, default = 832)
    parser.add_argument('--lr', type = float, default = 1e-5)
    parser.add_argument('--eps', type = float, defaeult = 1e-4)
    parser.add_argument('--weight_decay', type = float, default = 1e-2)
    parser.add_argument('--warm_steps', type = int, default = 2000)
    parser.add_argument('--epochs', type = int, default = 200)
    
    
    
    return parser.parse_args()


if __name__ == '__main__':
    
    args = get_args()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    if args.dataset == 'KEmoCon':
        ###### COMING SOON #####
        pass
    
    elif args.dataset == 'ZuCo':
        
        if args.task == 'RD':
            ###### COMING SOON #####
            pass
        
        elif args.task == 'SA':
            
            assert (args.level == 'sentence' or args.level == 'word' or args.level == 'concatword'), 'Please choose a correct eeg feature type'
            
            if args.level == 'word':
                ###### COMING SOON #####
                pass
            elif args.level == 'concatword':
                ###### COMING SOON #####
                pass      
            
            else:
                # Load clean csv
                sentiment_labels = pd.read_csv('data/sentiment_labels_clean.csv')
                
                sr_eeg_data_path = 'data/SR'
                
                sentence_list = sentiment_labels.sentence.tolist()
                labels_list = sentiment_labels.sentiment_label.tolist()
                sentence_ids_list = sentiment_labels.sentence_id.tolist()
                
                eeg_dict = prepare_sr_eeg_data(sr_eeg_data_path, sentence_list, labels_list, sentence_ids_list)
                
                eeg_train_split, eeg_val_split, eeg_test_split = shuffle_split_data(eeg_dict)
                
                train, train_id_mapping = clean_dic(eeg_train_split)
                val, val_id_mapping = clean_dic(eeg_val_split)
                test, test_id_mapping = clean_dic(eeg_test_split)
                
                
                train_dataset = EEGDataset(train)
                val_dataset = EEGDataset(val)
                test_dataset = EEGDataset(test)
                                
                train_loader = DataLoader(
                    dataset=train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                )
                val_loader = DataLoader(
                    dataset=val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                )
                test_loader = DataLoader(
                    dataset=test_dataset,
                    batch_size=1,
                    shuffle=False,
                )
                
                if args.model == 'transformer':
                    model = Transformer(device = device, d_feature_text = TEXT_LEN, d_feature_eeg = EEG_LEN,\
                                            d_model = d_model, d_inner = d_inner, n_layers = num_layers, \
                                            n_head=num_heads, d_k = d_k, d_v = d_v, dropout= dropout, \
                                            class_num = class_num, args = args)
                    model = nn.DataParallel(model)
                    model = model.to(device)
                    
                optimizer = ScheduledOptim(
                    Adam(filter(lambda x: x.requires_grad, model.parameters()), 
                         betas = (0.9, 0.98), eps = args.eps, lr = args.lr, weight_decay = args.weight_decay),
                    d_model = d_model, warm_steps = args.warm_steps
                )
                
                all_train_loss, all_train_acc = [], []
                for epoch in range(args.epochs):
                    
                    print('[ Epoch', epoch, ']')
                    start = time.time()
                    
                    trian_loss, train_acc, cm, all_pred, all_labels = train(train_loader, device, model, optimizer, train_dataset.__len__(), args)
                    
                    
                    
                    
                    
                    
            
                
