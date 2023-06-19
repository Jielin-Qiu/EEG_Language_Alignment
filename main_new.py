import os
import argparse
import pandas as pd
import numpy as np

from utils import open_file
from dataset_new import prepare_sr_eeg_data


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--model', type=str, help="Please choose a model from the following list: ['transformer', 'biLSTM', 'MLP', 'resnet', 'fusion', 'CCA_fusion', 'CCA_ds', 'WD_fusion', 'WD_ds']")
    parser.add_argument('--modality', type = str, default = None, help="Please choose a modality from the following list: ['eeg', 'text', fusion]")
    parser.add_argument('--dataset', type=str, help="Please choose a dataset from the following list: ['KEmoCon', 'ZuCo']")
    parser.add_argument('--task', default ='SA', type=str, help="If dataset == Zuco, please choose a task from the following list: ['SA', 'RD']")
    parser.add_argument('--level', type=str, default = 'sentence', help="If ZuCo, please choose the level of EEG feature you want to work with from this list: ['word', 'concatword', 'sentence']")
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
                
                
            