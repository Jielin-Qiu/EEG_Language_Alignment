import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from config import *
import scipy.stats as stats
import os
import scipy.io as io
import math
from tqdm import tqdm
from scipy.stats import zscore
import random
from transformers import BertModel, BertTokenizer

class resnet_Text_EEGDataset(Dataset):
  def __init__(self, texts, signals, labels, tokenizer, max_len):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.signals = signals

  @property
  def n_insts(self):
    return len(self.labels)

  @property
  def text_len(self):
    return 32
  
  def sig_len(self):
    return self.signals.shape[1]

  def __len__(self):
    return self.n_insts

  def __getitem__(self, item):
    text = str(self.texts[item])
    label = self.labels[item]
    signal = self.signals[item]

    input_ids = [self.tokenizer.encode(text, add_special_tokens=False,max_length=MAX_LEN, padding = 'max_length', truncation = True, return_token_type_ids = False, return_attention_mask = True)]   
    input_ids = np.array(input_ids)
    input_ids = stats.zscore(input_ids, axis=None, nan_policy='omit')
    input_ids = np.array(input_ids)

    signal = np.array([signal])
    signal = torch.FloatTensor(signal)

    return signal, torch.FloatTensor(input_ids), torch.tensor(label, dtype=torch.long)

class Text_EEGDataset(Dataset):
  def __init__(self, texts, signals, labels, max_len):
    self.texts = texts
    self.labels = labels
    self.signals = torch.FloatTensor(signals)

  @property
  def n_insts(self):
    return len(self.labels)

  @property
  def text_len(self):
    return 768
  
  def sig_len(self):
    return self.signals.shape[1]

  def __len__(self):
    return self.n_insts

  def __getitem__(self, item):
    text = self.texts[item]
    label = self.labels[item]
    signal = self.signals[item]
    text = torch.FloatTensor(text)    
    return signal, text, torch.tensor(label, dtype=torch.long)
  
  
def prepare_sr_eeg_data(sr_eeg_data_path, sentence_list, labels_list, sentence_ids_list, args):
  
  eeg_dict = {}
  
  count = 0
  
  for i in tqdm(os.listdir(sr_eeg_data_path), desc = 'Creating SR EEG dataset: '):
      
      file_path = os.path.join(sr_eeg_data_path,i)
      
      if file_path == 'data/SR/resultsZDN_SR.mat':
        pass
      
      else:
        
        io_mat_file = io.loadmat(file_path, squeeze_me=True, struct_as_record=False)['sentenceData']

        for j in range(len(io_mat_file)):
          
        
          t1 = io_mat_file[j].mean_t1[:104]
          t2 = io_mat_file[j].mean_t2[:104]
          
          a1 = io_mat_file[j].mean_a1[:104]
          a2 = io_mat_file[j].mean_a2[:104]

          g1 = io_mat_file[j].mean_g1[:104]
          g2 = io_mat_file[j].mean_g2[:104]
          
          b1 = io_mat_file[j].mean_b1[:104]
          b2 = io_mat_file[j].mean_b2[:104]
          
          nan_mask_t1 = np.isnan(t1)
          nan_mask_t2 = np.isnan(t2)
          
          nan_mask_a1 = np.isnan(a1)
          nan_mask_a2 = np.isnan(a2)
          
          nan_mask_b1 = np.isnan(b1)
          nan_mask_b2 = np.isnan(b2)
          
          nan_mask_g1 = np.isnan(g1)
          nan_mask_g2 = np.isnan(g2)
          
          if np.all(nan_mask_t1) or np.all(nan_mask_t2) or np.all(nan_mask_a1) or np.all(nan_mask_a2) \
            or np.all(nan_mask_b1) or np.all(nan_mask_b2) or np.all(nan_mask_g1) or np.all(nan_mask_g2):

            pass
          
          else:
            t1[nan_mask_t1] = 0
            t2[nan_mask_t2] = 0
            
            a1[nan_mask_a1] = 0
            a2[nan_mask_a2] = 0
            
            b1[nan_mask_b1] = 0
            b2[nan_mask_b2] = 0
            
            g1[nan_mask_g1] = 0
            g2[nan_mask_g2] = 0
            
          
            sentence = io_mat_file[j].content
            
            if sentence == 'Ultimately feels emp11111ty and unsatisfying, like swallowing a Communion wafer without the wine.':
                sentence = 'Ultimately feels empty and unsatisfying, like swallowing a Communion wafer without the wine.'
            elif sentence == "Bullock's complete lack of focus and ability quickly derails the film.1":
                sentence =  "Bullock's complete lack of focus and ability quickly derails the film."
            
            sentence_idx = sentence_list.index(sentence)
            
            label = labels_list[sentence_idx]
            if label == -1:
              label = 2.0
            
            sentence_id = sentence_ids_list[sentence_idx]
            
            if sentence_id in eeg_dict:
              
              eeg_dict[sentence_id]['t1'].append(t1.tolist())
              eeg_dict[sentence_id]['t2'].append(t2.tolist())
              
              eeg_dict[sentence_id]['a1'].append(a1.tolist())
              eeg_dict[sentence_id]['a2'].append(a2.tolist())
              
              eeg_dict[sentence_id]['b1'].append(b1.tolist())
              eeg_dict[sentence_id]['b2'].append(b2.tolist())
              
              eeg_dict[sentence_id]['g1'].append(g1.tolist())
              eeg_dict[sentence_id]['g2'].append(g2.tolist())
              
              assert label == eeg_dict[sentence_id]['label']
              
              assert sentence == eeg_dict[sentence_id]['sentence']
              
            else:
              
                eeg_dict[sentence_id] = {
                  'label' : label,
                  'sentence' : sentence,
                  't1' : [t1.tolist()],
                  't2' : [t2.tolist()],
                  'a1' : [a1.tolist()],
                  'a2' : [a2.tolist()],
                  'b1' : [b1.tolist()],
                  'b2' : [b2.tolist()],
                  'g1' : [g1.tolist()],
                  'g2' : [g2.tolist()]
                }
                
            assert (len(t1) == len(t2) == len(a1) == len(a2) == len(b1) \
                == len(b2) == len(g1) == len(g2))
      if args.dev == True:
        count +=1
        if count == 1:
          break  
        
  for k in eeg_dict.keys():
    
    t1 = np.array(eeg_dict[k]['t1'], dtype=np.float32)
    t2 = np.array(eeg_dict[k]['t2'], dtype=np.float32)
    
    a1 = np.array(eeg_dict[k]['a1'], dtype=np.float32)
    a2 = np.array(eeg_dict[k]['a2'], dtype=np.float32)
    
    b1 = np.array(eeg_dict[k]['b1'], dtype=np.float32)
    b2 = np.array(eeg_dict[k]['b2'], dtype=np.float32)
    
    g1 = np.array(eeg_dict[k]['g1'], dtype=np.float32)
    g2 = np.array(eeg_dict[k]['g2'], dtype=np.float32)
    
    mean_t1 = np.mean(t1, axis=0)
    mean_t2 = np.mean(t2, axis=0)
    
    mean_a1 = np.mean(a1, axis=0)
    mean_a2 = np.mean(a2, axis=0)
    
    mean_b1 = np.mean(b1, axis=0)
    mean_b2 = np.mean(b2, axis=0)
    
    mean_g1 = np.mean(g1, axis=0)
    mean_g2 = np.mean(g2, axis=0)
    
    eeg_dict[k]['t1'] = zscore(mean_t1)
    eeg_dict[k]['t2'] = zscore(mean_t2)
    
    eeg_dict[k]['a1'] = zscore(mean_a1)
    eeg_dict[k]['a2'] = zscore(mean_a2)
    
    eeg_dict[k]['b1'] = zscore(mean_b1)
    eeg_dict[k]['b2'] = zscore(mean_b2)
    
    eeg_dict[k]['g1'] = zscore(mean_g1)
    eeg_dict[k]['g2'] = zscore(mean_g2)
    
    
  return eeg_dict


class EEGDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)
      
    def __getembed__(self, text):
      tokenized = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
      input_ids = tokenized['input_ids']
      with torch.no_grad():
          outputs = self.bert(input_ids)
          embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
      
      embeddings = zscore(embeddings)
      
      assert embeddings.shape == (768,)
      
      return torch.tensor(embeddings, dtype = torch.float32)
      
    def __getitem__(self, idx):
        item = self.data[idx]
        embeddings = self.__getembed__(item['sentence'])
        
        sample = {
            'label': torch.tensor(item['label'], dtype=torch.long),
            'sentence': embeddings,
            # 'a1': item['a1'],
            # 'a2': item['a2'],
            # 't1': item['t1'],
            # 't2': item['t2'],
            # 'b1': item['b1'],
            # 'b2': item['b2'],
            # 'g1': item['g1'],
            # 'g2': item['g2'],
            'seq': torch.tensor(np.concatenate([item['t1'], item['t2'], item['a1'], item['a2'], \
              item['b1'], item['b2'], item['g1'], item['g2']]), dtype=torch.float32)
        }
        assert sample['seq'].shape == (832,)

        return sample


def clean_dic(eeg_dict):
  new_dict = {}
  id_mapping = {}
  count = 0

  for key, value in tqdm(eeg_dict.items(), desc = 'Cleaning Dictionary: '):
      new_key = count
      id_mapping[key] = new_key
      new_dict[new_key] = value
      count += 1

  # Print the new dictionary and ID mapping
  
  return new_dict, id_mapping

def shuffle_split_data(eeg_dict):

  # Shuffle the keys of the original dictionary
  keys = list(eeg_dict.keys())
  random.shuffle(keys)

  # Calculate the proportions
  total_instances = len(eeg_dict)
  train_proportion = int(0.7 * total_instances)
  val_proportion = int(0.15 * total_instances)

  # Split the data dictionary
  train_data = {}
  val_data = {}
  test_data = {}

  # Iterate over the shuffled keys and distribute the instances
  for i, key in tqdm(enumerate(keys), desc = 'Spltting Dictionary'):
      value = eeg_dict[key]
      if i < train_proportion:
          train_data[key] = value
      elif train_proportion <= i < train_proportion + val_proportion:
          val_data[key] = value
      else:
          test_data[key] = value
          
  return train_data, val_data, test_data
  
  
