import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from config import *
import scipy.stats as stats


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