import numpy as np
import pandas as pd
import scipy.io as io
import gzip
import math
import os
import re
import scipy
import h5py
import json

def open_file(path_to_file):
    file_extension = os.path.splitext(path_to_file)[1]
    file_reader = {
        '.json': lambda: json.load(open(path_to_file)),
        '.txt': lambda: open(path_to_file, 'r').read().split('\n'),
        '.npy': lambda: np.load(path_to_file, allow_pickle=True).item(),
        '.h5': lambda: h5py.File(path_to_file, 'r'),
        '.csv': lambda: pd.read_csv(path_to_file)
    }
    return file_reader.get(file_extension, lambda: None)()

def get_matfiles(task:str, subdir = '\\results_zuco\\'):
    """
        Args: Task number ("task1", "task2", "task3") plus sub-directory
        Return: 12 matlab files (one per subject) for given task
    """
    path = os.getcwd() + subdir + task
    files = [os.path.join(path,file) for file in os.listdir(path)[1:]]
    assert len(files) == 12, 'each task must contain 12 .mat files'
    return files

class DataTransformer:
    """
        Transforms ET (and EEG data) to use for further analysis (per test subject)
    """
    
    def __init__(self, task:str, level:str, scaling='min-max', fillna='zeros'):
        """
            Args: task ("task1", "task2", or "task3"), data level, scaling technique, how to treat NaNs
        """
        tasks = ['task1', 'task2', 'task3']
        if task in tasks:
            self.task = task
        else:
            raise Exception('Task can only be one of "task1", "task2", or "task3"') 
        levels = ['sentence', 'word']
        if level in levels:
            self.level = level
        else:
            raise Exception('Data can only be processed on sentence or word level')
        #display raw (absolut) values or normalize data according to specified feature scaling technique
        feature_scalings = ['min-max', 'mean-norm', 'standard', 'raw']
        if scaling in feature_scalings:
            self.scaling = scaling
        else:
            raise Exception('Features must either be min-max scaled, mean-normalized or standardized')
        fillnans = ['zeros', 'mean', 'min']
        if fillna in fillnans:
            self.fillna = fillna
        else:
            raise Exception('Missing values should be replaced with zeros, the mean or min per feature')
    
    def __call__(self, subject:int):
        """
            Args: test subject (0-11)
            Return: DataFrame with normalized features (i.e., attributes) on sentence or word level
        """
        # subject should not be a property of data transform object (thus, it's not in the init method), 
        # since we want to apply the same data transformation to each subject
        subjects = list(range(12))
        if subject not in subjects:
            raise Exception('Access subject data with an integer value between 0 - 11')  
        files = get_matfiles(self.task)
        data = io.loadmat(files[subject], squeeze_me=True, struct_as_record=False)['sentenceData']
        
        if self.level == 'sentence':
            fields = ['SentLen',  'omissionRate', 'nFixations', 'meanPupilSize', 'GD', 'TRT', 
                      'FFD', 'SFD', 'GPT']
            if self.task == 'task1' and subject == 2:
                features = np.zeros((len(data)-101, len(fields)))
            elif self.task == 'task2' and (subject == 6 or subject == 11):
                features = np.zeros((len(data)-50, len(fields)))
            elif self.task == 'task3' and subject == 3:
                features = np.zeros((len(data)-47, len(fields)))
            elif self.task == 'task3' and subject == 7:
                features = np.zeros((len(data)-48, len(fields)))
            elif self.task == 'task3' and subject == 11:
                features = np.zeros((len(data)-89, len(fields)))
            else:
                features = np.zeros((len(data), len(fields)))

        elif self.level == 'word':
            if self.task == 'task1' and subject == 2:
                n_words = sum([len(sent.word) for i, sent in enumerate(data[:-1]) if i < 150 or i > 249])
            elif self.task == 'task2' and subject == 6:
                n_words = sum([len(sent.word) for i, sent in enumerate(data) if i > 49])  
            elif self.task == 'task2' and subject == 11:
                n_words = sum([len(sent.word) for i, sent in enumerate(data) if i < 50 or i > 99])
            elif self.task == 'task3' and subject == 3:
                n_words = sum([len(sent.word) for i, sent in enumerate(data) if i < 178 or i > 224])
            elif self.task == 'task3' and subject == 7:
                n_words = sum([len(sent.word) for i, sent in enumerate(data) if i < 359])
            elif self.task == 'task3' and subject == 11:
                n_words = sum([len(sent.word) for i, sent in enumerate(data) if i < 270 or (i > 313 and i < 362)])
            else:
                n_words = sum([len(sent.word) for sent in data])
            fields = ['Sent_ID', 'Word_ID', 'Word', 'nFixations', 'meanPupilSize', 
                      'GD', 'TRT', 'FFD', 'SFD', 'GPT', 'WordLen']
            df = pd.DataFrame(index=range(n_words), columns=[fields])
            k = 0
        
        idx = 0
        for i, sent in enumerate(data):
            if (self.task == 'task1' and subject == 2) and ((i >= 150 and i <= 249) or i == 399):
                continue
            elif (self.task == 'task2' and subject == 6) and (i <= 49):
                continue
            elif (self.task == 'task2' and subject == 11) and (i >= 50 and i <= 99):
                continue
            elif (self.task == 'task3' and subject == 3) and (i >= 178 and i <= 224):
                continue
            elif (self.task == 'task3' and subject == 7) and (i >= 359):
                continue
            elif (self.task == 'task3' and subject == 11) and ((i >= 270 and i <= 313) or (i >= 362 and i <= 406)):
                continue
            else:
                nwords_fixated = 0
                for j, word in enumerate(sent.word):
                    token = re.sub('[^\w\s]', '', word.content)
                    #lowercase words at the beginning of the sentence only
                    token = token.lower() if j == 0 else token 
                    if self.level == 'sentence':
                        word_features = [getattr(word, field) if hasattr(word, field)\
                                         and not isinstance(getattr(word, field), np.ndarray) else\
                                         0 for field in fields[2:]]
                        features[idx, 2:] += word_features
                        nwords_fixated += 0 if len(set(word_features)) == 1 and next(iter(set(word_features))) == 0 else 1
                    elif self.level == 'word':
                        df.iloc[k, 0] = str(idx)+'_NR' if self.task=='task1' or self.task=='task2'\
                                        else str(idx)+'_TSR'
                        df.iloc[k, 1] = j
                        df.iloc[k, 2] = token
                        df.iloc[k, 3:-1] = [getattr(word, field) if hasattr(word, field)\
                                            and not isinstance(getattr(word, field), np.ndarray) else\
                                            0 for field in fields[3:-1]]
                        df.iloc[k, -1] = len(token)
                        k += 1

                if self.level == 'sentence':
                    features[idx, 0] = len(sent.word)
                    features[idx, 1] = sent.omissionRate
                    #normalize by number of words for which fixations were reported
                    features[idx, 2:] /= nwords_fixated
                    
                idx += 1

        #handle -inf, inf and NaN values
        if self.level == 'sentence': 
            features = self.check_inf(features)
            
        elif self.level == 'word':
            if self.fillna == 'zeros':
                df.iloc[:,:].fillna(0, inplace=True)
            elif self.fillna == 'min':
                for i, field in enumerate(fields):
                    df.iloc[:,i].fillna(getattr(df, field).values.min(), inplace=True)
            elif self.fillna == 'mean':
                for i, field in enumerate(fields):
                    df.iloc[:,i].fillna(getattr(df, field).values.mean(), inplace=True)
                    
            df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, inplace=True)

        #normalize data according to feature scaling technique
        if self.scaling == 'min-max':
            if self.level == 'sentence':
                features = np.array([(feat - min(feat))/(max(feat) - min(feat)) for feat in features.T])
            elif self.level == 'word':
                df.iloc[:, 3:] = [(getattr(df,field).values - getattr(df,field).values.min())/\
                                  (getattr(df,field).values.max() - getattr(df,field).values.min())\
                                  for field in fields[3:]]
                
        elif self.scaling == 'mean-norm':
            if self.level == 'sentence':
                features = np.array([(feat - np.mean(feat))/(max(feat) - min(feat)) for feat in features.T])
            elif self.level == 'word':
                df.iloc[:, 3:] = [(getattr(df,field).values - getattr(df,field).values.mean())/\
                                  (getattr(df,field).values.max() - getattr(df,field).values.min())\
                                  for field in fields[3:]]
                
        elif self.scaling == 'standard':
            if self.level == 'sentence':
                features = np.array([(feat - np.mean(feat))/np.std(feat) for feat in features.T])
            elif self.level == 'word':
                df.iloc[:, 3:] = [(getattr(df,field).values - getattr(df,field).values.mean())/\
                                  getattr(df,field).values.std() for field in fields[3:]]
                
        if self.level == 'sentence':
            if self.scaling == 'raw':
                df = pd.DataFrame(data=features, index=range(features.shape[0]), columns=[fields])
            else:
                df = pd.DataFrame(data=features.T, index=range(features.shape[1]), columns=[fields])
                
            if self.fillna == 'zeros':
                df.iloc[:,:].fillna(0, inplace=True)
            elif self.fillna == 'min':
                for i, field in enumerate(fields):
                    df.iloc[:,i].fillna(getattr(df, field).values.min(), inplace=True)
            elif self.fillna == 'mean':
                for i, field in enumerate(fields):
                    df.iloc[:,i].fillna(getattr(df, field).values.mean(), inplace=True)
           
        return df
    
    @staticmethod
    def check_inf(features):
        pop_idx = 0
        for idx, feat in enumerate(features):
            if True in np.isneginf(feat) or True in np.isinf(feat):
                features = np.delete(features, idx-pop_idx, axis=0)
                pop_idx += 1
        return features
    
    
def split_data(sbjs): 
    """
        Args: Data per sbj on sentence level for task 1
        Purpose: Function is necessary to control for order effects (only relevant for Task 1 (NR))
    """
    first_half, second_half = [], []
    for sbj in sbjs:
        first_half.append(sbj[:len(sbj)//2])
        second_half.append(sbj[len(sbj)//2:])
    return first_half, second_half