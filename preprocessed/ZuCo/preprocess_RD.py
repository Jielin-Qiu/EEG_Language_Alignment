import scipy.io as io
import os
import pandas as pd
import numpy as np
import scipy.stats as stats


# --- sentence 

patient = ['ZAB', 'ZDM', 'ZDN', 'ZJM', 'ZJN', 'ZJS', 'ZKH', 'ZKW', 'ZMG']
file_name = f"task1- SR/Matlab_files/results{patient[0]}_TSR.mat"

data = io.loadmat(file_name, squeeze_me=True, struct_as_record=False)['sentenceData']


EEG = [ 'mean_t1',
'mean_t2',
'mean_a1',
'mean_a2',
'mean_b1', 
'mean_b2', 
'mean_g1', 
'mean_g2'] 



patient = patient[0]

df = pd.DataFrame()
labels = pd.DataFrame()

for i in range(len(data)):

    sub_df = pd.DataFrame()

    mean_t1 = data[i].mean_t1[0:104]
    mean_t2 = data[i].mean_t2[0:104]
    
    mean_a1 = data[i].mean_a1[0:104]
    mean_a2 = data[i].mean_a2[0:104]
    
    mean_b1 = data[i].mean_b1[0:104]
    mean_b2 = data[i].mean_b2[0:104]
    
    mean_g1 = data[i].mean_g1[0:104]
    mean_g2 = data[i].mean_g2[0:104]
    
    content = pd.Series(data[i].content)

    arr = np.concatenate((mean_t1, mean_t2, mean_a1, mean_a2, mean_b1, mean_b2, mean_g1, mean_g2))

    arr = pd.DataFrame(arr).T
    sub_df = pd.concat([sub_df, arr], axis = 1, ignore_index=True)
    
    labels = labels.append(content, ignore_index = True)


    df = df.append(sub_df, ignore_index=True)
        
read = pd.read_csv('labels.csv')

read = read[['relation']]

df = pd.concat([labels, read, df], axis = 1, ignore_index=True)

df = df.fillna(0)

df.to_csv(f'{patient}_sentence.csv', index=False)

# --- word

count = 0
df = pd.DataFrame()
sub_df = pd.DataFrame()
word_df = pd.DataFrame()
l = []
labels = pd.read_csv('labels.csv')

for i in range(len(data)):
    
    relation = labels.iloc[i,1]

    for j in range(len(data[i].word)):


        FFD_t1 = data[i].word[j].FFD_t1[0:104]
        FFD_t2 = data[i].word[j].FFD_t2[0:104]

        FFD_a1 = data[i].word[j].FFD_a1[0:104]
        FFD_a2 = data[i].word[j].FFD_a2[0:104]

        FFD_b1 = data[i].word[j].FFD_b1[0:104]
        FFD_b2 = data[i].word[j].FFD_b2[0:104]

        FFD_g1 = data[i].word[j].FFD_g1[0:104]
        FFD_g2 = data[i].word[j].FFD_g2[0:104]

        word = data[i].word[j].content

        arr = np.concatenate((FFD_t1, FFD_t2, FFD_a1, FFD_a2, FFD_b1, FFD_b2, FFD_g1, FFD_g2))

        arr = pd.DataFrame(arr).T
        word = pd.Series(word)
        word_df = word_df.append(word, ignore_index = True)

        sub_df = sub_df.append(arr, ignore_index=True)
        l.append(relation)


df = pd.concat([word_df, sub_df], axis = 1, ignore_index = True)
df = df.fillna(0)
df['label'] = l
df.to_csv(f'{patient[0]}_word.csv', index=False)


# --- concat word

new_df = pd.DataFrame()
for i in sorted(os.listdir('ZAB')):
    if '.csv' in i:
        read = pd.read_csv(f'ZAB/{i}')
        text = []
        df = pd.DataFrame()
        for j in range(len(read)):
            text.append(read['0'][j])
            eeg = read.iloc[j, 1:]
            df = pd.concat([df, eeg],axis=0, ignore_index = True)
            
        df = df.T
        text = ' '.join(text)
        dic = {
            'new_word' : [text]
        }
        dic_df = pd.DataFrame(dic)
        df = pd.concat([dic_df, df],axis=1)
        new_df = pd.concat([new_df, df], ignore_index = True)

        