import scipy.io as io
import os
import pandas as pd
import numpy as np
import scipy.stats as stats


# --- sentence 

patient = ['ZAB', 'ZDM', 'ZDN', 'ZJM', 'ZJN', 'ZJS', 'ZKH', 'ZKW', 'ZMG']
file_name = f"task1- SR/Matlab_files/results{patient[0]}_SR.mat"

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
eeg = pd.DataFrame()
L = []

for j in range(len(data)):

    arr =  data[j].mean_g2

    if np.isnan(arr).all():

        arr = [0] * 105

    df = pd.DataFrame(arr).T

    eeg = pd.concat([eeg, df], axis = 0)

eeg= eeg.reset_index()

for k in range(len(data)):
    sent = data[k].content
    L.append(sent)

sent_df = pd.DataFrame(L, columns = ['new_words'])

new_df = pd.concat([sent_df, eeg], axis = 1)
new_df.to_csv(f'eeg/{patient}_mean_g2_df.csv')



df = pd.DataFrame()
for i in sorted(os.listdir('eeg')):
    if'.csv' in i:
        read= pd.read_csv(f'eeg/{i}')
        df = pd.concat([df, read],axis=1)

df.to_csv(f'{patient[0]}_sentence.csv')


sentiment = pd.read_csv('label.csv')

read = pd.read_csv(f'{patient[0]}_sentence.csv')
df = pd.concat([sentiment, read], axis = 1)
df.to_csv(f'{patient[0]}_sentence.csv')

read = pd.read_csv(f'{patient[0]}_sentence.csv')
data = read.loc[read['2'] !=0]

data.to_csv(f'{patient[0]}_sentence.csv', index = False)


# --- word

count = 0
for i in range(len(data)):
    
    
    df = pd.DataFrame()
    sub_df = pd.DataFrame()
    word_df = pd.DataFrame()
    
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

        df = df.append(arr, ignore_index=True)                

    df = pd.concat([word_df, df], axis = 1, ignore_index = True)

    df.to_csv(f'{patient[0]}{count}_word.csv', index=False)
    count+=1


for i in sorted(os.listdir()):
    if '_word.csv' in i:
        df = pd.DataFrame()
        patient = i[0:3]
        for j in sorted(os.listdir(f'{i}')):
            
            read = pd.read_csv(f'{i}/{j}')
            df = df.append(read, ignore_index = True)
            
        df.to_csv(f'{patient[0]}_word.csv', index = False)

label = pd.read_csv('label_SA.csv', header = None)

words = []
y = []
for i in range(len(data)):
    sentiment = label.iloc[i]
    
    for j in range(len(data[i].word)):
        
        words.append(data[i].word[j].content)
        y.append(sentiment)

l = pd.DataFrame(y)
l.to_csv('y.csv', index = False)

l = pd.read_csv('y.csv')

new_df = pd.concat([l, df], axis=1)

new_df.to_csv(f'{patient[0]}_word.csv')

df = pd.read_csv(f'{patient[0]}_word.csv')
        
data = df.loc[df['2'] !=0]

data.to_csv(f'{patient[0]}_word.csv', index = False)


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

        