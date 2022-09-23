import pandas as pd 
import numpy as np
import os


def to_emotion(rating):
  rating = int(rating)
  if rating == 2:
    return 1
  elif rating == 4:
    return 3
  elif rating == 3:
    return 2
  elif rating == 5:
    return 4
  else: 
    return 0


for p in range(10, 33, 2): 

    read= pd.read_csv(f'eeg_emotion_6class/even/0{p}_eeg.csv')
    read2 = pd.read_csv(f'eeg_emotion_6class/even/0{p}_eeg.csv')
    read2 = read2.drop_duplicates(subset=['seconds'], keep='first')
    read2 = read2[['pid', 'trans_time', 'seconds', 'arousal', 'valence',
    'happy', 'angry', 'nervous', 'sad']].reset_index()

    read = read.astype({'delta': str,
                    'lowAlpha' : str,
                    'highAlpha' : str,
                    'lowBeta' : str,
                    'highBeta' : str,
                    'lowGamma' : str,
                    'middleGamma' : str,
                    'theta' : str
                             })


    x = read.groupby('seconds')[['delta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta',
       'lowGamma', 'middleGamma', 'theta']].agg(','.join).reset_index()

    delta = x['delta'].str.split(',', expand=True)
    delta = delta.rename(columns = {0: 'delta0',
                                1: 'delta1',
                                2: 'delta2',
                                3: 'delta3',
                                4: 'delta4',
                                5: 'delta5',
                                6: 'delta6'})
    lowAlpha = x['lowAlpha'].str.split(',', expand=True)
    lowAlpha = lowAlpha.rename(columns = {0: 'lowAlpha0',
                                1: 'lowAlpha1',
                                2: 'lowAlpha2',
                                3: 'lowAlpha3',
                                4: 'lowAlpha4',
                                5: 'lowAlpha5',
                                6: 'lowAlpha6'})
    highAlpha = x['highAlpha'].str.split(',', expand=True)
    highAlpha = highAlpha.rename(columns = {0: 'highAlpha0',
                                1: 'highAlpha1',
                                2: 'highAlpha2',
                                3: 'highAlpha3',
                                4: 'highAlpha4',
                                5: 'highAlpha5',
                                6: 'highAlpha6'})
    lowBeta = x['lowBeta'].str.split(',', expand=True)
    lowBeta = lowBeta.rename(columns = {0: 'lowBeta0',
                                1: 'lowBeta1',
                                2: 'lowBeta2',
                                3: 'lowBeta3',
                                4: 'lowBeta4',
                                5: 'lowBeta5',
                                6: 'lowBeta6'})
    highBeta = x['highBeta'].str.split(',', expand=True)
    highBeta = highBeta.rename(columns = {0: 'highBeta0',
                                1: 'highBeta1',
                                2: 'highBeta2',
                                3: 'highBeta3',
                                4: 'highBeta4',
                                5: 'highBeta5',
                                6: 'highBeta6'})
    lowGamma = x['lowGamma'].str.split(',', expand=True)
    lowGamma = lowGamma.rename(columns = {0: 'lowGamma0',
                                1: 'lowGamma1',
                                2: 'lowGamma2',
                                3: 'lowGamma3',
                                4: 'lowGamma4',
                                5: 'lowGamma5',
                                6: 'lowGamma6'})
    middleGamma = x['middleGamma'].str.split(',', expand=True)
    middleGamma = middleGamma.rename(columns = {0: 'middleGamma0',
                                1: 'middleGamma1',
                                2: 'middleGamma2',
                                3: 'middleGamma3',
                                4: 'middleGamma4',
                                5: 'middleGamma5',
                                6: 'middleGamma6'})
    theta = x['theta'].str.split(',', expand=True)
    theta = theta.rename(columns = {0: 'theta0',
                                1: 'theta1',
                                2: 'theta2',
                                3: 'theta3',
                                4: 'theta4',
                                5: 'theta5',
                                6: 'theta6'})
    df = pd.concat([delta, lowAlpha, highAlpha, lowBeta, highBeta, lowGamma, middleGamma, theta, read2], axis= 1)

    columns = ['delta0', 'delta1', 'delta2', 'delta3', 'delta4', 'delta5', 
           'lowAlpha0', 'lowAlpha1', 'lowAlpha2', 'lowAlpha3', 'lowAlpha4', 'lowAlpha5',
           'highAlpha0', 'highAlpha1', 'highAlpha2', 'highAlpha3', 'highAlpha4', 'highAlpha5', 
           'lowBeta0', 'lowBeta1', 'lowBeta2', 'lowBeta3', 'lowBeta4', 'lowBeta5', 
           'highBeta0', 'highBeta1', 'highBeta2', 'highBeta3', 'highBeta4', 'highBeta5', 
           'lowGamma0', 'lowGamma1', 'lowGamma2', 'lowGamma3', 'lowGamma4', 'lowGamma5', 
           'middleGamma0', 'middleGamma1', 'middleGamma2', 'middleGamma3', 'middleGamma4', 'middleGamma5',
           'theta0', 'theta1', 'theta2', 'theta3', 'theta4', 'theta5']

    df[columns] = df[columns].apply(pd.to_numeric)
    
    new_df = df[columns]
    
    
    df = df.apply(pd.to_numeric)

    
    df.angry = df.angry.round().astype(int)
    df.nervous = df.nervous.round().astype(int)
    df.sad = df.sad.round().astype(int)
    df.happy = df.happy.round().astype(int)
    df.arousal = df.arousal.round().astype(int)
    df.valence = df.valence.round().astype(int)

    df = df[['delta0', 'lowAlpha0', 'highAlpha0','lowBeta0','highBeta0', 'lowGamma0', 'middleGamma0', 'theta0',
         'delta1', 'lowAlpha1', 'highAlpha1', 'lowBeta1', 'highBeta1', 'lowGamma1', 'middleGamma1', 'theta1',
         'delta2', 'lowAlpha2', 'highAlpha2', 'lowBeta2', 'highBeta2', 'lowGamma2', 'middleGamma2', 'theta2',
         'delta3', 'lowAlpha3', 'highAlpha3', 'lowBeta3', 'highBeta3', 'lowGamma3', 'middleGamma3', 'theta3',
         'delta4', 'lowAlpha4', 'highAlpha4', 'lowBeta4', 'highBeta4', 'lowGamma4', 'middleGamma4', 'theta4',
         'delta5', 'lowAlpha5', 'highAlpha5', 'lowBeta5', 'highBeta5', 'lowGamma5', 'middleGamma5', 'theta5',
         'pid', 'trans_time', 'seconds', 'arousal', 'valence', 'happy', 'angry', 'nervous', 'sad']]
    
    df = df.fillna(0)
    
    df['arousal'] = df.arousal.apply(to_emotion)
    df['valence'] = df.valence.apply(to_emotion)
    df['happy'] = df.happy.apply(to_emotion)
    df['angry'] = df.angry.apply(to_emotion)
    df['nervous'] = df.nervous.apply(to_emotion)
    df['sad'] = df.sad.apply(to_emotion)

    df.to_csv(f'eeg_separate/even/0{p}_eeg_norm.csv')
            

new_df = pd.DataFrame()
new_df2 = pd.DataFrame()

for i in sorted(os.listdir('eeg_separate/odd')):
    if '.csv' in i:
       
        read1 = pd.read_csv(f'eeg_separate/odd/{i}')
    
        new_df = pd.concat([new_df, read1], axis=0)

new_df = new_df.reset_index()

for i in sorted(os.listdir('eeg_separate/even')):
    if '.csv' in i:
        read2 = pd.read_csv(f'eeg_separate/even/{i}')
        new_df2 = pd.concat([new_df2, read2], axis=0)

new_df2 = new_df2.reset_index()

new_df = pd.concat([new_df, new_df2], axis = 1)

new_df = new_df.fillna(0)


new_df.to_csv('df.csv')


df = pd.read_csv('word.csv')
df_eeg = pd.read_csv('df.csv')

df = df[['new_words',
       'seconds', 'arousal_trans', 'valence_trans', 'happy_trans',
       'angry_trans', 'nervous_trans', 'sad_trans', 'seconds2_trans',
       'arousal2_trans', 'valence2_trans', 'happy2_trans', 'angry2_trans',
       'nervous2_trans', 'sad2_trans']]

columns = ['delta0', 'delta1', 'delta2', 'delta3', 'delta4', 'delta5', 
           'lowAlpha0', 'lowAlpha1', 'lowAlpha2', 'lowAlpha3', 'lowAlpha4', 'lowAlpha5',
           'highAlpha0', 'highAlpha1', 'highAlpha2', 'highAlpha3', 'highAlpha4', 'highAlpha5', 
           'lowBeta0', 'lowBeta1', 'lowBeta2', 'lowBeta3', 'lowBeta4', 'lowBeta5', 
           'highBeta0', 'highBeta1', 'highBeta2', 'highBeta3', 'highBeta4', 'highBeta5', 
           'lowGamma0', 'lowGamma1', 'lowGamma2', 'lowGamma3', 'lowGamma4', 'lowGamma5', 
           'middleGamma0', 'middleGamma1', 'middleGamma2', 'middleGamma3', 'middleGamma4', 'middleGamma5',
           'theta0', 'theta1', 'theta2', 'theta3', 'theta4', 'theta5',
          'delta0_2', 'delta1_2', 'delta2_2', 'delta3_2', 'delta4_2', 'delta5_2', 
           'lowAlpha0_2', 'lowAlpha1_2', 'lowAlpha2_2', 'lowAlpha3_2', 'lowAlpha4_2', 'lowAlpha5_2',
           'highAlpha0_2', 'highAlpha1_2', 'highAlpha2_2', 'highAlpha3_2', 'highAlpha4_2', 'highAlpha5', 
           'lowBeta0_2', 'lowBeta1_2', 'lowBeta2_2', 'lowBeta3_2', 'lowBeta4_2', 'lowBeta5_2', 
           'highBeta0_2', 'highBeta1_2', 'highBeta2_2', 'highBeta3_2', 'highBeta4_2', 'highBeta5_2', 
           'lowGamma0_2', 'lowGamma1_2', 'lowGamma2_2', 'lowGamma3_2', 'lowGamma4_2', 'lowGamma5_2', 
           'middleGamma0_2', 'middleGamma1_2', 'middleGamma2_2', 'middleGamma3_2', 'middleGamma4_2', 'middleGamma5_2',
           'theta0_2', 'theta1_2', 'theta2_2', 'theta3_2', 'theta4_2', 'theta5_2']

df_eeg = df_eeg[columns]


df = pd.concat([df, df_eeg], axis = 1)

df.to_csv('df.csv')
                       