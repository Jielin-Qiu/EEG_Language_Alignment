import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import os
import seaborn as sns
from sklearn.manifold import TSNE
from embeddings import *
from config import *
from sklearn.model_selection import train_test_split
import torch
import argparse
import scipy.io as io
import mne
from mne import EvokedArray

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--model', type=str, help="Please choose a model from the following list: ['transformer', 'biLSTM', 'MLP', 'resnet', 'fusion', 'CCA_fusion', 'CCA_ds', 'WD_fusion', 'WD_ds']")
    parser.add_argument('--modality', type = str, help="Please choose a modality from the following list: ['eeg', 'text']")
    parser.add_argument('--dataset', type=str, help="Please choose a dataset from the following list: ['KEmoCon', 'ZuCo']")
    parser.add_argument('--task', default ='SA', type=str, help="If dataset == Zuco, please choose a task from the following list: ['SA', 'RD']")
    parser.add_argument('--level', type=str, default = 'sentence', help="If ZuCo, please choose the level of EEG feature you want to work with from this list: ['word', 'concatword', 'sentence']")
    parser.add_argument('--plot', type=str, help = "Please choose the type of plot from the following list: ['TSNE', 'brain_topo', 'word_align']")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.plot == 'TSNE':
        if args.dataset == 'KEmoCon':

            df = pd.read_csv('preprocessed_kemo/KEmoCon/df.csv')

            X = df.drop([emotion], axis = 1)
            y= df[[emotion]]

            X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 2, test_size = 0.2, shuffle = True, stratify = y)
            X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, random_state= 2, test_size = 0.5, shuffle = True, stratify = y_val)
            df_test = pd.concat([X_test, y_test], axis = 1)
            df_train = pd.concat([X_train, y_train], axis = 1)
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

            embeddings_train = get_embeddings(df_train_text[:,1], device)
            embeddings_val = get_embeddings(df_val_text[:,1], device)
            embeddings_test = get_embeddings(df_test_text[:,1], device)

            if args.modality == 'text':

                X_val_text = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred_val.txt')
                X_test_text = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred.txt')

                y_val_text = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label_val.txt')
                y_test_text = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label.txt')
            
            if args.modality == 'eeg':
                X_val_eeg = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred_val.txt')
                X_test_eeg = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred.txt')

                y_val_eeg = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label_val.txt')
                y_test_eeg = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label.txt')
            
            if args.modality == 'fusion':
                X_val_fusion = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred_val.txt')
                X_test_fusion = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred.txt')

                y_val_fusion = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label_val.txt')
                y_test_fusion = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label.txt')

            # --- RAW
            x = embeddings_train
            y = df_train_text[:, 0]

            tsne = TSNE(n_components=2, verbose=1, random_state=42, perplexity = 20, n_iter = 5000, learning_rate = 'auto')
            z = tsne.fit_transform(x) 
            df = pd.DataFrame()
            df["y"] = y
            df["comp-1"] = z[:,0]
            df["comp-2"] = z[:,1]

            sns.scatterplot(x = 'comp-1', y = 'comp-2', hue=df.y.tolist(),
                            palette=sns.color_palette("hls", 3),
                            data=df).set(title="Raw Text") 

            plt.savefig('rawtext_tsne.png', bbox_inches="tight")

            x =df_train_eeg[:, 1:]
            y = df_train_eeg[:, 0]

            tsne = TSNE(n_components=2, verbose=1, random_state=42, perplexity = 20, n_iter = 5000, learning_rate = 'auto')
            z = tsne.fit_transform(x) 
            df = pd.DataFrame()
            df["y"] = y
            df["comp-1"] = z[:,0]
            df["comp-2"] = z[:,1]
            sns.set(font_scale=1.5)

            sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                            palette=sns.color_palette("hls", 3),
                            data=df).set(title='Raw EEG') 

            plt.savefig('raweeg_tsne.png', bbox_inches="tight")

            # --- DS

            if args.modality == 'text':
                x = X_val_text[11300:]
                y = y_val_text[11300:]

                tsne = TSNE(n_components=2, verbose=1, random_state=42, perplexity = 20, n_iter = 5000, learning_rate = 'auto')
                z = tsne.fit_transform(x) 
                df = pd.DataFrame()

                df["y"] = y
                df["comp-1"] = z[:,0]
                df["comp-2"] = z[:,1]

                sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                                palette=sns.color_palette("hls", 3),
                                data=df).set(title=f"{args.modality} {args.model}") 
                plt.savefig(f'{args.modality}_{args.model}.png', bbox_inches="tight")

            if args.modality == 'eeg':
                x = X_val_eeg[11300:]
                y = y_val_eeg[11300:]

                tsne = TSNE(n_components=2, verbose=1, random_state=42, perplexity = 20, n_iter = 5000, learning_rate = 'auto')
                z = tsne.fit_transform(x) 
                df = pd.DataFrame()

                df["y"] = y
                df["comp-1"] = z[:,0]
                df["comp-2"] = z[:,1]

                sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                                palette=sns.color_palette("hls", 3),
                                data=df).set(title=f"{args.modality} {args.model}") 
                plt.savefig(f'{args.modality}_{args.model}.png', bbox_inches="tight")



            if args.modality == 'fusion':
                x = X_val_fusion[11300:]
                y = y_val_fusion[11300:]

                tsne = TSNE(n_components=2, verbose=1, random_state=42, perplexity = 20, n_iter = 5000, learning_rate = 'auto')
                z = tsne.fit_transform(x) 
                df = pd.DataFrame()

                df["y"] = y
                df["comp-1"] = z[:,0]
                df["comp-2"] = z[:,1]

                sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                                palette=sns.color_palette("hls", 3),
                                data=df).set(title=f"{args.modality} {args.model}") 
                plt.savefig(f'{args.modality}_{args.model}.png', bbox_inches="tight")


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

            X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, random_state= 2, test_size = 0.5, shuffle = True)
            df_test = pd.concat([X_test, y_test], axis = 1)
            df_train = pd.concat([X_train, y_train], axis = 1)
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

            embeddings_train = get_embeddings(df_train_text[:,1], device)
            embeddings_val = get_embeddings(df_val_text[:,1], device)
            embeddings_test = get_embeddings(df_test_text[:,1], device)

            if args.modality == 'text':

                X_val_text = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred_val.txt')
                X_test_text = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred.txt')

                y_val_text = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label_val.txt')
                y_test_text = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label.txt')
            
            if args.modality == 'eeg':
                X_val_eeg = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred_val.txt')
                X_test_eeg = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred.txt')

                y_val_eeg = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label_val.txt')
                y_test_eeg = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label.txt')
            
            if args.modality == 'fusion':
                X_val_fusion = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred_val.txt')
                X_test_fusion = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred.txt')

                y_val_fusion = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label_val.txt')
                y_test_fusion = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label.txt')

            # --- RAW
            x = embeddings_train
            y = df_train_text[:, 0]

            tsne = TSNE(n_components=2, verbose=1, random_state=42, perplexity = 20, n_iter = 5000, learning_rate = 'auto')
            z = tsne.fit_transform(x) 
            df = pd.DataFrame()
            df["y"] = y
            df["comp-1"] = z[:,0]
            df["comp-2"] = z[:,1]

            sns.scatterplot(x = 'comp-1', y = 'comp-2', hue=df.y.tolist(),
                            palette=sns.color_palette("hls", 3),
                            data=df).set(title="Raw Text") 

            plt.savefig('rawtext_tsne.png', bbox_inches="tight")

            x =df_train_eeg[:, 1:]
            y = df_train_eeg[:, 0]

            tsne = TSNE(n_components=2, verbose=1, random_state=42, perplexity = 20, n_iter = 5000, learning_rate = 'auto')
            z = tsne.fit_transform(x) 
            df = pd.DataFrame()
            df["y"] = y
            df["comp-1"] = z[:,0]
            df["comp-2"] = z[:,1]
            sns.set(font_scale=1.5)

            sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                            palette=sns.color_palette("hls", 3),
                            data=df).set(title='Raw EEG') 

            plt.savefig('raweeg_tsne.png', bbox_inches="tight")

            # --- DS

            if args.modality == 'text':
                x = X_val_text[11300:]
                y = y_val_text[11300:]

                tsne = TSNE(n_components=2, verbose=1, random_state=42, perplexity = 20, n_iter = 5000, learning_rate = 'auto')
                z = tsne.fit_transform(x) 
                df = pd.DataFrame()

                df["y"] = y
                df["comp-1"] = z[:,0]
                df["comp-2"] = z[:,1]

                sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                                palette=sns.color_palette("hls", 3),
                                data=df).set(title=f"{args.modality} {args.model}") 
                plt.savefig(f'{args.modality}_{args.model}.png', bbox_inches="tight")

            if args.modality == 'eeg':
                x = X_val_eeg[11300:]
                y = y_val_eeg[11300:]

                tsne = TSNE(n_components=2, verbose=1, random_state=42, perplexity = 20, n_iter = 5000, learning_rate = 'auto')
                z = tsne.fit_transform(x) 
                df = pd.DataFrame()

                df["y"] = y
                df["comp-1"] = z[:,0]
                df["comp-2"] = z[:,1]

                sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                                palette=sns.color_palette("hls", 3),
                                data=df).set(title=f"{args.modality} {args.model}") 
                plt.savefig(f'{args.modality}_{args.model}.png', bbox_inches="tight")



            if args.modality == 'fusion':
                x = X_val_fusion[11300:]
                y = y_val_fusion[11300:]

                tsne = TSNE(n_components=2, verbose=1, random_state=42, perplexity = 20, n_iter = 5000, learning_rate = 'auto')
                z = tsne.fit_transform(x) 
                df = pd.DataFrame()

                df["y"] = y
                df["comp-1"] = z[:,0]
                df["comp-2"] = z[:,1]

                sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                                palette=sns.color_palette("hls", 3),
                                data=df).set(title=f"{args.modality} {args.model}") 
                plt.savefig(f'{args.modality}_{args.model}.png', bbox_inches="tight")


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

            X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, random_state= 2, test_size = 0.5, shuffle = True)
            df_test = pd.concat([X_test, y_test], axis = 1)
            df_train = pd.concat([X_train, y_train], axis = 1)
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

            embeddings_train = get_embeddings(df_train_text[:,1], device)
            embeddings_val = get_embeddings(df_val_text[:,1], device)
            embeddings_test = get_embeddings(df_test_text[:,1], device)

            if args.modality == 'text':

                X_val_text = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred_val.txt')
                X_test_text = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred.txt')

                y_val_text = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label_val.txt')
                y_test_text = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label.txt')
            
            if args.modality == 'eeg':
                X_val_eeg = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred_val.txt')
                X_test_eeg = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred.txt')

                y_val_eeg = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label_val.txt')
                y_test_eeg = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label.txt')
            
            if args.modality == 'fusion':
                X_val_fusion = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred_val.txt')
                X_test_fusion = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred.txt')

                y_val_fusion = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label_val.txt')
                y_test_fusion = np.loadtxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label.txt')

            # --- RAW
            x = embeddings_train
            y = df_train_text[:, 0]

            tsne = TSNE(n_components=2, verbose=1, random_state=42, perplexity = 20, n_iter = 5000, learning_rate = 'auto')
            z = tsne.fit_transform(x) 
            df = pd.DataFrame()
            df["y"] = y
            df["comp-1"] = z[:,0]
            df["comp-2"] = z[:,1]

            sns.scatterplot(x = 'comp-1', y = 'comp-2', hue=df.y.tolist(),
                            palette=sns.color_palette("hls", 3),
                            data=df).set(title="Raw Text") 

            plt.savefig('rawtext_tsne.png', bbox_inches="tight")

            x =df_train_eeg[:, 1:]
            y = df_train_eeg[:, 0]

            tsne = TSNE(n_components=2, verbose=1, random_state=42, perplexity = 20, n_iter = 5000, learning_rate = 'auto')
            z = tsne.fit_transform(x) 
            df = pd.DataFrame()
            df["y"] = y
            df["comp-1"] = z[:,0]
            df["comp-2"] = z[:,1]
            sns.set(font_scale=1.5)

            sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                            palette=sns.color_palette("hls", 3),
                            data=df).set(title='Raw EEG') 

            plt.savefig('raweeg_tsne.png', bbox_inches="tight")

            # --- DS

            if args.modality == 'text':
                x = X_val_text[11300:]
                y = y_val_text[11300:]

                tsne = TSNE(n_components=2, verbose=1, random_state=42, perplexity = 20, n_iter = 5000, learning_rate = 'auto')
                z = tsne.fit_transform(x) 
                df = pd.DataFrame()

                df["y"] = y
                df["comp-1"] = z[:,0]
                df["comp-2"] = z[:,1]

                sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                                palette=sns.color_palette("hls", 3),
                                data=df).set(title=f"{args.modality} {args.model}") 
                plt.savefig(f'{args.modality}_{args.model}.png', bbox_inches="tight")

            if args.modality == 'eeg':
                x = X_val_eeg[11300:]
                y = y_val_eeg[11300:]

                tsne = TSNE(n_components=2, verbose=1, random_state=42, perplexity = 20, n_iter = 5000, learning_rate = 'auto')
                z = tsne.fit_transform(x) 
                df = pd.DataFrame()

                df["y"] = y
                df["comp-1"] = z[:,0]
                df["comp-2"] = z[:,1]

                sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                                palette=sns.color_palette("hls", 3),
                                data=df).set(title=f"{args.modality} {args.model}") 
                plt.savefig(f'{args.modality}_{args.model}.png', bbox_inches="tight")



            if args.modality == 'fusion':
                x = X_val_fusion[11300:]
                y = y_val_fusion[11300:]

                tsne = TSNE(n_components=2, verbose=1, random_state=42, perplexity = 20, n_iter = 5000, learning_rate = 'auto')
                z = tsne.fit_transform(x) 
                df = pd.DataFrame()

                df["y"] = y
                df["comp-1"] = z[:,0]
                df["comp-2"] = z[:,1]

                sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                                palette=sns.color_palette("hls", 3),
                                data=df).set(title=f"{args.modality} {args.model}") 
                plt.savefig(f'{args.modality}_{args.model}.png', bbox_inches="tight")


    if args.plot == 'word_align':

        sent = pd.read_csv(f'preprocessed/ZuCo/SA/{patient}_word.csv')

        negative_word = sent[5044:5059]
        positive_word = sent[1482:1501]

        negative_word_eeg = negative_word.iloc[:, 1:]
        theta1_negative = negative_word_eeg.iloc[:, 0:104]
        theta2_negative = negative_word_eeg.iloc[:, 104:208]
        alpha1_negative = negative_word_eeg.iloc[:, 208:312]
        alpha2_negative = negative_word_eeg.iloc[:, 312:416]
        beta1_negative = negative_word_eeg.iloc[:, 416:520]
        beta2_negative = negative_word_eeg.iloc[:, 520:624]
        gamma1_negative = negative_word_eeg.iloc[:, 624:728]
        gamma2_negative = negative_word_eeg.iloc[:, 728:832]

        positive_word_eeg = positive_word.iloc[:, 1:]
        theta1_positive = positive_word_eeg.iloc[:, 0:104]
        theta2_positive = positive_word_eeg.iloc[:, 104:208]
        alpha1_positive = positive_word_eeg.iloc[:, 208:312]
        alpha2_positive = positive_word_eeg.iloc[:, 312:416]
        beta1_positive = positive_word_eeg.iloc[:, 416:520]
        beta2_positive = positive_word_eeg.iloc[:, 520:624]
        gamma1_positive = positive_word_eeg.iloc[:, 624:728]
        gamma2_positive = positive_word_eeg.iloc[:, 728:832]

        theta1_negative_list = []
        theta2_negative_list = []
        alpha1_negative_list = []
        alpha2_negative_list = []
        beta1_negative_list = []
        beta2_negative_list = []
        gamma1_negative_list = []
        gamma2_negative_list = []

        theta1_positive_list = []
        theta2_positive_list = []
        alpha1_positive_list = []
        alpha2_positive_list = []
        beta1_positive_list = []
        beta2_positive_list = []
        gamma1_positive_list = []
        gamma2_positive_list = []

        for i in range(len(positive_word)):
            ave = theta1_positive.values[i].mean()
            theta1_positive_list.append(ave)

            ave = theta2_positive.values[i].mean()
            theta2_positive_list.append(ave)

            ave = alpha1_positive.values[i].mean()
            alpha1_positive_list.append(ave)

            ave = alpha2_positive.values[i].mean()
            alpha2_positive_list.append(ave)

            ave = beta1_positive.values[i].mean()
            beta1_positive_list.append(ave)

            ave = beta2_positive.values[i].mean()
            beta2_positive_list.append(ave)

            ave = gamma1_positive.values[i].mean()
            gamma1_positive_list.append(ave)

            ave = gamma2_positive.values[i].mean()
            gamma2_positive_list.append(ave)

        for i in range(len(negative_word)):
            ave = theta1_negative.values[i].mean()
            theta1_negative_list.append(ave)

            ave = theta2_negative.values[i].mean()
            theta2_negative_list.append(ave)

            ave = alpha1_negative.values[i].mean()
            alpha1_negative_list.append(ave)

            ave = alpha2_negative.values[i].mean()
            alpha2_negative_list.append(ave)

            ave = beta1_negative.values[i].mean()
            beta1_negative_list.append(ave)

            ave = beta2_negative.values[i].mean()
            beta2_negative_list.append(ave)

            ave = gamma1_negative.values[i].mean()
            gamma1_negative_list.append(ave)

            ave = gamma2_negative.values[i].mean()
            gamma2_negative_list.append(ave)
        
        new_df_positive = []
        new_df_negative = []

        for i in range(len(theta1_negative_list)):

            sub = []

            sub.append(theta1_negative_list[i])

            sub.append(theta2_negative_list[i])

            sub.append(alpha1_negative_list[i])

            sub.append(alpha2_negative_list[i])

            sub.append(beta1_negative_list[i])

            sub.append(beta2_negative_list[i])

            sub.append(gamma1_negative_list[i])

            sub.append(gamma2_negative_list[i])

            new_df_negative.append(sub)
        
        for i in range(len(theta1_positive_list)):

            sub = []

            sub.append(theta1_positive_list[i])

            sub.append(theta2_positive_list[i])

            sub.append(alpha1_positive_list[i])

            sub.append(alpha2_positive_list[i])

            sub.append(beta1_positive_list[i])

            sub.append(beta2_positive_list[i])

            sub.append(gamma1_positive_list[i])

            sub.append(gamma2_positive_list[i])

            new_df_positive.append(sub)

        df_positive = pd.DataFrame(new_df_positive)
        df_negative= pd.DataFrame(new_df_negative)

        y_axis_positive = []
        for i in positive_word['0']:
            y_axis_positive.append(i)

        y_axis_negative= []
        for i in negative_word['0']:
            y_axis_negative.append(i)


        x_axis_labels = ['theta1','theta2','alpha1','alpha2','beta1','beta2','gamma1','gamma2']
        y_axis_labels = y_axis_positive
        sns.heatmap(df_positive, xticklabels = x_axis_labels, yticklabels = y_axis_labels)
        plt.savefig('positive_word_heat.png', bbox_inches="tight")

        x_axis_labels = ['theta1','theta2','alpha1','alpha2','beta1','beta2','gamma1','gamma2'] 
        y_axis_labels = y_axis_negative
        sns.heatmap(df_negative, xticklabels = x_axis_labels, yticklabels = y_axis_labels)
        plt.savefig('negative_word_heat.png', bbox_inches="tight")

    if args.plots =='brain_topo':

        file_name = f"task1- SR/Matlab_files_1/results{patient}_SR.mat"

        chanlocs = ['E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E9', 'E10', 'E11', 'E12', 'E13', 'E15', 'E16', 'E18', 'E19', 'E20',
            'E22',
            'E23', 'E24', 'E26', 'E27', 'E28', 'E29', 'E30', 'E31', 'E33', 'E34', 'E35', 'E36', 'E37', 'E38', 'E39',
            'E40',
            'E41', 'E42', 'E43', 'E44', 'E45', 'E46', 'E47', 'E50', 'E51', 'E52', 'E53', 'E54', 'E55', 'E57', 'E58',
            'E59',
            'E60', 'E61', 'E62', 'E64', 'E65', 'E66', 'E67', 'E69', 'E70', 'E71', 'E72', 'E74', 'E75', 'E76', 'E77',
            'E78',
            'E79', 'E80', 'E82', 'E83', 'E84', 'E85', 'E86', 'E87', 'E89', 'E90', 'E91', 'E92', 'E93', 'E95', 'E96',
            'E97',
            'E98', 'E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E108', 'E109', 'E110', 'E111', 'E112',
            'E114',
            'E115', 'E116', 'E117', 'E118', 'E120', 'E121', 'E122', 'E123', 'E124']

        info = mne.create_info(ch_names=chanlocs, ch_types="eeg", sfreq=500)

        data = io.loadmat(file_name, squeeze_me=True, struct_as_record=False)['sentenceData']

        # --- positive
        # --- important
        upscale = data[1].word[7].content
        # --- unimportant words
        will = data[1].word[-4].content
       

        # --- negative
        # --- important words
        lame = data[98].word[1].content
        # --- unimportant words
        someone = data[98].word[9].content

        FFD_t1 = data[98].word[13].FFD_t1
        FFD_t2 = data[98].word[13].FFD_t2

        FFD_a1 = data[98].word[13].FFD_a1
        FFD_a2 = data[98].word[13].FFD_a2

        FFD_b1 = data[98].word[13].FFD_b1
        FFD_b2 = data[98].word[13].FFD_b2

        FFD_g1 = data[98].word[13].FFD_g1
        FFD_g2 = data[98].word[13].FFD_g2

        ll = [FFD_t1, FFD_t2, FFD_a1, FFD_a2, FFD_b1, FFD_b2, FFD_g1, FFD_g2]

        maxx = []
        minn = []
        for i in ll:
            maxx.append(i.max())
            
            minn.append(i.min())
            
        print(max(maxx))
        print(min(minn))


        evoked_nr = EvokedArray(FFD_t1.reshape(-1,1)[:104], info=info)
        evoked_nr.set_montage("GSN-HydroCel-128")
        ax = evoked_nr.plot_topomap(scalings=1, cmap='RdBu_r', vmin = -6.9, vmax = 6.9)
        ax.savefig("FFD_t1.pdf")

        evoked_nr = EvokedArray(FFD_t2.reshape(-1,1)[:104], info=info)
        evoked_nr.set_montage("GSN-HydroCel-128")
        ax = evoked_nr.plot_topomap(scalings=1, cmap='RdBu_r', vmin = -6.9, vmax = 6.9)
        ax.savefig("FFD_t2.pdf")

        evoked_nr = EvokedArray(FFD_a1.reshape(-1,1)[:104], info=info)
        evoked_nr.set_montage("GSN-HydroCel-128")
        ax = evoked_nr.plot_topomap(scalings=1, cmap='RdBu_r', vmin = -6.9, vmax = 6.9)
        ax.savefig("FFD_a1.pdf")


        evoked_nr = EvokedArray(FFD_a2.reshape(-1,1)[:104], info=info)
        evoked_nr.set_montage("GSN-HydroCel-128")
        ax = evoked_nr.plot_topomap(scalings=1, cmap='RdBu_r', vmin = -6.9, vmax = 6.9)
        ax.savefig("FFD_a2.pdf")


        evoked_nr = EvokedArray(FFD_b1.reshape(-1,1)[:104], info=info)
        evoked_nr.set_montage("GSN-HydroCel-128")
        ax = evoked_nr.plot_topomap(scalings=1, cmap='RdBu_r', vmin = -6.9, vmax = 6.9)

        ax.savefig("FFD_b1.pdf")


        evoked_nr = EvokedArray(FFD_b2.reshape(-1,1)[:104], info=info)
        evoked_nr.set_montage("GSN-HydroCel-128")
        ax = evoked_nr.plot_topomap(scalings=1, cmap='RdBu_r', vmin = -6.9, vmax = 6.9)
        ax.savefig("FFD_b2.pdf")

        evoked_nr = EvokedArray(FFD_g1.reshape(-1,1)[:104], info=info)
        evoked_nr.set_montage("GSN-HydroCel-128")
        ax = evoked_nr.plot_topomap(scalings=1, cmap='RdBu_r', vmin = -6.9, vmax = 6.9)

        ax.savefig("FFD_g1.pdf")

        evoked_nr = EvokedArray(FFD_g2.reshape(-1,1)[:104], info=info)
        evoked_nr.set_montage("GSN-HydroCel-128")
        ax = evoked_nr.plot_topomap(scalings=1, cmap='RdBu_r', vmin = -6.9, vmax = 6.9)

        ax.savefig("FFD_g2.pdf")

  
