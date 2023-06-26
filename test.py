import pandas as pd

sentiment_labels = pd.read_csv('data/sentiment_labels_clean.csv')

labels = []
for i in range(len(sentiment_labels)):
    label = sentiment_labels.iloc[i, 2]
    labels.append(label)
    
print(labels.count(0.0))
print(labels.count(1.0))
print(labels.count(-1.0))