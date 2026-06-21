# Auto detect text files and perform LF normalization
* text=auto
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download('stopwords')

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')
import os
folder_name='my project'
os.makedirs(folder_name,exist_ok=True)
print("folder created",folder_name)

data = [
    {
        "Unnamed: 0": 605,
        "label": "ham",
        "text": "Subject: enron methanol ; meter # : 988291",
        "label_num": 0
    },
    {
        "Unnamed: 0": 2349,
        "label": "ham",
        "text": "Subject: hpl nom for january 9 , 2001",
        "label_num": 0
    },
    {
        "Unnamed: 0": 3624,
        "label": "ham",
        "text": "Subject: neon retreat",
        "label_num": 0
    },
    {
        "Unnamed: 0": 4685,
        "label": "spam",
        "text": "Subject: photoshop , windows , office . cheap",
        "label_num": 1
    },
    {
        "Unnamed: 0": 2030,
        "label": "ham",
        "text": "Subject: re : indian springs",
        "label_num": 0
    }
]


df = pd.DataFrame(data)
print(df.head())
sns.countplot(x='label', data=df)
plt.title("Spam vs Ham Distribution")
plt.show()
