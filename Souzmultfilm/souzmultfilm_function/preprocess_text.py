import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import re
import nltk
import lightgbm as lgb

from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from lightgbm import LGBMClassifier
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from lightgbm import LGBMClassifier

# Загрузка русских стоп-слов
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

# Предобработка текста
def preprocess_text(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Очистка от специальных символов
    text = re.sub(r'[^а-яa-z0-9\s]', '', text)
    # Удаление стоп слов
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Стемминг
    stemmer = SnowballStemmer("russian")
    text = ' '.join([stemmer.stem(word) for word in text.split()])

    return text