import pandas as pd
import matplotlib.pyplot

from tqdm import tqdm
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def cleanText(text: str) -> str:
    result_em = re.findall('[;:][^\w|\s|;|:]?[^\w|\s|;|:]', text)
    result = text.lower()
    result = re.sub('\d', '', result)
    result = re.sub('<[^>]*>', '', result)
    result = re.sub('[!"#$%&\'()*+,\-./:;<=>?@[\]^_`{|}~]', '', result)
    for emoticon in result_em:
        result += emoticon
    return result

def remove_stop_words(text: list) -> list:
    stop_words = set(stopwords.words('english'))
    list_of_words = text
    return [word for word in list_of_words if word not in stop_words]

def stemming(list_text: list) -> list:
    ps = PorterStemmer()
    return [ps.stem(word) for word in list_text]


def text_tokenizer(text: str) -> list:
    todo = text
    todo = cleanText(todo)
    todo_list = todo.split(" ")
    todo_list = stemming(todo_list)
    todo_list = remove_stop_words(todo_list)
    return [word for word in todo_list if len(word) > 3]
