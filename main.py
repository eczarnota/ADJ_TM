import re

import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from wordcloud import WordCloud
from tqdm import tqdm

def cleanText(text: str) -> str:
    text = text.lower()
    text = re.sub('\d', '', text)
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[!"#$%&\'()*+,\-./:;<=>?@[\]^_`{|}~]', '', text)
    return text

def stemming(wordsList: list) -> list:
    ps = PorterStemmer()
    return [ps.stem(word) for word in wordsList]

def stemmingStr(input: str) -> list:
    wordsList = []
    ps = PorterStemmer()
    for word in [slowo for slowo in re.split('; |, | ', input.lower())]:
        wordsList.append(ps.stem(word))
    return wordsList

def textFiltering(input: list) -> list:
    stopWords = stopwords.words('english')
    return [word for word in input if word not in stopWords and len(word) > 3]

def text_tokenizer(input: str) -> list:
    text = cleanText(input)
    wordsList = text.split(" ")
    wordsList = stemming(wordsList)
    wordsList = textFiltering(wordsList)
    return wordsList

def createBow(words: list) -> dict:
    bow = {}
    for word in words:
        if word not in bow.keys():
            bow[word] = 1
        else:
            bow[word] += 1
    return bow

dataset = pd.read_csv('alexa_reviews.csv', sep=";", encoding='cp1252')
print(dataset.head())
dataset.info()
dataset = dataset.drop_duplicates()
print(dataset.groupby('rating').describe())

reviews = dataset['verified_reviews']

txt = ""
for i in tqdm(range(len(reviews))):
    txt += reviews.iloc[i] + " "
stemmed = textFiltering(stemming(cleanText(txt).split()))
bow = createBow(stemmed)
wc = WordCloud()
wc.generate_from_frequencies(bow)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

countVectorizer = CountVectorizer(tokenizer=text_tokenizer)
X_transform = countVectorizer.fit_transform(dataset['verified_reviews'])
x_train, x_test, y_train, y_test = train_test_split(X_transform, dataset['rating'], test_size=0.5, random_state=0)

linearSvc = LinearSVC()
fig, ax = plt.subplots(1,1)
linearSvc.fit(x_train, y_train)
y_pred = linearSvc.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
cm = normalize(cm, axis=0, norm='l1')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=linearSvc.classes_)
ax.title.set_text(f"{linearSvc}")
disp.plot(ax=ax)
plt.show()

