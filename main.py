import re
import pandas as pd
import matplotlib.pyplot

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from wordcloud import WordCloud
from tqdm import tqdm

def cleanText(input):
    emoticons = re.findall('[;:][^\w|\s|;|:]?[^\w|\s|;|:]', input)
    result = input.lower()
    result = re.sub('\d', '', result)
    result = re.sub('<[^>]*>', '', result)
    result = re.sub('[!"#$%&\'()*+,\-./:;<=>?@[\]^_`{|}~]', '', result)
    result = " ".join(result.split())
    for emoticon in emoticons:
        result += emoticon
    return result

def stemming(input: str) -> list:
    wordsList = []
    ps = PorterStemmer()
    for word in [slowo for slowo in re.split('; |, | ', input.lower())]:
        wordsList.append(ps.stem(word))
    return wordsList

def textFiltering(input: list):
    # text = ' '.join([slowo for slowo in re.split('; |, | ', text.lower()) if slowo not in stop_words])
    # return text
    stopWords = stopwords.words('english')
    return [word for word in input if word not in stopWords and len(word) > 3]

def text_tokenizer(input: str) -> list:
    text = cleanText(input)
    text = stemming(text)
    wordsList = textFiltering(text)
    return wordsList


vectorizer = TfidfVectorizer(tokenizer=text_tokenizer)

#

input = 'Lorem 666 ipsum dolor :) sit amet, consectetur; adipiscing elit. Sed eget mattis sem. ;) Mauris ;( egestas erat quam, :< ut faucibus eros congue :> et. In blandit, mi eu porta; lobortis, tortor :-) nisl facilisis leo, at ;< tristique augue risus eu risus ;-).'
result = cleanText(input)
print(result)
result1 = stemming(result)
print(result1)
result3 = textFiltering(input)
print(result3)
result4 = text_tokenizer(input)
print(result4)

vectorizer = CountVectorizer()
X_transform = vectorizer.fit_transform(text_tokenizer(input))
print(X_transform.toarray())