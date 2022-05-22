import pandas as pd
from sklearn.feature_extraction.text import (
    CountVectorizer
)


df_raiting = pd.read_csv(r"C:\Users\EC\TextMining\alexa_reviews.csv", header=0, names=['rating'])
df_verification = pd.read_csv(r"C:\Users\EC\TextMining\alexa_reviews.csv", header=0, names=['verified_reviews'])

df_joined = pd.concat([df_raiting, df_verification])
