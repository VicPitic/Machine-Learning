from flask import Flask, redirect, render_template, url_for, request

import pandas as pd
import numpy as np
import re
import nltk
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder


"""
Old
twenty_train = fetch_20newsgroups(subset="train", shuffle=True)

categories = ['alt.atheism',
              'comp.graphics',
              'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware',
              'comp.windows.x',
              'misc.forsale',
              'rec.autos',
              'rec.motorcycles',
              'rec.sport.baseball',
              'rec.sport.hockey',
              'sci.crypt',
              'sci.electronics',
              'sci.med',
              'sci.space',
              'soc.religion.christian',
              'talk.politics.guns',
              'talk.politics.misc',
              'talk.religion.misc']

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

tfid_transformer = TfidfTransformer()
X_train_tfidf = tfid_transformer.fit_transform(X_train_counts)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

docs_new = []
docs_new.append(word)
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfid_transformer.transform(X_new_counts)

predicted_text = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted_text):
    print(twenty_train.target_names[category])
"""
def PredictCategory(word):
    df = pd.read_csv("dataset.csv")

    df.head()

    df.groupby("Cat2").mean()

    df.dropna(inplace=True)

    df.shape

    df.head()

    df1 = df[["Text", "Cat2"]]

    df1.shape

    enc = LabelEncoder().fit(df1.Cat2)
    encoded = enc.transform(df1.Cat2)

    df.head()

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(df.Text, encoded, test_size=0.03, random_state=42)

    hh = word

    X_test.reset_index(drop=True, inplace=True)

    X_test[0] = hh

    X_test

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(stop_words="english", decode_error="ignore")
    vectorizer.fit(X_train)

    from sklearn.naive_bayes import MultinomialNB

    cls = MultinomialNB()

    cls.fit(vectorizer.transform(X_train), y_train)

    y_pred = cls.predict(vectorizer.transform(X_test))

    lss = enc.inverse_transform(y_pred)

    #print("The category matched is =", end=" ")
    #print(lss[0])
    return lss[0]

@app.route("/", methods=["POST", "GET"])
def main():
    if(request.method == "POST"):
        word = request.form["word_placeholder"]
        return redirect(url_for("cuvant", wrd=word))
    else:
        return render_template("index.html")

@app.route("/cuvant")
def cuvant():
    cuv = PredictCategory(request.args.get("wrd"))
    return render_template("cuvant.html", wrd=cuv)

if(__name__ == "__main__"):
    app.run()