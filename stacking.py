import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ STACKING CLASS FROM WEEK 10 LAB ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
np.random.seed(1)


class StackingClassifier():

    def __init__(self, classifiers, metaclassifier):
        self.classifiers = classifiers
        self.metaclassifier = metaclassifier

    def fit(self, X, y):
        for clf in self.classifiers:
            clf.fit(X, y)
        X_meta = self._predict_base(X)
        self.metaclassifier.fit(X_meta, y)

    def _predict_base(self, X):
        yhats = []
        for clf in self.classifiers:
            yhat = clf.predict_proba(X)
            yhats.append(yhat)
        yhats = np.concatenate(yhats, axis=1)
        # print(yhats.shape)
        assert yhats.shape[0] == X.shape[0]
        return yhats

    def predict(self, X):
        X_meta = self._predict_base(X)
        yhat = self.metaclassifier.predict(X_meta)
        return yhat

    def score(self, X, y):
        yhat = self.predict(X)
        return accuracy_score(y, yhat)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# READ DATA INTO LISTS
training = open("processed_train_raw.tsv")
dev = open("processed_dev_raw.tsv")
x_train = []
y_train = []
x_dev = []
y_dev = []
x_test = []

for line in training:
    line = line.split("\t")
    x_train.append(line[2])
    y_train.append(line[1])

for line in dev:
    line = line.split("\t")
    x_dev.append(line[2])
    y_dev.append(line[1])


# CREATE A TRAINING DATA VECTOR
vectorizer_t = TfidfVectorizer(stop_words="english")
v_train = vectorizer_t.fit_transform(x_train)


# TOKENIZE THE DEV DATA
x_dev_t = []
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
for str in x_dev:
    tokens = word_tokenize(str)
    tokens = [w for w in tokens if not w in stop_words]
    x_dev_t.append(tokens)

# FIND THE BEST FEATURES
kbest_t = SelectKBest(chi2, k=400)
kbest_t.fit(v_train, y_train)
mask = kbest_t.get_support()
xk_train = kbest_t.transform(v_train)
xk_train = xk_train.toarray()

best_features = []
for i in range(len(mask)):
    if (mask[i]):
        best_features.append(vectorizer_t.get_feature_names()[i])

# GENERATE DEV DATA ARRAY
dev_data = []
for line in x_dev_t:
    instance = []
    for f in best_features:
        instance.append(line.count(f))
    dev_data.append(instance)

dev_data = pd.DataFrame(dev_data)
dev_data = dev_data.values

classifiers = [MultinomialNB(),
               BaggingClassifier(RandomForestClassifier(n_estimators=10), max_samples=0.8, max_features=0.2),
               LogisticRegression(),
               KNeighborsClassifier(n_neighbors=20)]

meta_classifier = MultinomialNB()
stacker = StackingClassifier(classifiers, meta_classifier)
stacker.fit(xk_train, y_train)
print('stacker acc:', stacker.score(dev_data, y_dev))
