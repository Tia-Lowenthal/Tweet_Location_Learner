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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FINDING BEST K FEATURES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

multinomialNB = []
kNearestNeighbour = []
logisticRegression = []
randomForest = []

for k in range(50, 651, 50):
    kbest_t = SelectKBest(chi2, k=k)
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

    # ~~~~~~~~~~~~~~~~ NAIVE BAYES ~~~~~~~~~~~~~~~~
    mnb = MultinomialNB()
    mnb.fit(xk_train, y_train)
    mnb_acc = accuracy_score(mnb.predict(dev_data), y_dev)

    # ~~~~~~~~~~~~~~~~ LOGISTIC REGRESSION ~~~~~~~~~~~~~~~~
    logReg = LogisticRegression()
    logReg.fit(xk_train, y_train)
    logreg_acc = accuracy_score(logReg.predict(dev_data), y_dev)
    logisticRegression.append(accuracy_score(logReg.predict(dev_data), y_dev))

    # ~~~~~~~~~~~~~~~~ K NEAREST NEIGHBOUR ~~~~~~~~~~~~~~~~

    knn = KNeighborsClassifier()
    knn.fit(xk_train, y_train)
    knn_acc = accuracy_score(knn.predict(dev_data), y_dev)
    kNearestNeighbour.append(knn_acc)

    # ~~~~~~~~~~~~~~~~ RANDOM FOREST ~~~~~~~~~~~~~~~~

    rf = RandomForestClassifier()
    rf.fit(xk_train, y_train)
    rf_acc = accuracy_score(rf.predict(dev_data), y_dev)
    randomForest.append(rf_acc)


fout = open("bestk-output.txt", "w")
fout.write("Gaussian NB\n")
for i in gaussianNB:
    fout.write("{}, ".format(i))
fout.write("\nMultinomial NB\n")
for i in multinomialNB:
    fout.write("{}, ".format(i))
fout.write("\nBernoulli NB\n")
for i in bernoulliNB:
    fout.write("{}, ".format(i))
fout.write("\nLinear SVM\n")
for i in linearSVM:
    fout.write("{}, ".format(i))
fout.write("\nRBF SVM\n")
for i in rbfSVM:
    fout.write("{}, ".format(i))
fout.write("\nLogistic Regression\n")
for i in logisticRegression:
    fout.write("{}, ".format(i))
fout.write("\nKNN\n")
for i in kNearestNeighbour:
    fout.write("{}, ".format(i))
fout.write("\nRandom Forest\n")
for i in randomForest:
    fout.write("{}, ".format(i))
fout.close()