import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
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

# FIND THE OPTIMAL NUMBER OF TREES IN RF
randomForest = []
for n in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators=n)
    rf.fit(xk_train, y_train)
    rf_acc = accuracy_score(rf.predict(dev_data), y_dev)
    randomForest.append(rf_acc)


# FIND THE OPTIMAL VALUE OF K FOR KNN
kNearestNeighbour = []
for k in range(5, 20, 5):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(xk_train, y_train)
    knn_acc = accuracy_score(knn.predict(dev_data), y_dev)
    kNearestNeighbour.append(knn_acc)

fout = open("hyperparameter-tuning-output.txt", "w")
fout.write("\nKNN\n")
for i in kNearestNeighbour:
    fout.write("{}, ".format(i))
fout.write("\nRandom Forest\n")
for i in randomForest:
    fout.write("{}, ".format(i))
fout.close()