import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier



# READ DATA INTO LISTS
training = open("processed_train_raw.tsv")
dev = open("processed_dev_raw.tsv")
testing = open("processed_test_raw.tsv")
x_train = []
y_train = []
x_test = []
test_ids = []

for line in training:
    line = line.split("\t")
    x_train.append(line[2])
    y_train.append(line[1])

for line in dev:
    line = line.split("\t")
    x_train.append(line[2])
    y_train.append(line[1])

for line in testing:
    line = line.split("\t")
    x_test.append(line[2])
    test_ids.append(line[0])

# CREATE A TRAINING DATA VECTOR
vectorizer_t = TfidfVectorizer(stop_words="english")
v_train = vectorizer_t.fit_transform(x_train)


# TOKENIZE THE TEST DATA
x_test_t = []
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
for str in x_test:
    tokens = word_tokenize(str)
    tokens = [w for w in tokens if not w in stop_words]
    x_test_t.append(tokens)

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


#TRAIN CLASSIFIERS

mnb = MultinomialNB()
lr = LogisticRegression()
brf = BaggingClassifier(RandomForestClassifier(n_estimators=10), max_samples=0.8, max_features=0.2)

mnb.fit(xk_train, y_train)
lr.fit(xk_train, y_train)
brf.fit(xk_train, y_train)


# GENERATE PREDICTIONS
output = open("MNB_predictions.txt", "w")
output.write("Id,Class\n")
i = 0;
predictions = []
for line in x_test_t:
    instance = []
    for f in best_features:
        instance.append(line.count(f))
    instance = pd.DataFrame(instance)
    instance = instance.values
    instance = instance.reshape(1, -1)
    output.write(test_ids[i])
    output.write("," + mnb.predict(instance)[0] + "\n")
    i += 1
output.close()

output = open("LR_predictions.txt", "w")
output.write("Id,Class\n")
i = 0;
predictions = []
for line in x_test_t:
    instance = []
    for f in best_features:
        instance.append(line.count(f))
    instance = pd.DataFrame(instance)
    instance = instance.values
    instance = instance.reshape(1, -1)
    output.write(test_ids[i])
    output.write("," + lr.predict(instance)[0] + "\n")
    i += 1
output.close()

output = open("BRF_predictions.txt", "w")
output.write("Id,Class\n")
i = 0;
predictions = []
for line in x_test_t:
    instance = []
    for f in best_features:
        instance.append(line.count(f))
    instance = pd.DataFrame(instance)
    instance = instance.values
    instance = instance.reshape(1, -1)
    output.write(test_ids[i])
    output.write("," + brf.predict(instance)[0] + "\n")
    i += 1
output.close()
print("blah")