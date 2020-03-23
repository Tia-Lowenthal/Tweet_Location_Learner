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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np


# READ DATA INTO LISTS
training = open("processed_train_raw.tsv")
dev = open("processed_dev_raw.tsv")
x_train = []
y_train = []
x_dev = []
y_dev = []

for line in training:
    line = line.split("\t")
    x_train.append(line[2])
    y_train.append(line[1])

for line in dev:
    line = line.split("\t")
    x_train.append(line[2])
    y_train.append(line[1])

# CREATE A TRAINING DATA VECTOR
vectorizer_t = TfidfVectorizer(stop_words="english")
v_train = vectorizer_t.fit_transform(x_train)


# TOKENIZE THE TEST DATA
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


dev_data = []
for line in x_dev_t:
    instance = []
    for f in best_features:
        instance.append(line.count(f))
    dev_data.append(instance)

dev_data = pd.DataFrame(dev_data)
dev_data = dev_data.values


y_dev_i = []
for i in range(len(y_dev)):
    if y_dev[i] == 'Melbourne':
        y_dev_i.append(0)
    elif y_dev[i] == 'Sydney':
        y_dev_i.append(1)
    elif y_dev[i] == 'Brisbane':
        y_dev_i.append(2)
    else:
        y_dev_i.append(3)

class_names = np.array(['Melbourne', 'Sydney', 'Brisbane', 'Perth'])


classifier = MultinomialNB(())
y_pred = classifier.fit(xk_train, y_train).predict(dev_data)

y_pred_i = []
for i in range(len(y_pred)):
    if y_pred[i] == 'Melbourne':
        y_pred_i.append(0)
    elif y_pred[i] == 'Sydney':
        y_pred_i.append(1)
    elif y_pred[i] == 'Brisbane':
        y_pred_i.append(2)
    else:
        y_pred_i.append(3)


# class definition used from:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_dev_i, y_pred_i, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_dev_i, y_pred_i, classes=class_names, normalize=True,
                      title='MNB Normalized Confusion Matrix')

plt.show()