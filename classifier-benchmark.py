import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD
import time

EN_TOKENIZED_DATA = '../data/tokens_by_lang/surveys_en.csv'
DE_TOKENIZED_DATA = '../data/tokens_by_lang/surveys_de.csv'

VECTORIZERS = [
    {"name": "TfidVectorizer", "vectorizer": TfidfVectorizer(max_features=500, ngram_range=(2,3))},
    # {"name": "HashingVectorizer", "vectorizer": HashingVectorizer(ngram_range=(2,3))},
    {"name": "CountVectorizer", "vectorizer": CountVectorizer(max_features=500, ngram_range=(2,3))}
]

def x_to_self(x):
    return x

def x_to_array(x):
    return x.toarray()

CLASSIFIERS = [
    # {"adjustX": x_to_self, "name": "KNeighborsClassifier", "classifier": KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree')},
    {"adjustX": x_to_self, "name": "SVC", "classifier": SVC(kernel="linear", C=0.025, probability=True)},
    {"adjustX": x_to_self, "name": "DecisionTreeClassifier", "classifier": DecisionTreeClassifier(max_depth=5)},
    {"adjustX": x_to_self, "name": "RandomForestClassifier", "classifier": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)},
    {"adjustX": x_to_self, "name": "MLPClassifier", "classifier": MLPClassifier(alpha=1, solver='sgd', max_iter=1000)},
    {"adjustX": x_to_self, "name": "AdaBoostClassifier", "classifier": AdaBoostClassifier()},
    {"adjustX": x_to_self, "name": "MultinomialNB", "classifier": MultinomialNB()}

    # takes too long {"adjustX": x_to_self, "name": "SVC", "classifier": SVC(gamma=2, C=1, probability=True)},
    # flatten X needed (?) {"adjustX": x_to_array, "name": "GaussianProcessClassifier", "classifier": GaussianProcessClassifier(1.0 * RBF(1.0))},
    # flatten X needed (?) {"adjustX": x_to_array, "name": "GaussianNB", "classifier": GaussianNB()},
    # flatten X needed (?) {"adjustX": x_to_array, "name": "QuadraticDiscriminantAnalysis", "classifier": QuadraticDiscriminantAnalysis()},
]

def load_dataset(path):
    df = pd.read_csv(path, sep=',', error_bad_lines=False)
    df = df[['tokens', 'status', 'rejection_reason', 'language', 'detected_language']]
    return df

def filter_status(df):
    df = df[df['status'].isin(['ACCEPTED', 'REJECTED'])]
    # df = df[df['rejection_reason'].notna()]
    return df

def filter_language(df):
    df = df[df['language'] == df['detected_language']]
    return df

def vectorize_data(inputCsvPath, vectorizer):
    df = load_dataset(inputCsvPath)
    df = filter_status(df)
    df = filter_language(df)

    X = df['tokens']
    y = df['status']

    y = y.map({'ACCEPTED': 1, 'REJECTED': 0})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    #vectorizer = TfidfVectorizer(max_features=500, ngram_range=(2,3))

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    return X_train, X_test, y_train, y_test


def create_classifier(X_train, y_train, data_file):
    classifier = MultinomialNB()
    print(f'training classifier for {data_file}')
    classifier.fit(X_train, y_train)
    return classifier


def calculate_accuracy(y_predicted, y_test):
    return accuracy_score(y_test, y_predicted)

def map_probabilities_by_treshold(probs, treshold):
    return list(map(lambda prob: 1 if prob[0] < treshold else 0, probs))

def make_magic_happen(data_file):
    results = []

    for vectorizer in VECTORIZERS:
        X_train, X_test, y_train, y_test = vectorize_data(data_file, vectorizer["vectorizer"])

        for classifier in CLASSIFIERS:
            print("Running " + vectorizer["name"] + " with " + classifier["name"])
            # X_train = classifier["adjustX"](X_train)
            # X_test = classifier["adjustX"](X_test)
            start = time.time()
            classifier["classifier"].fit(X_train, y_train)

            y_predicted = classifier["classifier"].predict_proba(X_test)
            y_predicted2 = map_probabilities_by_treshold(y_predicted, 0.5) # closer to 0 -> less false-positives
            # print(y_predicted)

            accuracy = calculate_accuracy(y_predicted2, y_test)
            matrix = confusion_matrix(y_test, y_predicted2)
            end = time.time()
            res_row = {
                "file": data_file,
                "vectorizer": vectorizer["name"],
                "classifier": classifier["name"],
                "accuracy": accuracy,
                "TN": matrix[0][0],
                "FP": matrix[0][1],
                "FN": matrix[1][0],
                "TP": matrix[1][1],
                "time": time.strftime('%H:%M:%S', time.gmtime(end-start))
            }
            print(res_row)

            results.append(res_row)
            # print(data_file, ': ', accuracy)
            # print('confusion matrix without normalization:\n', matrix)
            # print('confusion matrix with normalization:\n', matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis])
    resultDataFrame = pd.DataFrame(results, columns=['file', 'vectorizer', 'classifier', 'accuracy', 'TN', 'FP', 'FN', 'TP', 'time'])
    resultDataFrame.to_csv("../data2/results.csv")

if __name__ == '__main__':
    make_magic_happen(EN_TOKENIZED_DATA)
    # make_magic_happen(DE_TOKENIZED_DATA)
