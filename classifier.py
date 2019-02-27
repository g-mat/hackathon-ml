import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

EN_TOKENIZED_DATA = '../data/tokens_by_lang/surveys_en.csv'
DE_TOKENIZED_DATA = '../data/tokens_by_lang/surveys_de.csv'


def load_dataset(path):
    df = pd.read_csv(path, sep=',', error_bad_lines=False)
    df = df[['tokens', 'status', 'language', 'detected_language']]
    return df


def filter_status(df):
    df = df[df['status'].isin(['ACCEPTED', 'REJECTED'])]
    return df

def filter_language(df):
    df = df[df['language'] == df['detected_language']]
    return df

def vectorize_data(inputCsvPath):
    df = load_dataset(inputCsvPath)
    df = filter_status(df)
    df = filter_language(df)

    X = df['tokens']
    y = df['status']

    y = y.map({'ACCEPTED': 1, 'REJECTED': 0})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    vectorizer = TfidfVectorizer(max_features=2000)

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


def make_magic_happen(data_file):
    X_train, X_test, y_train, y_test = vectorize_data(data_file)
    classifier = create_classifier(X_train, y_train, data_file)

    y_predicted = classifier.predict(X_test)
    accuracy = calculate_accuracy(y_predicted, y_test)
    print(confusion_matrix(y_test, y_predicted))
    print(data_file, ': ', accuracy)


if __name__ == '__main__':
    make_magic_happen(EN_TOKENIZED_DATA)
    make_magic_happen(DE_TOKENIZED_DATA)
