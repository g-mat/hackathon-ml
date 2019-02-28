import itertools
import os

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

nltk.download('stopwords')

color = sns.color_palette()

LANGUAGES = ['en', 'de']
STOP_WORD_DICT = {'en': 'english', 'de': 'german'}
BRANDS = ['stepstone', 'tjg', 'ycg', 'irishjobs.ie']
VECTORIZERS = {}
DIRECTORY = ''
TOKENIZER = RegexpTokenizer(r'\w+')
CLASS_COLUMN = 'status'
# CLASS_COLUMN = 'rejection_reason'


def load_dataset(path):
    df = pd.read_csv(path, error_bad_lines=False)
    df = filter_status(df)
    df = filter_language(df)
    df = df[['text', 'status', 'rejection_reason', 'detected_language']]
    return df


def filter_status(df):
    df = df[df['status'].isin(['ACCEPTED', 'REJECTED'])]
    # df = df[df['rejection_reason'].notna()]
    return df


def filter_language(df):
    df = df[df['language'] == df['detected_language']]
    return df


def vectorize_data(df, brand, lang):
    X = df['text']
    df['rejection_reason'] = detailed_status(df)
    y = df[CLASS_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    vectorizer = VECTORIZERS[lang]

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    plt.figure(figsize=(12, 8))
    sns.countplot(x=CLASS_COLUMN, data=df)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Moderation status', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.title(f'Moderation status distribution for {brand} ({lang})', fontsize=15)
    plt.savefig(f'{DIRECTORY}/moderation_distribution.png')

    return X_train, X_test, y_train, y_test


def detailed_status(df):
    return np.where(df['rejection_reason'].isnull(), df['status'], df['rejection_reason'])


def create_classifier(X_train, y_train):
    svm_clf = svm.SVC(C=1, cache_size=200, class_weight='balanced', coef0=0.0,
                      decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
                      max_iter=-1, probability=False, random_state=None, shrinking=True,
                      tol=0.001, verbose=False)
    return svm_clf.fit(X=X_train, y=y_train)


def generate_word_count_plot(df, brand, lang):
    vectorizer = VECTORIZERS[lang]
    class_Xtr = vectorizer.transform(df['text'])
    class_y = df[CLASS_COLUMN]
    class_features = vectorizer.get_feature_names()
    class_top_dfs = top_feats_by_class(class_Xtr, class_y, class_features)
    plot_tfidf_classfeats_h(class_top_dfs, brand, lang, 2)


def make_magic_happen(brand, lang):
    print(f'magic: {brand} {lang}')

    global DIRECTORY
    DIRECTORY = f'data/{brand}_{lang}'
    file = f'{DIRECTORY}/surveys.csv'
    if not os.path.exists(file):
        return

    df = load_dataset(file)

    X_train, X_test, y_train, y_test = vectorize_data(df, brand, lang)
    generate_word_count_plot(df, brand, lang)
    classifier = create_classifier(X_train, y_train)

    y_predicted = classifier.predict(X_test)
    compute_confusion_matrix(y_test, y_predicted, df[CLASS_COLUMN].unique())


def prepare_vectorizer(lang):
    stop_words = set(stopwords.words(STOP_WORD_DICT[lang]))
    VECTORIZERS[lang] = TfidfVectorizer(
        min_df=5, max_features=3000, strip_accents='unicode', lowercase=True,
        analyzer='word', ngram_range=(1, 3), use_idf=True,
        smooth_idf=True, sublinear_tf=True, tokenizer=TOKENIZER.tokenize, stop_words=stop_words
    )


def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['text', CLASS_COLUMN]
    return df


def top_feats_in_doc(Xtr, features, row_id, top_n=20):
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)


def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=10):
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=20):
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y == label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs


def plot_tfidf_classfeats_h(dfs, brand, lang, num_class=9):
    fig = plt.figure(figsize=(12, 100), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        # z = int(str(int(i/3)+1) + str((i%3)+1))
        ax = fig.add_subplot(num_class, 1, i + 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=16)
        ax.set_ylabel("Word", labelpad=16, fontsize=16)
        ax.set_title("Status = " + str(df.label), fontsize=25)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax.barh(x, df[CLASS_COLUMN], align='center')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1] + 1])
        yticks = ax.set_yticklabels(df.text)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.savefig(f'{DIRECTORY}/word_distribution.png')


def compute_confusion_matrix(y_test, y_predicted, class_names):
    accuracy = accuracy_score(y_test, y_predicted)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_predicted)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title=f'Confusion matrix, without normalization (accuracy {accuracy})')
    plt.savefig(f'{DIRECTORY}/confusion_matrix.png')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title=f'Normalized confusion matrix (accuracy {accuracy})')

    plt.savefig(f'{DIRECTORY}/confusion_matrix_normalized.png')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


if __name__ == '__main__':
    for lang in LANGUAGES:
        prepare_vectorizer(lang)
        for brand in BRANDS:
            make_magic_happen(brand, lang)
