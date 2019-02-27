import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

EN_SURVEY_DATA_FILE_PATH = 'surveys_by_lang/surveys_en.csv'
DE_SURVEY_DATA_FILE_PATH = 'surveys_by_lang/surveys_de.csv'
EN_TOKEN_DATA_FILE_PATH = 'tokens_by_lang/surveys_en.csv'
DE_TOKEN_DATA_FILE_PATH = 'tokens_by_lang/surveys_de.csv'

TOKENIZER = RegexpTokenizer(r'\w+')
STEMMER = PorterStemmer()


def get_dataset(path):
    df = pd.read_csv(path, sep=',', error_bad_lines=False)
    df = df.dropna(subset=['text'])
    return df


def create_tokens(df, stop_words):
    df['tokens'] = df['text'].apply(lambda text: TOKENIZER.tokenize(text))
    df['tokens'] = df['tokens'].apply(lambda tokens: [token for token in tokens if token not in stop_words])
    df['tokens'] = df['tokens'].apply(lambda tokens: [STEMMER.stem(token) for token in tokens])
    return df


def process(survey_file, tokens_file, language):
    stop_words = set(stopwords.words(language))
    df = get_dataset(survey_file)
    df = create_tokens(df, stop_words)
    df.to_csv(tokens_file, index=False)


if __name__ == '__main__':
    process(EN_SURVEY_DATA_FILE_PATH, EN_TOKEN_DATA_FILE_PATH, 'english')
    process(DE_SURVEY_DATA_FILE_PATH, DE_TOKEN_DATA_FILE_PATH, 'german')
