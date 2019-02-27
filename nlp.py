import pandas as pd
import nltk

from langdetect import detect as _detect
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

EN_SURVEY_DATA_FILE_PATH = 'surveys_by_lang/surveys_en.csv'
DE_SURVEY_DATA_FILE_PATH = 'surveys_by_lang/surveys_en.csv'
EN_TOKEN_DATA_FILE_PATH = 'tokens_by_lang/surveys_en.csv'
DE_TOKEN_DATA_FILE_PATH = 'tokens_by_lang/surveys_en.csv'


def get_dataset(path):
    df = pd.read_csv(path, sep=',', error_bad_lines=False)
    df = df.dropna(subset=['text'])
    return df


def create_tokens(df, stop_words, language):
    df['tokens'] = df['text'].apply(lambda text: word_tokenize(text, language))
    df['tokens'] = df['tokens'].apply(lambda tokens: [token for token in tokens if token not in stop_words])
    return df


def process(survey_file, tokens_file, language):
    stop_words = set(stopwords.words(language))
    df = get_dataset(survey_file)
    df = create_tokens(df, stop_words, language)
    df.to_csv(tokens_file, index=False)


if __name__ == '__main__':
    process(EN_SURVEY_DATA_FILE_PATH, EN_TOKEN_DATA_FILE_PATH, 'english')
    process(DE_SURVEY_DATA_FILE_PATH, DE_TOKEN_DATA_FILE_PATH, 'german')
