import pandas as pd
from langdetect import detect as _detect

DATA_FILE_PATH = ''


def detect(text):
    try:
        return _detect(text)
    except:
        return 'NOPE'


def get_dataset(path):
    df = pd.read_csv(path, sep='\t', error_bad_lines=False, header=None)
    df.columns = ['survey_id', 'jobsite', 'lang', 'source', 'text',
                  'pros', 'cons', 'company_name', 'accepted', 'something',
                  'date1', 'date2']
    df = df.dropna(subset=['text'])
    return df


def add_detected_language(df):
    df['detected_language'] = df['text'].apply(detect)
    return df


def divide_per_language(df):
    unique_langs = df['detected_language'].unique()
    for lang in unique_langs:
        lang_df = df[df['detected_language'] == lang]
        lang_df.to_csv(f'surveys_{lang}.csv', index=False)


if __name__ == '__main__':
    df = get_dataset(DATA_FILE_PATH)
    df = add_detected_language(df)
    divide_per_language(df)
