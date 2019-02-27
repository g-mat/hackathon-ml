import pandas as pd
from langdetect import detect as _detect

DATA_FILE_PATH = '../data/surveys_escaped.csv'

COLUMN_NAMES = [
    'survey_uuid',
    'jobsite',
    'language',
    'campaing_id',
    'areas_of_improvement',
    'pros',
    'responsibilities',
    'review_title',
    'status',
    'rejection_reason',
    'survey_start_time',
    'moderation_end_time'
]


def detect(text):
    try:
        return _detect(text)
    except:
        print(text)
        return 'NOPE'


def get_dataset(path):
    df = pd.read_csv(path, sep='\t', error_bad_lines=False, header=None)
    df.columns = COLUMN_NAMES
    return df


def add_detected_language(df):
    df['text'] = df['areas_of_improvement'].map(str) + \
                 df['pros'].map(str) + \
                 df['responsibilities'].map(str) + \
                 df['review_title'].map(str)
    df['detected_language'] = df['text'].apply(detect)
    print(df.detected_language.value_counts())
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
