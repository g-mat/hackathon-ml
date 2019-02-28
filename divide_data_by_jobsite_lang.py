import pandas as pd
import os
from langdetect import detect as _detect

DATA_FILE_PATH = 'surveys_escaped.csv'
BRAND_DICT = {
    "ycg": ["hotelcareer.ch", "hotelcareer.at", "touristikcareer.de", "hotelcareer.fr", "hotelcareer.de", "gastrojobs.ch", "gastrojobs.de", "gastrojobs.at", "hotelcareer.com"],
    "tjg": ["milkround.com", "careerstructure.com", "totaljobs.com", "caterer.com", "retailchoice.com", "cwjobs.co.uk"],
    "irishjobs.ie": ["irishjobs.ie"],
    "stepstone": ["stepstone.at", "stepstone.be", "stepstone.de"]
}
JOBSITE_DICT = {}
for brand, jobsites in BRAND_DICT.items():
    for jobsite in jobsites:
        JOBSITE_DICT[jobsite] = brand

LANGUAGES = ['en', 'de']


def detect(text):
    try:
        return _detect(text)
    except:
        return 'NOPE'


def get_dataset(path):
    df = pd.read_csv(path, error_bad_lines=False, escapechar='\\')
    df = df[['jobsite', 'areas_of_improvement', 'pros', 'responsibilities', 'review_title', 'status', 'rejection_reason', 'language']]
    df['jobsite'] = df['jobsite'].map(JOBSITE_DICT)
    return df


def add_detected_language(df):
    cols = ['areas_of_improvement', 'pros', 'responsibilities', 'review_title']
    df['text'] = df[cols].apply(lambda row: ' '.join(row.dropna()), axis=1)
    df['detected_language'] = df['text'].apply(detect)
    return df


def divide_per_brand_and_language(df):
    for lang in LANGUAGES:
        lang_df = df[df['detected_language'] == lang]
        for brand in BRAND_DICT.keys():
            lang_brand_df = lang_df[lang_df['jobsite'] == brand]
            directory = f'data/{brand}_{lang}'
            if not os.path.exists(f'{directory}'):
                os.makedirs(f'{directory}')
            lang_brand_df.to_csv(f'{directory}/surveys.csv', index=False)


if __name__ == '__main__':
    df = get_dataset(DATA_FILE_PATH)
    df = add_detected_language(df)
    divide_per_brand_and_language(df)
