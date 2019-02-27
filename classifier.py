import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_FILE = 'D:\\Dev\\machine_learning\\data\\tokens_by_lang\\surveys_en.csv'

INPUT_COLUMNS = [
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
    'moderation_end_time',
    'text',
    'detected_language',
    'tokens'
]

def loadDataset(path):
    df = pd.read_csv(path, sep=',', error_bad_lines=False, header=None)
    df.columns = INPUT_COLUMNS
    return df

def classify(inputCsvPath):
    dataSet = loadDataset(inputCsvPath)

    trainReviews = dataSet.loc[0:999, 'tokens'].values
    # trainStatuses = dataSet.loc[:24999, 'status'].values
    testReviews = dataSet.loc[1000:1999, 'tokens'].values
    # testStatuses = dataSet.loc[25000:, 'status'].values

    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(trainReviews)
    test_vectors = vectorizer.transform(testReviews)
    print(train_vectors.shape, test_vectors.shape)

if __name__ == '__main__':
    classify(DATA_FILE)