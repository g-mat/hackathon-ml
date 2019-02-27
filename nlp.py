import json_lines
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

en_stop_words = set(stopwords.words('english'))
de_stop_words = set(stopwords.words('german'))

with open('surveys.jsonl', 'rb') as f:
    for item in json_lines.reader(f):
        if item['language'] == 'en':
            phrase = ' '.join(filter(None, [item['pros'], item['areas_of_improvement'], item['responsibilities'], item['review_title']])).lower()
            tokens = word_tokenize(phrase, 'english')
            filtered_tokens = [token for token in tokens if token not in en_stop_words]
            print(filtered_tokens)
