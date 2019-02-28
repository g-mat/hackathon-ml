from flask import Flask, jsonify, request
from langdetect import detect
from textblob import TextBlob
from profanity import profanity as prof
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/get_language', methods=['POST'])
def language():
    text = request.get_json()['text']
    return jsonify({'lang': detect(text)})


@app.route('/get_sentiment', methods=['POST'])
def sentiment():
    blob = TextBlob(request.get_json()['text'])
    return jsonify({
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    })


@app.route('/is_profanity', methods=['POST'])
def profanity():
    text = request.get_json()['text']
    return jsonify({'is_profanity': prof.contains_profanity(text)})


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=5666)
