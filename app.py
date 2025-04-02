from flask import Flask, request, jsonify
from transformers import pipeline
import os
import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

app = Flask(__name__)

# Load the model
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def get_sentiment_score(comment):
    neutral_words = [
        "okay", "fine", "average", "mediocre", "decent", "alright", 
        "good enough", "not bad", "satisfactory", "acceptable"
    ]
    
    if any(word in comment.lower() for word in neutral_words):
        return 3.0  
    
    result = sentiment_pipeline(comment)[0]
    sentiment_label = result['label']
    score = result['score']
    
    if sentiment_label == 'POSITIVE':
        return 1.0 + (score * 4.0)  
    else:
        return 5.0 - (score * 4.0) 

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get("comment", "The service was worst")
    
    if not text:
        return jsonify({"error": "No comment provided"}), 400

    sentiment_score = get_sentiment_score(text)
    print(f"Sentiment score for '{text}': {sentiment_score}")
    return jsonify({"sentiment_score": sentiment_score})

if __name__ == "__main__":  
    app.run(debug=True)
