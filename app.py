from flask import Flask, request, jsonify
import os
import warnings
import logging
import firebase_admin
from firebase_admin import credentials, firestore
import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime
from flask_cors import CORS 


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app) 

try:
    if os.environ.get('FIREBASE_CREDENTIALS'):
        import json
        cred_dict = json.loads(os.environ.get('FIREBASE_CREDENTIALS'))
        cred = credentials.Certificate(cred_dict)
    else:
        cred = credentials.Certificate("key.json")
    
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    firebase_initialized = True
except Exception as e:
    print(f"Firebase initialization error: {e}")
    firebase_initialized = False

try:
    nltk.download('vader_lexicon', quiet=True)
    sentiment_analyzer = SentimentIntensityAnalyzer()
except Exception as e:
    print(f"Error setting up NLTK: {e}")

def get_sentiment_score(comment):
    if not comment or not isinstance(comment, str):
        return 3.0  

    sentiment_scores = sentiment_analyzer.polarity_scores(comment)
    compound_score = sentiment_scores['compound']  
    return round(1 + (compound_score + 1) * 2, 2) 

def updateDealSentiment(bankID, cityID, dealID, userID, userName, comment):
    if not firebase_initialized:
        return {
            'status': 'error',
            'message': 'Firebase is not initialized'
        }
    
    try:
        sentiment_score = get_sentiment_score(comment)
        current_date = datetime.datetime.now().strftime("%B %d, %Y %H:%M:%S")
        deal_ref = db.collection('Banks').document(bankID).collection('Cities').document(cityID).collection('Deals').document(dealID)
        deal_doc = deal_ref.get()
        
        if deal_doc.exists:
            deal_data = deal_doc.to_dict()
            
            if 'comments' not in deal_data:
                deal_data['comments'] = []
            
            new_comment = {
                'userID': userID,
                'userName': userName,
                'comment': comment,
                'sentimentScore': sentiment_score,
                'date': current_date
            }
            deal_data['comments'].append(new_comment)
            
            total_score = sum(comment_item['sentimentScore'] for comment_item in deal_data['comments'])
            avg_sentiment = total_score / len(deal_data['comments'])
            
            deal_ref.update({
                'comments': deal_data['comments'],
                'avgSentiment': round(avg_sentiment, 2)
            })
            
            return {
                'status': 'success',
                'message': 'Deal sentiment updated successfully',
                'newSentimentScore': sentiment_score,
                'newAvgSentiment': round(avg_sentiment, 2),
                'date': current_date
            }
        else:
            return {
                'status': 'error',
                'message': f'Deal with ID {dealID} not found'
            }
    except Exception as e:
        logging.error(f"Error updating deal sentiment: {str(e)}")
        return {
            'status': 'error',
            'message': f'Failed to update deal sentiment: {str(e)}'
        }

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        comment = data.get("comment", "")
        if not comment:
            return jsonify({"error": "No comment provided"}), 400

        sentiment_score = get_sentiment_score(comment)
        
        if all(key in data for key in ["bankID", "cityID", "dealID", "userID", "userName"]):
            update_result = updateDealSentiment(
                data["bankID"], 
                data["cityID"], 
                data["dealID"], 
                data["userID"], 
                data["userName"],
                comment
            )
            return jsonify({"sentiment_score": sentiment_score, "update_result": update_result})
        
        return jsonify({"sentiment_score": sentiment_score})
    
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "running", "message": "Sentiment Analysis API is working. Use /analyze endpoint with POST requests."})

port = int(os.environ.get("PORT", 8080))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)
