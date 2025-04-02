from flask import Flask, request, jsonify
import os
import warnings
import logging
import firebase_admin
from firebase_admin import credentials, firestore
import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer 

# Suppress unnecessary warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

app = Flask(__name__)

# Initialize Firebase (you'll need to replace this with your own credentials)
try:
    # Use environment variable for Firebase credentials or a local file
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

# Download and set up NLTK VADER for sentiment analysis (much lighter than transformers)
try:
    nltk.download('vader_lexicon', quiet=True)
    sentiment_analyzer = SentimentIntensityAnalyzer()
except Exception as e:
    print(f"Error setting up NLTK: {e}")

def get_sentiment_score(comment):
    """Calculate sentiment score from 1-5 based on the comment using VADER"""
    if not comment or not isinstance(comment, str):
        return 3.0  # Default neutral score for empty or invalid input
    
    # Check for neutral words
    neutral_words = [
        "okay", "fine", "average", "mediocre", "decent", "alright", 
        "good enough", "not bad", "satisfactory", "acceptable"
    ]
    
    if any(word in comment.lower() for word in neutral_words):
        return 3.0
    
    # Get sentiment scores from VADER
    sentiment_scores = sentiment_analyzer.polarity_scores(comment)
    compound_score = sentiment_scores['compound']  # Between -1 and 1
    
    # Convert compound score to 1-5 scale
    # -1 → 5, 0 → 3, 1 → 1
    return 3 - (compound_score * 2)

def updateDealSentiment(bankID, cityID, dealID, userID, comment):
    """
    Update deal sentiment in Firebase with the new comment
    
    Args:
        bankID (str): ID of the bank
        cityID (str): ID of the city
        dealID (str): ID of the deal
        userID (str): ID of the user
        comment (str): User's comment
    
    Returns:
        dict: Status of the update operation
    """
    if not firebase_initialized:
        return {
            'status': 'error',
            'message': 'Firebase is not initialized'
        }
    
    try:
        # Calculate sentiment score for the comment
        sentiment_score = get_sentiment_score(comment)
        
        # Reference to the deal document
        deal_ref = db.collection('Banks').document(bankID).collection('Cities').document(cityID).collection('Deals').document(dealID)
        
        # Get the current deal data
        deal_doc = deal_ref.get()
        
        if deal_doc.exists:
            deal_data = deal_doc.to_dict()
            
            # Initialize comments array if it doesn't exist
            if 'comments' not in deal_data:
                deal_data['comments'] = []
            
            # Add new comment with sentiment score
            new_comment = {
                'userID': userID,
                'comment': comment,
                'sentimentScore': sentiment_score
            }
            deal_data['comments'].append(new_comment)
            
            # Calculate average sentiment
            total_score = sum(comment_item['sentimentScore'] for comment_item in deal_data['comments'])
            avg_sentiment = total_score / len(deal_data['comments'])
            
            # Update the document
            deal_ref.update({
                'comments': deal_data['comments'],
                'avgSentiment': round(avg_sentiment, 2)
            })
            
            return {
                'status': 'success',
                'message': 'Deal sentiment updated successfully',
                'newSentimentScore': sentiment_score,
                'newAvgSentiment': round(avg_sentiment, 2)
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
    """API endpoint to analyze sentiment of a comment"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        comment = data.get("comment", "")
        if not comment:
            return jsonify({"error": "No comment provided"}), 400

        sentiment_score = get_sentiment_score(comment)
        
        # If the API call includes deal info, update Firebase
        if all(key in data for key in ["bankID", "cityID", "dealID", "userID"]):
            update_result = updateDealSentiment(
                data["bankID"], 
                data["cityID"], 
                data["dealID"], 
                data["userID"], 
                comment
            )
            return jsonify({
                "sentiment_score": sentiment_score,
                "update_result": update_result
            })
        
        # Otherwise just return the sentiment score
        return jsonify({"sentiment_score": sentiment_score})
    
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

# Add this route to test if the app is running
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "running",
        "message": "Sentiment Analysis API is working. Use /analyze endpoint with POST requests."
    })

# Use this if you're running with gunicorn
port = int(os.environ.get("PORT", 8080))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)