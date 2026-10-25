from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import json

from preprocessing import clean_tweet


app = Flask(__name__)
CORS(app)


# Load trained model
model = joblib.load(
    "models/sentiment_model.pkl"
)

vectorizer = joblib.load(
    "models/vectorizer.pkl"
)


# Load sentiment statistics
with open("models/sentiment_stats.json", "r") as file:
    sentiment_stats = json.load(file)


# Number of predictions made during current session
prediction_count = 0


@app.route('/')
def home():

    return render_template(
        'index.html',
        total_tweets=sum(sentiment_stats.values()),
        positive=sentiment_stats.get("Positive", 0),
        negative=sentiment_stats.get("Negative", 0),
        neutral=sentiment_stats.get("Neutral", 0)
    )


@app.route('/predict', methods=['POST'])
def predict():

    global prediction_count

    data = request.get_json()

    tweet = data.get('tweet', '')

    if not tweet.strip():

        return jsonify({
            "error": "Please enter a tweet."
        }), 400


    # Clean tweet
    cleaned = clean_tweet(tweet)


    # Convert text to TF-IDF
    vect_text = vectorizer.transform([cleaned])


    # Prediction
    prediction = model.predict(vect_text)[0]


    # Probability
    probabilities = model.predict_proba(vect_text)[0]

    confidence = round(
        max(probabilities) * 100,
        2
    )


    # Count prediction
    prediction_count += 1


    # Detailed explanation
    if prediction == "Positive":

        explanation = (
            "The tweet contains words or expressions "
            "that indicate a positive attitude toward "
            "COVID-19 vaccination."
        )

    elif prediction == "Negative":

        explanation = (
            "The tweet contains words or expressions "
            "that indicate dissatisfaction, concern, "
            "or a negative attitude toward vaccination."
        )

    else:

        explanation = (
            "The tweet mainly provides information or "
            "a neutral statement without a strong "
            "positive or negative sentiment."
        )


    return jsonify({

        "tweet": tweet,

        "cleaned_text": cleaned,

        "sentiment": prediction,

        "confidence": f"{confidence}%",

        "explanation": explanation,

        "prediction_count": prediction_count
    })


@app.route('/stats')
def stats():

    return jsonify({

        "total_tweets": sum(sentiment_stats.values()),

        "positive": sentiment_stats.get(
            "Positive", 0
        ),

        "negative": sentiment_stats.get(
            "Negative", 0
        ),

        "neutral": sentiment_stats.get(
            "Neutral", 0
        ),

        "analyzed_now": prediction_count
    })


if __name__ == '__main__':

    app.run(
        debug=True,
        port=5000
    )