import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

from preprocessing import clean_tweet


# 1. Load dataset from Google Sheets
url = "https://docs.google.com/spreadsheets/d/1-_aprlN7ov57q3GBFiWjhQhFSjHpHRnS/export?format=xlsx"

df = pd.read_excel(url)

df = df[['text']].dropna()

print("Total tweets loaded:", len(df))

# Show only a few tweets instead of the entire dataset
print("\nSample tweets:")
print(df.head(5))


# 2. Clean tweets
df['clean_text'] = df['text'].apply(clean_tweet)


# 3. Create sentiment labels
def get_sentiment(text):

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.05:
        return "Positive"

    elif polarity < -0.05:
        return "Negative"

    else:
        return "Neutral"


df['sentiment'] = df['text'].apply(get_sentiment)


# 4. Save sentiment statistics
sentiment_counts = df['sentiment'].value_counts().to_dict()

with open("models/sentiment_stats.json", "w") as file:
    json.dump(sentiment_counts, file)


# 5. TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']


# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 7. Train Multinomial Naive Bayes
model = MultinomialNB()

model.fit(X_train, y_train)


# 8. Prediction
y_pred = model.predict(X_test)


# 9. Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("COVID-19 VACCINE SENTIMENT")
print("==============================")

print("Total tweets:", len(df))
print("Training tweets:", X_train.shape[0])
print("Testing tweets:", X_test.shape[0])

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 10. Save model
joblib.dump(
    model,
    "models/sentiment_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/vectorizer.pkl"
)

print("\nModel and vectorizer saved successfully.")