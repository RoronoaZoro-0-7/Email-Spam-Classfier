import nltk
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
import joblib

nltk.download('punkt')
nltk.download('stopwords')

# Load data
DATA_PATH = 'spam.csv'
df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')
df = df.rename(columns={'v1': 'target', 'v2': 'text'})

# Clean data
df = df.drop_duplicates(keep='first')
df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    cleaned = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    final = [word for word in cleaned if word not in stop_words]
    ps = PorterStemmer()
    final = [ps.stem(word) for word in final]
    return " ".join(final)

df['transformed_text'] = df['text'].apply(transform_text)

# Vectorization
cv = CountVectorizer()
X = cv.fit_transform(df['transformed_text']).toarray()
Y = df['target'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer for reuse
joblib.dump(model, 'spam_model.pkl')
joblib.dump(cv, 'vectorizer.pkl')

def predict_spam(text):
    """Predict if the given text is spam (1) or ham (0)."""
    loaded_model = joblib.load('spam_model.pkl')
    loaded_cv = joblib.load('vectorizer.pkl')
    transformed = transform_text(text)
    vect = loaded_cv.transform([transformed]).toarray()
    pred = loaded_model.predict(vect)[0]
    return pred

if __name__ == '__main__':
    # Simple CLI for testing
    test_text = input('Enter a message to classify: ')
    result = predict_spam(test_text)
    print('Spam' if result == 1 else 'Ham') 