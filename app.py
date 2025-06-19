from flask import Flask, render_template, request
import joblib
from spam_classifier import transform_text

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('spam_model.pkl')
cv = joblib.load('vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        email_text = request.form['email_text']
        transformed = transform_text(email_text)
        vect = cv.transform([transformed]).toarray()
        pred = model.predict(vect)[0]
        prediction = 'Spam' if pred == 1 else 'Ham'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True) 