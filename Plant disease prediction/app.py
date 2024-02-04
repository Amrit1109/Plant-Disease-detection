from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('Indian_Medicinal_Plant_Diseases.csv')

symptoms = data['Symptoms'].tolist()
diseases = data['Disease'].tolist()

X_train, X_test, y_train, y_test = train_test_split(symptoms, diseases, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train_tfidf, y_train)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    if request.method == 'POST':
        user_input = request.form['symptoms']

        user_input_tfidf = tfidf_vectorizer.transform([user_input])

        prediction = random_forest_classifier.predict(user_input_tfidf)

        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
