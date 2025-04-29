from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Load models (adjust the paths properly)
clean_model_path = os.path.join('..','src', 'models', 'best_model_logistic_regression.pkl')
noisy_model_path = os.path.join('..','src', 'models', 'logistic_regression_20250429_210939.pkl')

# Load the models
with open(clean_model_path, 'rb') as f:
    clean_model = pickle.load(f)

with open(noisy_model_path, 'rb') as f:
    noisy_model = pickle.load(f)

# Define Sentiment Mapping
# Note: Model outputs are now 0,1,2,3,4 instead of -2,-1,0,1,2
sentiment_mapping = {
    0: 'Angry',     # shifted from -2
    1: 'Sad',       # shifted from -1
    2: 'Neutral',   # shifted from 0
    3: 'Happy',     # shifted from 1
    4: 'Excited'    # shifted from 2
}

@app.route('/', methods=['GET', 'POST'])
def predict_sentiment():
    prediction_clean = None
    prediction_noisy = None
    
    if request.method == 'POST':
        comment = request.form['comment']

        input_data = {
            'Comment': [comment],
            'Comment_Length': [len(comment)],
            'Likes': [0],  
            'Has_Typo': [0],  
            'Slang_Presence': [0]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Predict using both models
        pred_clean = clean_model.predict(input_df)[0]
        pred_noisy = noisy_model.predict(input_df)[0]
        
        # Map back to sentiment
        prediction_clean = sentiment_mapping.get(pred_clean, "Unknown")
        prediction_noisy = sentiment_mapping.get(pred_noisy, "Unknown")
    
    return render_template('index.html', prediction_clean=prediction_clean, prediction_noisy=prediction_noisy)

if __name__ == '__main__':
    app.run(debug=True)
