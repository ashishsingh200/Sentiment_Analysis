# ================================
# 0. Imports
# ================================
import pandas as pd
import numpy as np
import os
import pickle
import logging

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ================================
# 1. Load Data Safely
# ================================
df = pd.read_csv('../data/processed/processed_sentiments.csv')

logger.info("\n" + str(df[['Sentiment_Score']].head()))

if df['Sentiment_Score'].dtype == object:
    logger.info("Mapping Sentiment labels...")
    sentiment_mapping = {
        'Angry': -2,
        'Sad': -1,
        'Neutral': 0,
        'Happy': 1,
        'Excited': 2
    }
    df['Sentiment_Score'] = df['Sentiment_Score'].map(sentiment_mapping)

df = df.dropna(subset=['Sentiment_Score'])

df['Sentiment_Score'] = df['Sentiment_Score'].astype(int) + 2

# ================================
# 2. Feature Engineering
# ================================
X = df[['Comment', 'Comment_Length', 'Likes', 'Has_Typo', 'Slang_Presence']]
y = df['Sentiment_Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ================================
# 3. Preprocessor
# ================================
text_features = 'Comment'
numeric_features = ['Comment_Length', 'Likes', 'Has_Typo', 'Slang_Presence']

preprocessor = ColumnTransformer(transformers=[
    ('text', TfidfVectorizer(), text_features),
    ('num', 'passthrough', numeric_features)
])

# ================================
# 4. Models
# ================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "LightGBM": LGBMClassifier()
}

trained_models = {}
results = []

# ================================
# 5. Training Loop
# ================================
for model_name, model in models.items():
    logger.info(f"Training {model_name}...")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)

    # Save trained model in dictionary
    trained_models[model_name] = pipeline

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    test_preds = pipeline.predict(X_test)

    result = {
        "Model": model_name,
        "CV_Mean": np.mean(cv_scores),
        "CV_Std": np.std(cv_scores),
        "Test_Accuracy": accuracy_score(y_test, test_preds),
        "Test_F1": f1_score(y_test, test_preds, average='weighted')
    }
    results.append(result)

# ================================
# 6. Model Comparison Table
# ================================
model_comparison = pd.DataFrame(results)
logger.info("\n" + str(model_comparison))

# ================================
# 7. Save Best Model
# ================================
model_comparison_sorted = model_comparison.sort_values(by='CV_Mean', ascending=False).reset_index(drop=True)

best_model_name = model_comparison_sorted.loc[0, 'Model']
best_model_pipeline = trained_models[best_model_name]

save_path = os.path.join('..', 'models', f'best_model_{best_model_name.replace(' ', '_').lower()}.pkl')
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, 'wb') as f:
    pickle.dump(best_model_pipeline, f)

logger.info(f"✅ Best model '{best_model_name}' saved at: {save_path}")
