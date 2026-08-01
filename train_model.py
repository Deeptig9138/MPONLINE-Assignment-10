import os
import getpass
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data():
    csv_file = 'heart.csv'
    
    # Download if file missing or empty (0 bytes)
    if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
        if os.path.exists(csv_file):
            os.remove(csv_file)
        print("Downloading dataset from Kaggle...")
        os.environ['KAGGLE_USERNAME'] = input("Kaggle Username: ")
        os.environ['KAGGLE_KEY'] = getpass.getpass("Kaggle API Key: ")
        
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files('johnsmith88/heart-disease-dataset', path='.', unzip=True)
    
    df = pd.read_csv(csv_file)
    return df

def main():
    # 1. Load Data
    df = load_data()
    print("Dataset Head:")
    print(df.head())
    
    # 2. Separate Features and Target
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 3. Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Model Development (Random Forest Classifier)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Model Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy Score: {acc:.4f}")
    
    # 6. Save Model Artifact
    joblib.dump(model, 'model.pkl')
    print("Trained model successfully saved as 'model.pkl'.")

if __name__ == '__main__':
    main()
