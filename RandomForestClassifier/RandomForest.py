import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# step 1: collecter et prétraiter les données
def load_data():
    # exemple de données 
    data = {
        'cpu_usage': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'memory_usage': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'storage_usage': [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400],
        'network_usage': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'offloadable': [0, 0, 1, 1, 0, 0, 1, 1, 0, 1]  # 0: Non-Offloadable, 1: Offloadable
    }
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    X = df.drop('offloadable', axis=1)
    y = df['offloadable']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# step 2: entraîner le modèle
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'offload_model.pkl')
    return model

# step 3: évaluer le modèle
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print('Confusion Matrix:')
    print(cm)
    print('Classification Report:')
    print(cr)

# Étape 4: prédictions
def make_prediction(model, new_data):
    prediction = model.predict(new_data)
    return prediction

if __name__ == "__main__":
    # charger et prétraiter les données
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # entraîner le modèle
    model = train_model(X_train, y_train)
    
    # évaluer le modèle
    evaluate_model(model, X_test, y_test)
    
    # faire une prédiction pour un nouveau module
    new_module = np.array([[55, 10, 900, 22]])  # [cpu,memory,storage,network]
    prediction = make_prediction(model, new_module)
    offloadable = 'Offloadable' if prediction[0] == 1 else 'Non-Offloadable'
    print(f'Le module est : {offloadable}')
