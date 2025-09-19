# src/evaluate.py
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate(model_path, X_test, y_test):
    data = joblib.load(model_path)
    model = data['model']
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return rmse, r2
