# src/train.py
import os
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

from src.data_utills import load_data, split_target, impute_numeric

def train(path, target='MEDV', out_dir='models'):
    df = load_data(path)
    X, y = split_target(df, target)
    X, imputer = impute_numeric(X, strategy='median')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())])
    params = {'ridge__alpha': [0.01, 0.1, 1, 10, 100]}

    gs = GridSearchCV(pipe, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_

    y_pred = best.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Best params: {gs.best_params_}")
    print(f"Test RMSE: {rmse:.4f}, R2: {r2:.4f}")

    # cross-val on full data for a stable estimate
    scores = -cross_val_score(best, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"CV RMSE: {np.sqrt(scores).mean():.4f} ± {np.sqrt(scores).std():.4f}")

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'best_model.pkl')
    joblib.dump({'model': best, 'imputer': imputer, 'target': target, 'columns': X.columns.tolist()}, model_path)
    print("Saved model to", model_path)
    return model_path

if __name__ == "__main__":
    # example usage
    model_file = train("../data/HousingData.csv")
